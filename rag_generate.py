#!/usr/bin/env python3
"""End-to-end RAG generation benchmark: fixed pipeline, vary only the LLM.

Pipeline is frozen to the chunk/embed winners: LangChain RecursiveCharacterText-
Splitter (512/50) + e5-base-v2, reusing that benchmark's cached embeddings when
present. For each OHR-Bench Law QA: retrieve top-k chunks, prompt the model,
score the answer with OHR-Bench's own F1/EM (ohr_gen_eval.py).

Resumable: every answered question is appended to
    {cache}/gen_results/{model_slug}{sfx}.jsonl
atomically, and a rerun skips qids already present. A daily rate-limit (429)
flushes cleanly and moves on; rerunning the next day continues.

  --smoke K       first K of a seeded stratified draw (sfx=_smokeK)
  --sample N      seeded stratified sample of N (sfx=_nN), shared by all models
  (neither)       full 1,142 (sfx="")
  --models a,b,c  Groq model IDs (default: the durable post-Llama set)
  --k             retrieved chunks per question (default 5)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench import EMBEDDERS, get_chunk_emb, get_chunks, get_query_emb, load_docs
from groq_client import DailyLimitError, GroqClient, MockGroqClient
from matching import qa_doc_stem
from ohr_gen_eval import (
    QA_PROMPT,
    em_contains,
    exact_match_score,
    extract_response,
    f1_score,
)

DEFAULT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
CHUNKER = "langchain"
EMBEDDER = "e5-base-v2"
SAMPLE_SEED = 42


def log(msg):
    print(f"[rag-gen] {msg}", flush=True)


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def stratified_indices(qas: list[dict], n: int, seed: int) -> list[int]:
    """Proportional sample of qa indices stratified by evidence_source.

    Deterministic for a given (n, seed); guarantees >=1 per present source when
    n allows. Returns indices into `qas`, sorted for stable output.
    """
    if n >= len(qas):
        return list(range(len(qas)))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, qa in enumerate(qas):
        groups[qa.get("evidence_source", "unknown")].append(i)
    rng = random.Random(seed)
    order = sorted(groups)  # deterministic group order
    # largest-remainder allocation so counts sum exactly to n
    raw = {g: len(groups[g]) / len(qas) * n for g in order}
    alloc = {g: min(len(groups[g]), int(raw[g])) for g in order}
    # ensure at least 1 from each present source (budget permitting)
    for g in order:
        if alloc[g] == 0 and groups[g]:
            alloc[g] = 1
    while sum(alloc.values()) > n:  # trim from largest groups
        g = max(order, key=lambda x: alloc[x])
        alloc[g] -= 1
    remainders = sorted(order, key=lambda g: raw[g] - int(raw[g]), reverse=True)
    j = 0
    while sum(alloc.values()) < n:
        g = remainders[j % len(remainders)]
        if alloc[g] < len(groups[g]):
            alloc[g] += 1
        j += 1
    picked = []
    for g in order:
        idx = groups[g][:]
        rng.shuffle(idx)
        picked.extend(idx[:alloc[g]])
    return sorted(picked)


def build_sample(qas, cache: Path, mode: str, n: int):
    """Returns (indices, sfx). Caches the sample so all models share it."""
    if mode == "full":
        return list(range(len(qas))), ""
    sfx = f"_smoke{n}" if mode == "smoke" else f"_n{n}"
    path = cache / f"gen_sample{sfx}.json"
    if path.exists():
        data = json.loads(path.read_text())
        by_id = {qa["ID"]: i for i, qa in enumerate(qas)}
        return [by_id[q] for q in data["qids"] if q in by_id], sfx
    idx = stratified_indices(qas, n, SAMPLE_SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"mode": mode, "n": n, "seed": SAMPLE_SEED,
         "qids": [qas[i]["ID"] for i in idx]}, indent=1))
    log(f"built {mode} sample of {len(idx)} -> {path.name}")
    return idx, sfx


def load_done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["qid"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_jsonl(path: Path, record: dict):
    """Append one record durably (flush+fsync) so a crash can't corrupt prior lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def retrieve(query_emb, chunk_emb, chunks, qi, k):
    sims = chunk_emb @ query_emb[qi]
    kk = min(k, sims.shape[0])
    top = np.argpartition(-sims, kth=kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    return [chunks[int(c)] for c in top]


def run(qas, indices, models, sfx, cache, client, query_emb, chunk_emb, chunks, k):
    results_dir = cache / "gen_results"
    exhausted = []
    for model in models:
        out = results_dir / f"{model_slug(model)}{sfx}.jsonl"
        done = load_done_qids(out)
        todo = [i for i in indices if qas[i]["ID"] not in done]
        log(f"{model}: {len(done)} done, {len(todo)} to do -> {out.name}")
        for count, qi in enumerate(todo, 1):
            qa = qas[qi]
            retrieved = retrieve(query_emb, chunk_emb, chunks, qi, k)
            context = "\n\n".join(c["text"] for c in retrieved)
            user = f"Question: {qa['questions']}\n\nRetrieved Documents: {context}"
            try:
                res = client.chat(model, QA_PROMPT, user)
            except DailyLimitError as e:
                log(f"{model}: daily limit hit after {count - 1} new answers "
                    f"(retry-after ~{e.retry_after / 60:.0f}m); flushed, will resume")
                exhausted.append(model)
                break
            answer = extract_response(res.text)
            gt = qa["answers"]
            append_jsonl(out, {
                "qid": qa["ID"],
                "model": model,
                "question": qa["questions"],
                "ground_truth": gt,
                "answer": answer,
                "raw": res.text,
                "evidence_source": qa.get("evidence_source"),
                "answer_form": qa.get("answer_form"),
                "doc": qa_doc_stem(qa),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "latency_s": round(res.latency_s, 3),
                "f1": round(f1_score(answer, gt), 4),
                "em": exact_match_score(answer, gt),
                "em_contains": em_contains(answer, gt),
            })
            if count % 10 == 0 or count == len(todo):
                rem = res.ratelimit.get("x-ratelimit-remaining-tokens", "?")
                log(f"{model}: {count}/{len(todo)} (tok-remaining today: {rem})")
    return exhausted


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench" if on_colab
                                 else ".cache_bench"))
    ap.add_argument("--embed-cache-dir", type=Path, default=None,
                    help="where the chunk/embed npz cache lives (default: --cache-dir)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--mock", action="store_true", help="offline MockGroqClient (tests)")
    args = ap.parse_args()

    embed_cache = args.embed_cache_dir or args.cache_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    docs = load_docs(args.data_dir, None)
    if not docs:
        raise SystemExit(f"no gt docs under {args.data_dir} (run download_data.py)")
    stems = set(docs)
    qas = [q for q in json.loads((args.data_dir / "qas_law.json").read_text())
           if qa_doc_stem(q) in stems]
    questions = [q["questions"] for q in qas]

    cdata = get_chunks(CHUNKER, docs, embed_cache, "")
    chunks = cdata["chunks"]
    chunk_emb, _ = get_chunk_emb(CHUNKER, EMBEDDER, [c["text"] for c in chunks],
                                 embed_cache, "")
    query_emb = get_query_emb(EMBEDDER, questions, embed_cache, "")
    log(f"{len(docs)} docs, {len(qas)} QAs, {len(chunks)} chunks, "
        f"pipeline={CHUNKER}+{EMBEDDER}, k={args.k}")

    mode = "smoke" if args.smoke else "sample" if args.sample else "full"
    n = args.smoke or args.sample or len(qas)
    indices, sfx = build_sample(qas, args.cache_dir, mode, n)

    if args.mock:
        client = MockGroqClient()
    else:
        client = GroqClient(os.environ.get("GROQ_API_KEY", ""),
                            max_tokens=args.max_tokens)

    t0 = time.time()
    exhausted = run(qas, indices, models, sfx, args.cache_dir, client,
                    query_emb, chunk_emb, chunks, args.k)
    log(f"run finished in {(time.time() - t0) / 60:.1f} min "
        f"(sfx='{sfx or 'full'}', {len(indices)} questions x {len(models)} models)")
    if exhausted:
        log(f"daily limit hit for: {exhausted} -- rerun the same command tomorrow "
            f"to resume (answered questions are skipped).")


if __name__ == "__main__":
    main()
