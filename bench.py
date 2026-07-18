#!/usr/bin/env python3
"""Chunking x embedding retrieval benchmark on OHR-Bench Law QAs. Resumable.

Matrix: 3 chunkers (chunkers.py) x 3 embedders. For each config:
chunk all docs -> embed chunks + questions -> cosine top-k -> metrics.

Every stage is cached as an atomic file under --cache-dir and skipped when
present, so a Colab disconnect after any config loses nothing:
    chunks/{chunker}{sfx}.json          chunks + config + timing + stats
    qemb/{embedder}{sfx}.npz            query embeddings (shared across chunkers)
    emb/{chunker}__{embedder}{sfx}.npz  chunk embeddings + encode time
    results/{chunker}__{embedder}{sfx}.json

{sfx} = "_n<N>" when --limit-docs N is set (smoke runs never collide with full).

Metrics per config (see matching.py for the OHR-Bench-mirrored gate):
  recall@1 / recall@5   >=1 relevant chunk in top-k
  mrr@5                 reciprocal rank of first relevant in top-5
  ndcg@10               binary gains, ideal DCG from the full relevant set
  lcs@5                 OHR-Bench's exact continuous metric: word-LCS of the
                        evidence vs the doc+page-gated top-5 concatenation
  coverage              fraction of QAs with >=1 relevant chunk anywhere
                        (a property of the chunker: was evidence preserved?)
Relevant chunk := same doc AND page overlap AND lcs_score >= --relevance-threshold.
All Law QAs count in every denominator (as in OHR-Bench), including the few
chart/formula ones whose evidence may be unmatchable in text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from chunkers import CHUNKERS, chunk_corpus, count_tokens
from matching import evidence_pages, evidence_text, gate_chunks, lcs_score, qa_doc_stem

EMBEDDERS = {
    "e5-base-v2": {
        "model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch": 64,
    },
    "bge-base-en-v1.5": {
        "model": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "batch": 64,
    },
    "minilm-l6-v2": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "batch": 128,
    },
    # torch-free deterministic embedder for local tests: bag-of-words random
    # projection, so cosine similarity tracks word overlap
    "fake": {"model": None, "query_prefix": "", "passage_prefix": "", "batch": 0},
}

DEFAULT_EMBEDDERS = ["e5-base-v2", "bge-base-en-v1.5", "minilm-l6-v2"]


def atomic_write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def log(msg):
    print(f"[bench] {msg}", flush=True)


# ------------------------------------------------------------------ embedding


def fake_embed(texts: list[str], dim: int = 128) -> np.ndarray:
    from matching import normalize_answer

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in set(normalize_answer(text).split()):
            seed = int.from_bytes(hashlib.md5(word.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] += rng.standard_normal(dim).astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


_MODEL_CACHE = {}


def st_encode(embedder: str, texts: list[str], prefix: str) -> tuple[np.ndarray, float]:
    """Returns (normalized float32 embeddings, encode wall seconds)."""
    if embedder == "fake":
        t0 = time.time()
        return fake_embed([prefix + t for t in texts]), time.time() - t0
    spec = EMBEDDERS[embedder]
    if embedder not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        log(f"loading {spec['model']} ...")
        _MODEL_CACHE[embedder] = SentenceTransformer(spec["model"])
    model = _MODEL_CACHE[embedder]
    t0 = time.time()
    emb = model.encode(
        [prefix + t for t in texts],
        batch_size=spec["batch"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32), time.time() - t0


# ------------------------------------------------------------------- stages


def get_chunks(chunker: str, docs, cache: Path, sfx: str) -> dict:
    path = cache / "chunks" / f"{chunker}{sfx}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    log(f"chunking with {chunker} ...")
    chunks, config, wall_s = chunk_corpus(chunker, docs)
    tokens = [count_tokens(c["text"]) for c in chunks]
    data = {
        "chunker": chunker,
        "config": config,
        "wall_s": round(wall_s, 3),
        "n_chunks": len(chunks),
        "tokens_mean": round(statistics.mean(tokens), 1) if tokens else 0,
        "tokens_p50": statistics.median(tokens) if tokens else 0,
        "tokens_max": max(tokens) if tokens else 0,
        "chunks": chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    log(f"{chunker}: {len(chunks)} chunks in {wall_s:.1f}s "
        f"(mean {data['tokens_mean']} tok)")
    return data


def get_query_emb(embedder: str, questions: list[str], cache: Path, sfx: str) -> np.ndarray:
    path = cache / "qemb" / f"{embedder}{sfx}.npz"
    if path.exists():
        data = np.load(path)
        if data["emb"].shape[0] == len(questions):
            return data["emb"]
        log(f"qemb cache {path.name} has wrong count, recomputing")
    emb, secs = st_encode(embedder, questions, EMBEDDERS[embedder]["query_prefix"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, emb=emb, encode_s=np.array(secs))
    log(f"{embedder}: {len(questions)} queries embedded in {secs:.1f}s")
    return emb


def get_chunk_emb(chunker: str, embedder: str, chunk_texts: list[str],
                  cache: Path, sfx: str) -> tuple[np.ndarray, float]:
    path = cache / "emb" / f"{chunker}__{embedder}{sfx}.npz"
    if path.exists():
        data = np.load(path)
        if data["emb"].shape[0] == len(chunk_texts):
            return data["emb"], float(data["encode_s"])
        log(f"emb cache {path.name} has wrong count, recomputing")
    emb, secs = st_encode(embedder, chunk_texts, EMBEDDERS[embedder]["passage_prefix"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, emb=emb, encode_s=np.array(secs))
    log(f"{chunker}x{embedder}: {len(chunk_texts)} chunks embedded in {secs:.1f}s "
        f"({len(chunk_texts) / max(secs, 1e-9):.0f} chunks/s)")
    return emb, secs


# ------------------------------------------------------------------- scoring


def relevant_sets(chunks: list[dict], qas: list[dict], threshold: float):
    """Per QA: the set of relevant chunk ids (doc+page gate, then LCS >= thr)."""
    by_doc = defaultdict(list)
    for i, c in enumerate(chunks):
        by_doc[c["doc"]].append(i)
    rels = []
    for qa in qas:
        candidates = gate_chunks(chunks, qa, by_doc.get(qa_doc_stem(qa), []))
        ev = evidence_text(qa)
        rels.append({i for i in candidates
                     if lcs_score(chunks[i]["text"], ev) >= threshold})
    return rels


def score_config(chunks, qas, rels, chunk_emb, query_emb, top_k):
    sims = query_emb @ chunk_emb.T  # (n_q, n_chunks), cosine (both normalized)
    k = min(top_k, sims.shape[1])
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row_order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
    top = np.take_along_axis(top, row_order, axis=1)

    r1 = r5 = mrr = ndcg = lcs5 = covered = 0.0
    for qi, qa in enumerate(qas):
        rel = rels[qi]
        ranked = [int(c) for c in top[qi]]
        labels = [c in rel for c in ranked]
        if rel:
            covered += 1
        r1 += labels[0]
        r5 += any(labels[:5])
        mrr += next((1.0 / (r + 1) for r, hit in enumerate(labels[:5]) if hit), 0.0)
        dcg = sum(1.0 / math.log2(r + 2) for r, hit in enumerate(labels[:10]) if hit)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(rel), 10)))
        ndcg += dcg / idcg if idcg > 0 else 0.0
        # OHR-Bench's exact retrieval metric on the top-5
        gated = gate_chunks(chunks, qa, ranked[:5])
        if gated:
            lcs5 += lcs_score("\n\n".join(chunks[c]["text"] for c in gated),
                              evidence_text(qa))
    n = len(qas)
    return {
        "recall@1": r1 / n, "recall@5": r5 / n, "mrr@5": mrr / n,
        "ndcg@10": ndcg / n, "lcs@5": lcs5 / n, "coverage": covered / n,
    }


# ----------------------------------------------------------------------- main


def load_docs(data_dir: Path, limit: int | None):
    docs = {}
    for path in sorted((data_dir / "gt" / "law").glob("*.json")):
        pages = sorted(json.loads(path.read_text(encoding="utf-8")),
                       key=lambda p: p.get("page_idx", 0))
        docs[path.stem] = [p.get("text", "") for p in pages]
    if limit:
        docs = dict(sorted(docs.items())[:limit])
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench" if on_colab
                                 else ".cache_bench"))
    ap.add_argument("--chunkers", default=",".join(CHUNKERS))
    ap.add_argument("--embedders", default=",".join(DEFAULT_EMBEDDERS))
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--relevance-threshold", type=float, default=0.7)
    args = ap.parse_args()

    sfx = f"_n{args.limit_docs}" if args.limit_docs else ""
    cache = args.cache_dir
    (cache / "results").mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.data_dir, args.limit_docs)
    if not docs:
        raise SystemExit(f"no gt docs under {args.data_dir} (run download_data.py)")
    stems = set(docs)
    qas = [q for q in json.loads((args.data_dir / "qas_law.json").read_text(encoding="utf-8"))
           if qa_doc_stem(q) in stems]
    questions = [q["questions"] for q in qas]
    log(f"{len(docs)} docs, {len(qas)} QAs, cache={cache}{' sfx=' + sfx if sfx else ''}")

    chunker_names = [c.strip() for c in args.chunkers.split(",") if c.strip()]
    embedder_names = [e.strip() for e in args.embedders.split(",") if e.strip()]

    t_start = time.time()
    for chunker in chunker_names:
        cdata = get_chunks(chunker, docs, cache, sfx)
        chunks = cdata["chunks"]
        texts = [c["text"] for c in chunks]
        rels = None  # computed once per chunker, shared across embedders
        for embedder in embedder_names:
            out_path = cache / "results" / f"{chunker}__{embedder}{sfx}.json"
            if out_path.exists():
                log(f"{chunker}x{embedder}: results cached, skipping")
                continue
            chunk_emb, encode_s = get_chunk_emb(chunker, embedder, texts, cache, sfx)
            query_emb = get_query_emb(embedder, questions, cache, sfx)
            if rels is None:
                log(f"{chunker}: computing relevance sets "
                    f"(threshold {args.relevance_threshold}) ...")
                rels = relevant_sets(chunks, qas, args.relevance_threshold)
            metrics = score_config(chunks, qas, rels, chunk_emb, query_emb, args.top_k)
            result = {
                "chunker": chunker,
                "embedder": embedder,
                "chunker_config": cdata["config"],
                "embedder_model": EMBEDDERS[embedder]["model"] or "fake",
                "relevance_threshold": args.relevance_threshold,
                "n_docs": len(docs),
                "n_questions": len(qas),
                "n_chunks": cdata["n_chunks"],
                "tokens_mean": cdata["tokens_mean"],
                "tokens_p50": cdata["tokens_p50"],
                "chunk_time_s": cdata["wall_s"],
                "embed_s": round(encode_s, 2),
                "chunks_per_s": round(cdata["n_chunks"] / max(encode_s, 1e-9), 1),
                "index_mb": round(chunk_emb.nbytes / 1e6, 2),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            atomic_write_json(out_path, result)
            log(f"{chunker}x{embedder}: R@1 {metrics['recall@1']:.3f} "
                f"R@5 {metrics['recall@5']:.3f} nDCG@10 {metrics['ndcg@10']:.3f} "
                f"LCS@5 {metrics['lcs@5']:.3f}")
    log(f"all requested configs done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
