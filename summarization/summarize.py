#!/usr/bin/env python3
"""Generate summaries of OHR-Bench Law documents with three approaches.

Approaches (the only variable in this experiment):
  textrank  - extractive, sumy TextRankSummarizer (PageRank over sentence graph)
  lexrank   - extractive, sumy LexRankSummarizer (TF-IDF cosine centrality)
  llm       - abstractive, Llama 3.1 70B via NVIDIA NIM (OpenAI-compatible),
              fixed prompt; falls back to Groq llama-3.3-70b if LLM_PROVIDER=groq

Docs are sampled deterministically (seed 42) so the set is reproducible.
Resumable: each summary is checkpointed to <cache>/summaries/{doc}__{approach}.json
and skipped if it already exists, so a Colab disconnect loses nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

SENTENCE_COUNT = 8  # extractive target; LLM prompt targets a comparable length
SAMPLE_SEED = 42
SAMPLE_N = 13

APPROACHES = ["textrank", "lexrank", "llm"]

LLM_PROMPT = (
    "You are summarizing a legal document for a document-analysis system. "
    "Write a faithful, self-contained summary in about 8 sentences (~180 words). "
    "Capture the key parties, dates, obligations, and any amounts or conditions. "
    "Do not invent facts that are not in the document. Return only the summary.\n\n"
    "DOCUMENT:\n{document}\n\nSUMMARY:"
)

LLM_PROVIDERS = {
    "nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-70b-instruct",
        "key_env": "NVIDIA_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
}


# ----------------------------------------------------------------- data / io


def load_docs(data_dir: Path) -> dict[str, str]:
    """stem -> full document text ("\\n".join of page texts)."""
    docs = {}
    for path in sorted((data_dir / "gt" / "law").glob("*.json")):
        pages = sorted(json.loads(path.read_text(encoding="utf-8")),
                       key=lambda p: p.get("page_idx", 0))
        docs[path.stem] = "\n".join(p.get("text", "") for p in pages)
    return docs


def sample_docs(docs: dict[str, str], n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> list[str]:
    stems = sorted(docs)
    rng = random.Random(seed)
    return sorted(rng.sample(stems, min(n, len(stems))))


def atomic_write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# -------------------------------------------------------------- summarizers


def _sumy_summary(text: str, which: str, sentence_count: int = SENTENCE_COUNT) -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer() if which == "textrank" else LexRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


_LLM_CLIENT = {}


def _llm_summary(text: str, provider: str) -> str:
    from openai import OpenAI

    spec = LLM_PROVIDERS[provider]
    if provider not in _LLM_CLIENT:
        key = os.environ.get(spec["key_env"])
        if not key:
            raise RuntimeError(f"{spec['key_env']} not set (needed for LLM_PROVIDER={provider})")
        _LLM_CLIENT[provider] = OpenAI(base_url=spec["base_url"], api_key=key)
    client = _LLM_CLIENT[provider]
    last_err = None
    for attempt in range(6):  # backoff for the 40 RPM free tier / transient errors
        try:
            resp = client.chat.completions.create(
                model=spec["model"],
                messages=[{"role": "user", "content": LLM_PROMPT.format(document=text)}],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # rate limit / network; exponential backoff
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"  LLM call failed ({type(e).__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM failed after retries: {last_err}")


def summarize_one(text: str, approach: str, provider: str = "nim",
                  llm_fn=None) -> tuple[str, dict]:
    """Returns (summary, meta). `llm_fn` overrides the LLM call (used by tests)."""
    t0 = time.time()
    if approach in ("textrank", "lexrank"):
        summary = _sumy_summary(text, approach)
        model = f"sumy.{approach}(sentence_count={SENTENCE_COUNT})"
    elif approach == "llm":
        summary = llm_fn(text) if llm_fn else _llm_summary(text, provider)
        model = "fake-llm" if llm_fn else LLM_PROVIDERS[provider]["model"]
    else:
        raise ValueError(f"unknown approach: {approach}")
    meta = {"approach": approach, "model": model, "latency_s": round(time.time() - t0, 3),
            "summary_words": len(summary.split())}
    return summary, meta


# ----------------------------------------------------------------------- run


def run(docs: dict[str, str], stems: list[str], approaches: list[str],
        cache: Path, provider: str = "nim", llm_fn=None):
    out_dir = cache / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for approach in approaches:
            path = out_dir / f"{stem}__{approach}.json"
            if path.exists():
                print(f"  skip (cached): {stem} / {approach}", flush=True)
                continue
            summary, meta = summarize_one(docs[stem], approach, provider, llm_fn)
            record = {
                "doc": stem, "source_words": len(docs[stem].split()),
                "summary": summary, **meta,
            }
            atomic_write_json(path, record)
            print(f"  done: {stem} / {approach} "
                  f"({meta['summary_words']}w, {meta['latency_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/summarization"
                                 if on_colab else ".cache_summ"))
    ap.add_argument("--approaches", default=",".join(APPROACHES))
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "nim"),
                    choices=list(LLM_PROVIDERS))
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="only the first N sampled docs (smoke test)")
    args = ap.parse_args()

    docs = load_docs(args.data_dir)
    if not docs:
        raise SystemExit(f"no docs under {args.data_dir}/gt/law (run download_data.py)")
    stems = sample_docs(docs)
    if args.limit_docs:
        stems = stems[: args.limit_docs]
    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "sample_docs.txt").write_text("\n".join(stems) + "\n")
    print(f"{len(stems)} docs x {len(approaches)} approaches "
          f"(provider={args.provider}), cache={args.cache_dir}")
    print("sampled docs:", ", ".join(stems))
    run(docs, stems, approaches, args.cache_dir, args.provider)
    print("summaries complete.")


if __name__ == "__main__":
    main()
