#!/usr/bin/env python3
"""Run the 3-model document analysis over a deterministic 10-doc Law sample.

For each (document, model): send the ONE fixed prompt (prompt.py) to the model
via NVIDIA NIM (OpenAI-compatible), parse the JSON analysis, and checkpoint to
    <cache>/analyses/{doc}__{model}.json   {model_id, raw, parsed, latency_s, usage, ok}
Resumable (existing checkpoints are skipped). Rate limits / transient errors are
retried with exponential backoff. Model IDs are validated against the live
/v1/models catalog at startup.

Usage (Colab or local with NVIDIA_API_KEY set):
  python -m analysis.analyze --smoke 2      # 2 docs x 3 models (smoke)
  python -m analysis.analyze                # full 10 docs x 3 models
  python -m analysis.analyze --models llama-3.1-70b,qwen2.5-72b
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from pathlib import Path

from .data import MODELS, DEFAULT_SAMPLE_SIZE, load_docs, sample_stems
from .prompt import build_messages, parse_analysis

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# optional --provider groq escape hatch; Groq serves no equivalent of the two
# NIM comparators, so only the deployed model has a mapping. NIM is canonical.
GROQ_FALLBACK = {
    "llama-3.1-70b": "llama-3.3-70b-versatile",
}
MAX_RETRIES = 6


def log(msg: str) -> None:
    print(f"[analyze] {msg}", flush=True)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def get_client(provider: str):
    from openai import OpenAI

    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY not set")
        return OpenAI(base_url=GROQ_BASE_URL, api_key=key)
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise SystemExit("NVIDIA_API_KEY not set (Colab: add it as a secret)")
    return OpenAI(base_url=NIM_BASE_URL, api_key=key)


def resolve_model_ids(provider: str, short_names: list[str]) -> dict[str, str]:
    if provider == "groq":
        missing = [s for s in short_names if s not in GROQ_FALLBACK]
        if missing:
            raise SystemExit(f"no Groq equivalent for {missing}; use --provider nim "
                             f"(Groq fallback covers only {list(GROQ_FALLBACK)})")
        return {s: GROQ_FALLBACK[s] for s in short_names}
    return {s: MODELS[s] for s in short_names}


def validate_models(client, model_ids: dict[str, str]) -> None:
    """Assert every requested model id is served; else show closest matches + exit."""
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:  # network/catalog issue shouldn't hard-block the run
        log(f"WARN: could not list /v1/models ({e}); skipping validation")
        return
    missing = {s: mid for s, mid in model_ids.items() if mid not in served}
    if missing:
        lines = ["Some requested models are NOT served by this provider:"]
        for s, mid in missing.items():
            near = difflib.get_close_matches(mid, served, n=3, cutoff=0.3)
            lines.append(f"  {s} -> '{mid}' NOT FOUND. closest: {near or '(none)'}")
        lines.append(f"({len(served)} models available; e.g. {served[:5]})")
        raise SystemExit("\n".join(lines))
    log(f"validated {len(model_ids)} model ids against {len(served)} served models")


def make_nim_call_fn(client):
    """Returns call_fn(model_id, messages) -> (text, usage_dict, latency_s) with backoff."""

    def call_fn(model_id: str, messages: list[dict]):
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0,
                    max_tokens=2048,
                )
                latency = time.time() - t0
                usage = getattr(resp, "usage", None)
                usage_d = (
                    {"prompt_tokens": usage.prompt_tokens,
                     "completion_tokens": usage.completion_tokens}
                    if usage else {}
                )
                return resp.choices[0].message.content or "", usage_d, latency
            except Exception as e:  # rate limit / 5xx / transient
                last_err = e
                wait = min(2 ** attempt, 30)
                log(f"  {model_id}: {type(e).__name__} (attempt {attempt+1}/{MAX_RETRIES}), backoff {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"exhausted retries for {model_id}: {last_err}")

    return call_fn


def run_analyses(docs: dict[str, str], model_ids: dict[str, str], cache_dir: Path,
                 call_fn, resume: bool = True) -> list[Path]:
    """Analyze each doc with each model, checkpointing every output. Returns paths."""
    out_dir = cache_dir / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    total = len(docs) * len(model_ids)
    i = 0
    for stem, text in docs.items():
        messages = build_messages(text)
        for short, model_id in model_ids.items():
            i += 1
            path = out_dir / f"{stem}__{short}.json"
            if resume and path.exists():
                log(f"[{i}/{total}] {stem} x {short}: cached, skip")
                written.append(path)
                continue
            log(f"[{i}/{total}] {stem} x {short}: calling {model_id} ...")
            raw, usage, latency = call_fn(model_id, messages)
            parsed = parse_analysis(raw)
            atomic_write_json(path, {
                "doc": stem, "model": short, "model_id": model_id,
                "raw": raw, "parsed": parsed, "ok": parsed is not None,
                "latency_s": round(latency, 3), "usage": usage,
            })
            status = "ok" if parsed else "UNPARSEABLE"
            log(f"    -> {status} ({latency:.1f}s, {usage.get('completion_tokens','?')} out tok)")
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit-docs", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="quick run on N docs (e.g. 2) x all models")
    ap.add_argument("--provider", choices=["nim", "groq"], default="nim")
    args = ap.parse_args()

    short_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [s for s in short_names if s not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model short-names {unknown}; valid: {list(MODELS)}")

    all_docs = load_docs()
    if not all_docs:
        raise SystemExit("no GT docs found (run download_data.py first)")
    n = args.smoke if args.smoke is not None else args.limit_docs
    stems = sample_stems(all_docs.keys(), n=n)
    docs = {s: all_docs[s] for s in stems}
    (args.cache_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.cache_dir / "sample.json",
                      {"seed": 42, "n": len(stems), "stems": stems})
    log(f"{len(docs)} docs x {len(short_names)} models = {len(docs)*len(short_names)} calls")
    log(f"sample stems: {stems}")

    client = get_client(args.provider)
    model_ids = resolve_model_ids(args.provider, short_names)
    validate_models(client, model_ids)
    call_fn = make_nim_call_fn(client)

    t0 = time.time()
    run_analyses(docs, model_ids, args.cache_dir, call_fn)
    log(f"done in {(time.time()-t0)/60:.1f} min -> {args.cache_dir}/analyses")


if __name__ == "__main__":
    main()
