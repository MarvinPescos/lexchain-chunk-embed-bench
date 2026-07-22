#!/usr/bin/env python3
"""Run the 3-model document analysis over the deterministic 10-doc Law sample.

Model set (see data.MODELS): 3 ~70B-tier instruct models from different families,
all via OpenRouter under a zero-data-retention policy (the single variable is the
model). One frozen prompt (prompt.PROMPT_VERSION), temperature 0, identical schema.

Per (document, model): call the model (constrained JSON decoding + ZDR provider
routing), parse + schema-validate; on parse/validation failure retry with backoff
and log the raw output to <cache>/raw_failures/. Checkpoint every output to
    <cache>/analyses/{doc}__{model}.json
(resumable: existing checkpoints are skipped). A context-safety check runs
BEFORE any call and refuses to run if any document could be truncated.

Latency is wall-clock per call and labeled (default "openrouter_api").

Environment:
  OPENROUTER_API_KEY   required (Colab: add it as a secret)

Usage:
  python -m analysis.analyze --smoke 2                  # 2 docs x all 3 models
  python -m analysis.analyze                            # full 10 x 3 = 30
  python -m analysis.analyze --models llama-3.3-70b,qwen-2.5-72b
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import (
    DEFAULT_SAMPLE_SIZE,
    EXCLUDED_MODELS,
    MAX_OUTPUT_TOKENS,
    MODELS,
    PROVIDER_PREFS,
    context_check,
    load_docs,
    sample_stems,
)
from .prompt import PROMPT_VERSION, build_messages, parse_analysis, validate_schema

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 6        # transport-level (rate limit / 5xx)
PARSE_RETRIES = 2      # additional attempts when output fails parse/schema checks
TEMPERATURE = 0


def log(msg: str) -> None:
    print(f"[analyze] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def get_clients(backends: set[str]) -> dict:
    from openai import OpenAI

    clients = {}
    if "openrouter" in backends:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set (Colab: add it as a secret)")
        clients["openrouter"] = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    return clients


def validate_models(clients: dict, short_names: list[str]) -> None:
    """Assert every requested model is served by its backend; fail with hints."""
    problems = []
    for backend, client in clients.items():
        wanted = {s: MODELS[s]["id"] for s in short_names if MODELS[s]["backend"] == backend}
        if not wanted:
            continue
        try:
            served = [m.id for m in client.models.list().data]
        except Exception as e:
            problems.append(f"{backend}: cannot list models ({e}) -- is the endpoint up?")
            continue
        for s, mid in wanted.items():
            if mid not in served:
                near = difflib.get_close_matches(mid, served, n=3, cutoff=0.3)
                problems.append(f"{backend}: '{mid}' ({s}) not served -- closest: {near}")
    if problems:
        raise SystemExit("Model validation failed:\n  " + "\n  ".join(problems))
    log(f"validated {len(short_names)} models across {sorted(clients)} backends")


def _nvidia_free_vram_gb() -> float | None:
    """Free VRAM of GPU 0 in GB via nvidia-smi, or None if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return int(out[0]) / 1024 if out else None
    except (FileNotFoundError, ValueError, IndexError, Exception):
        return None


def assert_vram(short_names: list[str], free_gb=None) -> None:
    """Hard requirement: models with min_free_vram_gb refuse to run without it
    (CPU offload would silently wreck the latency comparison)."""
    needy = [(s, MODELS[s]["min_free_vram_gb"]) for s in short_names
             if MODELS[s].get("min_free_vram_gb")]
    if not needy:
        return
    free = free_gb if free_gb is not None else _nvidia_free_vram_gb()
    problems = []
    for s, need in needy:
        if free is None:
            problems.append(f"{s} needs >={need}GB free VRAM but no GPU/nvidia-smi "
                            f"was found (CPU offload is not acceptable)")
        elif free < need:
            problems.append(f"{s} needs >={need}GB free VRAM, only {free:.1f}GB free "
                            f"-- use an L4/A100 runtime")
    if problems:
        raise SystemExit("VRAM CHECK FAILED:\n  " + "\n  ".join(problems))
    log(f"VRAM check OK ({free:.1f}GB free) for {[s for s, _ in needy]}")


def collect_models_meta(cache_dir: Path, short_names: list[str]) -> dict:
    """Record exact id, family, tier, context, decoding, and privacy per model."""
    meta_path = cache_dir / "models_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for s in short_names:
        spec = MODELS[s]
        entry = {
            "id": spec["id"],
            "backend": spec["backend"],
            "family": spec.get("family"),
            "role": spec["role"],
            "native_ctx": spec["native_ctx"],
            "tier_note": spec.get("tier_note"),
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt_version": PROMPT_VERSION,
            "json_decoding": "response_format=json_object (constrained)",
            "privacy": dict(PROVIDER_PREFS) if spec["backend"] == "openrouter" else None,
            "recorded_at": now_iso(),
        }
        meta[s] = entry
    meta["_excluded"] = EXCLUDED_MODELS
    atomic_write_json(meta_path, meta)
    return meta


def make_call_fn(clients: dict):
    """call_fn(short_name, messages) -> (text, usage, latency_s) with backoff."""

    def call_fn(short: str, messages: list[dict]):
        spec = MODELS[short]
        client = clients[spec["backend"]]
        kwargs = dict(
            model=spec["id"],
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        if spec["backend"] == "openrouter":
            # constrained JSON decoding + zero-data-retention provider routing
            # (deny data collection, no fallback to a non-compliant provider).
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["extra_body"] = {"provider": dict(PROVIDER_PREFS)}
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = client.chat.completions.create(**kwargs)
                latency = time.time() - t0
                usage = getattr(resp, "usage", None)
                usage_d = (
                    {"prompt_tokens": usage.prompt_tokens,
                     "completion_tokens": usage.completion_tokens}
                    if usage else {}
                )
                # record which provider actually served the call (confirms the
                # DeepInfra pin held; documents provider/quant for the paper)
                usage_d["provider"] = getattr(resp, "provider", None)
                return resp.choices[0].message.content or "", usage_d, latency
            except Exception as e:
                # Permanent client errors (404 no ZDR endpoint, 400/401/403) are
                # deterministic -- don't waste backoff retries; surface at once.
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    raise RuntimeError(f"{spec['id']}: HTTP {status} "
                                       f"(no retry): {getattr(e, 'message', e)}") from e
                last_err = e
                wait = min(2 ** attempt, 30)
                log(f"  {spec['id']}: {type(e).__name__} "
                    f"(attempt {attempt+1}/{MAX_RETRIES}), backoff {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"exhausted retries for {spec['id']}: {last_err}")

    return call_fn


def _log_raw_failure(cache_dir: Path, stem: str, short: str, attempt: int,
                     raw: str, problems: list[str]) -> None:
    fail_dir = cache_dir / "raw_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    (fail_dir / f"{stem}__{short}__attempt{attempt}.txt").write_text(
        f"# problems: {problems}\n{raw}", encoding="utf-8"
    )


def run_analyses(docs: dict[str, str], short_names: list[str], cache_dir: Path,
                 call_fn, latency_label: str = "openrouter_api",
                 resume: bool = True, tolerate_failures: bool = False) -> list[Path]:
    """Analyze each doc with each model, checkpointing every output.

    tolerate_failures: when True (model-probing / smoke), a hard call failure is
    recorded (ok:false, call_failed:true) and the run continues to the next model
    instead of aborting -- so a ZDR-404 model is reported per-model rather than
    killing the whole run. Default False keeps the full run fail-loud."""
    out_dir = cache_dir / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    total = len(docs) * len(short_names)
    i = 0
    for stem, text in docs.items():
        messages = build_messages(text)
        for short in short_names:
            spec = MODELS[short]
            is_reference = spec["role"] != "candidate" or tolerate_failures
            i += 1
            path = out_dir / f"{stem}__{short}.json"
            if resume and path.exists():
                prev = json.loads(path.read_text(encoding="utf-8"))
                # skip completed work (success OR deterministic parse failure);
                # RE-ATTEMPT only transport failures (e.g. reference 403) so a
                # fixed credential is picked up on rerun.
                if not prev.get("call_failed"):
                    log(f"[{i}/{total}] {stem[:40]} x {short}: cached, skip")
                    written.append(path)
                    continue
                log(f"[{i}/{total}] {stem[:40]} x {short}: retrying previously-failed call")
            else:
                log(f"[{i}/{total}] {stem[:40]} x {short}: calling {spec['id']} ...")

            base_rec = {
                "doc": stem, "model": short, "model_id": spec["id"],
                "backend": spec["backend"], "role": spec["role"],
                "prompt_version": PROMPT_VERSION,
                "temperature": TEMPERATURE, "num_ctx": spec["num_ctx"],
                "strip_think": bool(spec.get("strip_think")),
                "latency_label": latency_label, "ts": now_iso(),
            }
            try:
                raw, usage, latency = "", {}, None
                parsed, problems = None, ["not called"]
                for attempt in range(1 + PARSE_RETRIES):
                    raw, usage, latency = call_fn(short, messages)
                    parsed = parse_analysis(raw)  # strips <think> blocks internally
                    problems = validate_schema(parsed) if parsed else ["unparseable JSON"]
                    if not problems:
                        break
                    _log_raw_failure(cache_dir, stem, short, attempt, raw, problems)
                    log(f"    parse/schema attempt {attempt+1} failed: {problems[:2]}")
                    time.sleep(min(2 ** attempt, 10))
                atomic_write_json(path, {
                    **base_rec, "raw": raw, "parsed": parsed,
                    "ok": parsed is not None and not problems,
                    "call_failed": False,
                    "schema_problems": problems if problems else [],
                    "latency_s": round(latency, 3) if latency is not None else None,
                    "usage": usage,
                })
                status = "ok" if (parsed and not problems) else f"PROBLEMS {problems[:1]}"
                log(f"    -> {status} ({latency:.1f}s, "
                    f"{usage.get('completion_tokens', '?')} out tok)")
            except Exception as e:
                # A hard failure (incl. a ZDR routing 404 under allow_fallbacks:
                # false) fails loudly by default rather than leaking to a
                # non-compliant provider. With tolerate_failures (model-probing),
                # it is recorded per-model and the run continues.
                if not is_reference:
                    raise
                atomic_write_json(path, {
                    **base_rec, "raw": "", "parsed": None, "ok": False,
                    "call_failed": True, "error": f"{type(e).__name__}: {e}",
                    "schema_problems": ["call failed"],
                    "latency_s": None, "usage": {},
                })
                log(f"    -> CALL FAILED ({type(e).__name__}); recorded ok:false, "
                    f"continuing. Fix and rerun to retry it.")
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
    ap.add_argument("--smoke", type=int, default=None, metavar="N")
    ap.add_argument("--latency-label", default="openrouter_api")
    ap.add_argument("--skip-context-check", action="store_true",
                    help="NOT recommended; the check prevents silent truncation")
    ap.add_argument("--skip-vram-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tolerate-failures", action="store_true",
                    help="record per-model call failures (e.g. ZDR-404) and continue "
                         "instead of aborting -- use when probing which models route")
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
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.cache_dir / "sample.json",
                      {"seed": 42, "n": len(stems), "stems": stems})
    log(f"{len(docs)} docs x {len(short_names)} models = "
        f"{len(docs) * len(short_names)} calls | prompt {PROMPT_VERSION}")

    if not args.skip_context_check:
        report = context_check(docs, short_names)
        log(f"context check OK (prompt overhead "
            f"{report['prompt_overhead_tokens']} tok; all docs fit all models)")
    if not args.skip_vram_check:
        assert_vram(short_names)

    backends = {MODELS[s]["backend"] for s in short_names}
    clients = get_clients(backends)
    validate_models(clients, short_names)
    collect_models_meta(args.cache_dir, short_names)
    call_fn = make_call_fn(clients)

    t0 = time.time()
    run_analyses(docs, short_names, args.cache_dir, call_fn,
                 latency_label=args.latency_label,
                 tolerate_failures=args.tolerate_failures)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {args.cache_dir}/analyses")

    # per-model status summary (ZDR routing pass/fail is visible here)
    log("per-model status:")
    for short in short_names:
        recs = [json.loads(p.read_text()) for p in
                (args.cache_dir / "analyses").glob(f"*__{short}.json")]
        ok = sum(r.get("ok") for r in recs)
        failed = [r for r in recs if r.get("call_failed")]
        if failed:
            err = failed[0].get("error", "")
            zdr = "ZDR/data-policy 404" if "404" in err else "call failed"
            log(f"  {short:16s} {ok}/{len(recs)} ok  --  {zdr}: {err[:90]}")
        else:
            log(f"  {short:16s} {ok}/{len(recs)} ok")


if __name__ == "__main__":
    main()
