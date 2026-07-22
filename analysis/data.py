"""Shared data loading + deterministic sampling for the analysis comparison.

Reuses the OHR-Bench Law ground truth already fetched by the repo's
download_data.py into data/gt/law/*.json (each file: list of {"text": ...} pages).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GT_DIR = REPO / "data" / "gt" / "law"

SAMPLE_SEED = 42
DEFAULT_SAMPLE_SIZE = 10
DOC_CHAR_CAP = 96_000  # ~24k tokens safety cap; law docs are short (~3 pages)

# The model set (reframed 2026-07-21): the single variable is the model. All
# three are ~70B-tier instruct models from different families, accessed through
# the SAME API path (OpenRouter) under a zero-data-retention policy -- API-only
# deployment, no self-hosting, which removes the size-bias confound.
#
# short name -> spec. Short names are used in filenames/results, never shown to
# blind raters.
#   backend      "openrouter" (OpenAI-compatible; the only backend now)
#   id           OpenRouter model id (confirmed live in the catalog)
#   role         "candidate" (all three are peers; no reference ceiling)
#   native_ctx   model's maximum context window (tokens); num_ctx N/A (hosted)
#   tier_note    capability-tier caveat for the paper
REFERENCE_ROLE = "reference (non-deployable)"  # retained; unused in this design

MODELS = {
    "llama-3.3-70b": {
        "backend": "openrouter", "id": "meta-llama/llama-3.3-70b-instruct",
        "role": "candidate", "family": "Meta", "native_ctx": 131072, "num_ctx": None,
        "tier_note": "70B dense",
    },
    "qwen-2.5-72b": {
        "backend": "openrouter", "id": "qwen/qwen-2.5-72b-instruct",
        "role": "candidate", "family": "Qwen", "native_ctx": 32768, "num_ctx": None,
        "tier_note": "72B dense (tightest context, 32k)",
    },
    "mixtral-8x22b": {
        "backend": "openrouter", "id": "mistralai/mixtral-8x22b-instruct",
        "role": "candidate", "family": "Mistral", "native_ctx": 65536, "num_ctx": None,
        "tier_note": "MoE 141B total / ~39B active; ~70B-class compute, older (Apr 2024)",
    },
}

# OpenRouter provider routing: zero data retention, no data collection, and NO
# fallback to a non-compliant provider (fail loudly rather than leak). Applied to
# every request via extra_body["provider"]. (Prompt logging is additionally an
# account-level OpenRouter privacy setting the user must disable.)
PROVIDER_PREFS = {"data_collection": "deny", "allow_fallbacks": False, "zdr": True}

# No eligibility exclusions in this design (API-only, all tier-matched).
EXCLUDED_MODELS: dict = {}

MAX_OUTPUT_TOKENS = 2048
CTX_SAFETY_MARGIN = 512


def load_docs(gt_dir: Path = GT_DIR) -> dict[str, str]:
    """stem -> full document text (pages joined)."""
    docs = {}
    for path in sorted(gt_dir.glob("*.json")):
        pages = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(p.get("text", "") for p in pages).strip()
        if text:
            docs[path.stem] = text[:DOC_CHAR_CAP]
    return docs


def sample_stems(all_stems, n: int = DEFAULT_SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    """Deterministic reproducible sample of document stems."""
    stems = sorted(all_stems)
    if n >= len(stems):
        return stems
    return sorted(random.Random(seed).sample(stems, n))


def context_check(docs: dict[str, str], model_names=None) -> dict:
    """Context-safety guard (fail loudly, never silently truncate).

    Tokenizes every document (cl100k_base as a cross-model estimate) plus the
    frozen prompt overhead and output budget, and asserts the total fits each
    model's requested num_ctx (and native_ctx). Raises SystemExit listing every
    (model, doc) that does not fit. Returns a report dict when all fit.
    """
    import tiktoken

    from .prompt import prompt_overhead_text

    enc = tiktoken.get_encoding("cl100k_base")
    overhead = len(enc.encode(prompt_overhead_text(), disallowed_special=()))
    doc_tokens = {s: len(enc.encode(t, disallowed_special=())) for s, t in docs.items()}
    budget_extra = overhead + MAX_OUTPUT_TOKENS + CTX_SAFETY_MARGIN

    names = model_names or list(MODELS)
    failures, per_model = [], {}
    for name in names:
        spec = MODELS[name]
        limit = spec["num_ctx"] or spec["native_ctx"]
        worst = max(doc_tokens.values()) + budget_extra
        per_model[name] = {"limit": limit, "worst_case_tokens": worst}
        for stem, dt in doc_tokens.items():
            need = dt + budget_extra
            if need > limit:
                failures.append(
                    f"  {name} (ctx {limit:,}): doc '{stem[:50]}' needs ~{need:,} tokens "
                    f"({dt:,} doc + {overhead} prompt + {MAX_OUTPUT_TOKENS} output "
                    f"+ {CTX_SAFETY_MARGIN} margin)"
                )
    if failures:
        raise SystemExit(
            "CONTEXT CHECK FAILED — refusing to run (documents are never truncated):\n"
            + "\n".join(failures)
            + "\nFix: use a model with a larger context, or change the model set."
        )
    return {"prompt_overhead_tokens": overhead, "doc_tokens": doc_tokens,
            "per_model": per_model}
