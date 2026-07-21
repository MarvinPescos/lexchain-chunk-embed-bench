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

# The model set (pivot 2026-07-21): LexChain needs a SELF-HOSTABLE model, so the
# candidates are 4 Ollama models runnable on commodity GPUs; the NIM-hosted 70B
# is kept ONLY as a reference ceiling and is labeled non-deployable everywhere.
#
# short name -> spec. Short names are used in filenames/results, never shown to
# blind raters.
#   backend      "ollama" (OpenAI-compatible local endpoint) or "nim"
#   id           model tag / API id
#   role         "candidate" or the reference label (surfaced in key + tables)
#   native_ctx   model's maximum context window (tokens)
#   num_ctx      context we request explicitly (Ollama options.num_ctx)
#   strip_think  strip <think>...</think> before parsing (Qwen3 thinking mode)
REFERENCE_ROLE = "reference (non-deployable)"

MODELS = {
    "llama3.1-8b": {
        "backend": "ollama", "id": "llama3.1:8b", "role": "candidate",
        "native_ctx": 131072, "num_ctx": 32768,
    },
    "qwen3-14b": {
        "backend": "ollama", "id": "qwen3:14b", "role": "candidate",
        "native_ctx": 40960, "num_ctx": 32768, "strip_think": True,
    },
    "mistral-nemo-12b": {
        "backend": "ollama", "id": "mistral-nemo:12b", "role": "candidate",
        "native_ctx": 131072, "num_ctx": 32768,
    },
    "gemma3-27b": {
        "backend": "ollama", "id": "gemma3:27b", "role": "candidate",
        "native_ctx": 131072, "num_ctx": 32768,
        # q4 weights ~17GB: hard-asserted before any call (analyze.assert_vram)
        "min_free_vram_gb": 17,
    },
    "llama-3.1-70b": {
        "backend": "nim", "id": "meta/llama-3.1-70b-instruct", "role": REFERENCE_ROLE,
        "native_ctx": 131072, "num_ctx": None,  # hosted; no num_ctx knob
    },
}

# Models considered but excluded at eligibility screening -- recorded in
# models_meta.json and the paper so the selection is auditable.
EXCLUDED_MODELS = {
    "phi4-14b": {
        "id": "phi4:14b",
        "reason": "excluded at eligibility: 16k context cannot hold 2/10 sampled "
                  "documents without truncation; candidates must process every "
                  "document whole.",
        "excluded_on": "2026-07-21",
    },
}

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
