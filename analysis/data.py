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

# The three models under comparison (cross-family, one provider = NVIDIA NIM).
# short name -> API model id. Short names are used only in filenames/results,
# never shown to blind raters.
MODELS = {
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "qwen2.5-72b": "qwen/qwen2.5-72b-instruct",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1",
}


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
