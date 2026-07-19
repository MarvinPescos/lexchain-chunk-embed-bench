#!/usr/bin/env python3
"""Build the blind human-evaluation scaffold (Layer 2 of the evaluation).

Produces two files:
  blind_eval.csv      one row per (doc, approach) = 13 x 3 = 39 rows, with the
                      approach HIDDEN. Per document the three summaries are
                      randomly permuted and given neutral labels "Summary 1/2/3",
                      so a rater cannot learn a global "variant X = approach Y"
                      mapping. doc_reference stays visible: raters must open the
                      source document to judge coverage and factual accuracy.
                      Blank rating columns: coverage, factual_accuracy, fluency
                      (each 1-5) and notes.
  unblinding_key.csv  presentation_id, doc_reference, variant -> approach.
                      Kept separate so the ratings stay blind.

Seeded (42) so the blinding is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

RATING_HELP = {
    "coverage": "1-5: are the key parties, dates, and obligations captured?",
    "factual_accuracy": "1-5: any hallucinations or factual errors vs the source?",
    "fluency": "1-5: is it well written and readable?",
}
BLIND_SEED = 42


def load_summaries(cache: Path) -> dict[str, dict[str, str]]:
    """doc -> {approach: summary}."""
    by_doc: dict[str, dict[str, str]] = {}
    for path in sorted((cache / "summaries").glob("*__*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        by_doc.setdefault(rec["doc"], {})[rec["approach"]] = rec["summary"]
    return by_doc


def build(cache: Path, seed: int = BLIND_SEED):
    by_doc = load_summaries(cache)
    if not by_doc:
        raise SystemExit(f"no summaries under {cache}/summaries (run summarize.py)")
    rng = random.Random(seed)

    blind_rows, key_rows = [], []
    pid = 0
    for doc in sorted(by_doc):
        approaches = sorted(by_doc[doc])  # deterministic before shuffle
        rng.shuffle(approaches)  # per-doc random order -> no learnable global pattern
        for slot, approach in enumerate(approaches, start=1):
            pid += 1
            presentation_id = f"P{pid:03d}"
            variant = f"Summary {slot}"
            blind_rows.append({
                "presentation_id": presentation_id,
                "doc_reference": doc,
                "variant": variant,
                "summary": by_doc[doc][approach],
                "coverage": "", "factual_accuracy": "", "fluency": "", "notes": "",
            })
            key_rows.append({
                "presentation_id": presentation_id,
                "doc_reference": doc,
                "variant": variant,
                "approach": approach,
            })

    blind_path = cache / "blind_eval.csv"
    with open(blind_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "presentation_id", "doc_reference", "variant", "summary",
            "coverage", "factual_accuracy", "fluency", "notes"])
        w.writeheader()
        w.writerows(blind_rows)

    key_path = cache / "unblinding_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "presentation_id", "doc_reference", "variant", "approach"])
        w.writeheader()
        w.writerows(key_rows)

    print(f"wrote {blind_path} ({len(blind_rows)} rows) and {key_path}")
    print("Rating columns (1-5), fill coverage / factual_accuracy / fluency + notes:")
    for k, v in RATING_HELP.items():
        print(f"  {k}: {v}")
    print("Approach labels are hidden; do NOT open unblinding_key.csv until ratings are done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/summarization"
                                 if on_colab else ".cache_summ"))
    ap.add_argument("--seed", type=int, default=BLIND_SEED)
    args = ap.parse_args()
    build(args.cache_dir, args.seed)


if __name__ == "__main__":
    main()
