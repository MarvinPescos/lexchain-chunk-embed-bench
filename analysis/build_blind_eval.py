#!/usr/bin/env python3
"""Build the blind human-scoring scaffold from the cached analyses.

Outputs (in <cache>):
  blind_eval.csv             30 rows (10 docs x 3 models); model identity hidden
                             behind a per-doc random label (Output A/B/C), rows
                             shuffled. Blank human-rating columns to fill.
  unblinding_key.csv         presentation_id -> doc, model, label (kept SEPARATE)
  ground_truth_key_template.csv  10 rows (one per doc); blank true-entity/risk
                             columns to fill BEFORE scoring, plus source text.

Seeded (42) so the blinding + ordering are reproducible.
Do NOT open unblinding_key.csv until ratings are finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .data import load_docs
from .prompt import ENTITY_CATEGORIES

BLIND_SEED = 42
LABELS = ["Output A", "Output B", "Output C"]

HUMAN_COLS = [
    "summary_coverage_1to5",
    "summary_accuracy_1to5",
    "summary_fluency_1to5",
    "entities_correct_1to5",
    "entities_missed_notes",
    "risk_flags_correct_1to5",
    "hallucinations_notes",
    "overall_notes",
]

GT_KEY_COLS = ["parties", "dates", "monetary_amounts", "obligations", "risk_clauses"]


def _fmt_entities(entities: dict) -> str:
    parts = []
    for cat in ENTITY_CATEGORIES:
        vals = entities.get(cat) or []
        parts.append(f"{cat}: " + ("; ".join(vals) if vals else "(none)"))
    return "\n".join(parts)


def _fmt_risks(risk_flags: list) -> str:
    if not risk_flags:
        return "(none)"
    return "\n".join(
        f"- [{(f.get('severity') or '?')}] {f.get('risk','')}" for f in risk_flags
    )


def load_analyses(cache_dir: Path) -> dict[str, dict[str, dict]]:
    """doc -> model -> parsed analysis (from checkpoints)."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted((cache_dir / "analyses").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        parsed = rec.get("parsed") or {"summary": "", "entities": {}, "risk_flags": []}
        out.setdefault(rec["doc"], {})[rec["model"]] = parsed
    return out


def build(cache_dir: Path) -> tuple[int, int]:
    analyses = load_analyses(cache_dir)
    if not analyses:
        raise SystemExit(f"no analyses in {cache_dir}/analyses (run analyze.py first)")
    docs_text = load_docs()
    rng = random.Random(BLIND_SEED)

    blind_rows, key_rows = [], []
    for doc in sorted(analyses):
        models = sorted(analyses[doc])  # deterministic before shuffle
        order = models[:]
        rng.shuffle(order)  # per-doc random assignment of labels -> models
        for label, model in zip(LABELS, order):
            pres_id = f"{doc[:12]}__{label.split()[-1]}"
            analysis = analyses[doc][model]
            blind_rows.append({
                "presentation_id": pres_id,
                "doc_reference": doc,
                "output_label": label,
                "summary": analysis.get("summary", ""),
                "entities": _fmt_entities(analysis.get("entities", {})),
                "risk_flags": _fmt_risks(analysis.get("risk_flags", [])),
                **{c: "" for c in HUMAN_COLS},
            })
            key_rows.append({
                "presentation_id": pres_id, "doc_reference": doc,
                "output_label": label, "model": model,
            })

    rng.shuffle(blind_rows)  # global row shuffle so doc order doesn't leak either

    blind_path = cache_dir / "blind_eval.csv"
    with open(blind_path, "w", newline="", encoding="utf-8") as f:
        cols = ["presentation_id", "doc_reference", "output_label",
                "summary", "entities", "risk_flags"] + HUMAN_COLS
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(blind_rows)

    key_path = cache_dir / "unblinding_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["presentation_id", "doc_reference",
                                          "output_label", "model"])
        w.writeheader()
        w.writerows(sorted(key_rows, key=lambda r: r["presentation_id"]))

    gt_path = cache_dir / "ground_truth_key_template.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_reference", "source_text"] + GT_KEY_COLS)
        w.writeheader()
        for doc in sorted(analyses):
            w.writerow({"doc_reference": doc,
                        "source_text": docs_text.get(doc, "")[:8000],
                        **{c: "" for c in GT_KEY_COLS}})

    return len(blind_rows), len(key_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    args = ap.parse_args()
    n_blind, _ = build(args.cache_dir)
    print(f"Wrote blind_eval.csv ({n_blind} rows), unblinding_key.csv, "
          f"ground_truth_key_template.csv -> {args.cache_dir}")
    print("Rating columns are 1-5 (5 best). Fill blind_eval.csv (identity hidden) and")
    print("ground_truth_key_template.csv (true entities/risks per doc) BEFORE aggregating.")
    print("Do NOT open unblinding_key.csv until ratings are done.")


if __name__ == "__main__":
    main()
