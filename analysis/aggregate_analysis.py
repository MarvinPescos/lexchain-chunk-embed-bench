#!/usr/bin/env python3
"""Aggregate the filled human scoring sheet + ground-truth key into the results.

Joins (all in <cache>):
  blind_eval.csv         (filled by raters: human 1-5 ratings, hidden identities)
  unblinding_key.csv     (presentation_id -> model)
  analyses/*.json        (each model's extracted entities/risks, for F1)
  ground_truth_key.csv   (you fill this from ground_truth_key_template.csv:
                          true parties/dates/monetary_amounts/obligations/risk_clauses)

Produces:
  results.csv / results.md   final UN-BLINDED table:
      model | summary coverage | accuracy | fluency | entity F1 | risk F1
  (+ per-category entity P/R/F1 and human entities/risk ratings as detail)

Two-phase: if human ratings or the ground-truth key are not yet filled, prints
what is pending and still reports whatever is available (latency, output sizes).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .matching_entities import prf_from_counts, score_category
from .prompt import ENTITY_CATEGORIES

RATING_COLS = {
    "summary_coverage_1to5": "coverage",
    "summary_accuracy_1to5": "accuracy",
    "summary_fluency_1to5": "fluency",
    "entities_correct_1to5": "entities_human",
    "risk_flags_correct_1to5": "risk_human",
}
GT_LIST_COLS = ["parties", "dates", "monetary_amounts", "obligations", "risk_clauses"]


def _split_cell(cell: str) -> list[str]:
    """A ground-truth key cell is a ';' or newline separated list."""
    if not cell:
        return []
    parts = []
    for line in str(cell).replace("\n", ";").split(";"):
        line = line.strip(" -\t")
        if line:
            parts.append(line)
    return parts


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(cache_dir: Path):
    key = {r["presentation_id"]: r["model"] for r in load_csv(cache_dir / "unblinding_key.csv")}
    if not key:
        raise SystemExit(f"no unblinding_key.csv in {cache_dir} (run build_blind_eval.py)")
    blind = load_csv(cache_dir / "blind_eval.csv")

    # ---- human ratings per model (un-blinded via key) ----
    ratings = defaultdict(lambda: defaultdict(list))
    human_filled = 0
    for row in blind:
        model = key.get(row["presentation_id"])
        if not model:
            continue
        row_has = False
        for col, short in RATING_COLS.items():
            v = _to_float(row.get(col, ""))
            if v is not None:
                ratings[model][short].append(v)
                row_has = True
        human_filled += int(row_has)

    # ---- entity / risk F1 vs the filled ground-truth key ----
    gt_rows = {r["doc_reference"]: r for r in load_csv(cache_dir / "ground_truth_key.csv")}
    analyses = defaultdict(dict)
    for p in sorted((cache_dir / "analyses").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        analyses[rec["model"]][rec["doc"]] = rec.get("parsed") or {}

    # counts[model][category] = [tp, fp, fn]
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    gt_filled = sum(
        1 for r in gt_rows.values() if any(_split_cell(r.get(c, "")) for c in GT_LIST_COLS)
    )
    for model, docs in analyses.items():
        for doc, parsed in docs.items():
            gt = gt_rows.get(doc)
            if not gt:
                continue
            ent = parsed.get("entities", {}) if parsed else {}
            for cat in ENTITY_CATEGORIES:  # entity categories
                s = score_category(cat, ent.get(cat, []), _split_cell(gt.get(cat, "")))
                c = counts[model][cat]
                c[0] += s["tp"]; c[1] += s["fp"]; c[2] += s["fn"]
            # risk: predicted risk descriptions vs gold risk_clauses
            pred_risks = [f.get("risk", "") for f in (parsed.get("risk_flags") or [])]
            s = score_category("risk_clauses", pred_risks, _split_cell(gt.get("risk_clauses", "")))
            c = counts[model]["risk_clauses"]
            c[0] += s["tp"]; c[1] += s["fp"]; c[2] += s["fn"]

    # ---- assemble per-model summary ----
    models = sorted(set(list(ratings) + list(analyses)))
    rows = []
    for model in models:
        r = {"model": model}
        for short in ("coverage", "accuracy", "fluency", "entities_human", "risk_human"):
            vals = ratings[model][short]
            r[short] = round(sum(vals) / len(vals), 3) if vals else None
        # entity F1 = micro over the four entity categories
        etp = sum(counts[model][cat][0] for cat in ENTITY_CATEGORIES)
        efp = sum(counts[model][cat][1] for cat in ENTITY_CATEGORIES)
        efn = sum(counts[model][cat][2] for cat in ENTITY_CATEGORIES)
        r["entity_f1"] = round(prf_from_counts(etp, efp, efn)["f1"], 3) if (etp + efp + efn) else None
        rtp, rfp, rfn = counts[model]["risk_clauses"]
        r["risk_f1"] = round(prf_from_counts(rtp, rfp, rfn)["f1"], 3) if (rtp + rfp + rfn) else None
        for cat in ENTITY_CATEGORIES:
            tp, fp, fn = counts[model][cat]
            r[f"{cat}_f1"] = round(prf_from_counts(tp, fp, fn)["f1"], 3) if (tp+fp+fn) else None
        rows.append(r)

    return rows, human_filled, gt_filled, len(blind), len(gt_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    args = ap.parse_args()
    rows, human_filled, gt_filled, n_blind, n_gt = aggregate(args.cache_dir)

    if human_filled < n_blind:
        print(f"PENDING: {human_filled}/{n_blind} blind rows have human ratings "
              f"— fill blind_eval.csv to complete coverage/accuracy/fluency.")
    if gt_filled == 0:
        print("PENDING: ground_truth_key.csv is empty/absent — fill it (rename from "
              "ground_truth_key_template.csv) so entity/risk F1 can be computed.")

    df = pd.DataFrame(rows)
    headline_cols = ["model", "coverage", "accuracy", "fluency", "entity_f1", "risk_f1"]
    headline = df[[c for c in headline_cols if c in df.columns]].rename(columns={
        "coverage": "summary coverage", "accuracy": "accuracy", "fluency": "fluency",
        "entity_f1": "entity F1", "risk_f1": "risk F1",
    })
    md = headline.to_markdown(index=False, floatfmt=".3f")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.cache_dir / "results.csv", index=False)
    (args.cache_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print("\n=== Final comparison (un-blinded) ===\n")
    print(md)
    print(f"\nWrote results.csv (full detail incl. per-category F1) and results.md "
          f"-> {args.cache_dir}")


if __name__ == "__main__":
    main()
