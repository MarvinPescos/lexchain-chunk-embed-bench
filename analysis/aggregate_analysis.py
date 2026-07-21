#!/usr/bin/env python3
"""Aggregate the filled blind sheet + hand-authored key into the final results.

Joins (all in <cache>):
  blind_eval.csv         filled by raters: SummEval 1-5 ratings, identities hidden
  unblinding_key.csv     presentation_id -> model (+ role)
  analyses/*.json        each model's entities/risk checklist + latency, for F1/timing
  ground_truth_key.csv   HAND-AUTHORED true entities + risk clauses per doc

Outputs:
  results.csv / results.md  final un-blinded table:
      model | coherence | consistency | fluency | relevance | entity F1 | risk F1 | mean latency
    candidates sorted by risk F1 (desc); the 70B reference row separated and
    labeled "reference (non-deployable)".
  wins.csv                  per-document win counts among CANDIDATES (n=10 is too
    small for significance tests; we report wins + means). Ties: all tied win.

Risk F1: the model's checklist categories marked present (category + quote) are
fuzzy-matched against the key's hand-written risk_clauses -- same matcher as
entities (matching_entities.py).

Two-phase: with unfilled ratings/key it prints what is pending and reports what
it can (latency, parse status).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .data import MODELS, REFERENCE_ROLE
from .matching_entities import prf_from_counts, score_category
from .prompt import ENTITY_CATEGORIES

RATING_COLS = {  # blind-sheet column -> short name (SummEval dimensions)
    "coherence_1to5": "coherence",
    "consistency_1to5": "consistency",
    "fluency_1to5": "fluency",
    "relevance_1to5": "relevance",
}
DIMENSIONS = list(RATING_COLS.values()) + ["entity_f1", "risk_f1"]
GT_LIST_COLS = ["parties", "dates", "monetary_amounts", "obligations", "risk_clauses"]


def _split_cell(cell: str) -> list[str]:
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


def _pred_risk_strings(parsed: dict) -> list[str]:
    """Checklist -> matchable strings: 'category quote' for present categories."""
    out = []
    for f in parsed.get("risk_flags") or []:
        if f.get("present"):
            cat = str(f.get("category", "")).replace("_", " ")
            out.append((cat + " " + str(f.get("quote", ""))).strip())
    return out


def _doc_prf(parsed: dict, gt_row: dict) -> tuple[dict, dict]:
    """Per-doc entity + risk tp/fp/fn against the hand-authored key row."""
    ent_c = {"tp": 0, "fp": 0, "fn": 0}
    ent = parsed.get("entities", {}) if parsed else {}
    for cat in ENTITY_CATEGORIES:
        s = score_category(cat, ent.get(cat, []), _split_cell(gt_row.get(cat, "")))
        for k in ent_c:
            ent_c[k] += s[k]
    risk_s = score_category("risk_clauses", _pred_risk_strings(parsed or {}),
                            _split_cell(gt_row.get("risk_clauses", "")))
    return ent_c, {k: risk_s[k] for k in ("tp", "fp", "fn")}


def aggregate(cache_dir: Path):
    key_rows = load_csv(cache_dir / "unblinding_key.csv")
    if not key_rows:
        raise SystemExit(f"no unblinding_key.csv in {cache_dir} (run build_blind_eval.py)")
    key = {r["presentation_id"]: r for r in key_rows}
    blind = load_csv(cache_dir / "blind_eval.csv")
    gt_rows = {r["doc_reference"]: r for r in load_csv(cache_dir / "ground_truth_key.csv")}

    analyses = defaultdict(dict)   # model -> doc -> checkpoint record
    for p in sorted((cache_dir / "analyses").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        analyses[rec["model"]][rec["doc"]] = rec

    # ---- per-(model, doc) records ----
    per_doc: dict[tuple[str, str], dict] = {}
    human_filled = 0
    for row in blind:
        k = key.get(row["presentation_id"])
        if not k:
            continue
        rec = per_doc.setdefault((k["model"], row["doc_reference"]), {})
        got_any = False
        for col, short in RATING_COLS.items():
            v = _to_float(row.get(col, ""))
            if v is not None:
                rec[short] = v
                got_any = True
        human_filled += int(got_any)

    gt_filled = sum(1 for r in gt_rows.values()
                    if any(_split_cell(r.get(c, "")) for c in GT_LIST_COLS))
    for model, docs in analyses.items():
        for doc, rec_cp in docs.items():
            rec = per_doc.setdefault((model, doc), {})
            if rec_cp.get("latency_s") is not None:
                rec["latency_s"] = rec_cp["latency_s"]
                rec["latency_label"] = rec_cp.get("latency_label", "")
            gt = gt_rows.get(doc)
            if gt and rec_cp.get("parsed"):
                ent_c, risk_c = _doc_prf(rec_cp["parsed"], gt)
                rec["entity_f1"] = prf_from_counts(**ent_c)["f1"]
                rec["risk_f1"] = prf_from_counts(**risk_c)["f1"]
                rec["_ent_counts"], rec["_risk_counts"] = ent_c, risk_c

    # ---- per-model summary (micro-F1 over docs; means for ratings/latency) ----
    models = sorted({m for m, _ in per_doc})
    rows = []
    for model in models:
        recs = [r for (m, _), r in per_doc.items() if m == model]
        row = {"model": model,
               "role": MODELS[model]["role"] if model in MODELS else "candidate",
               "model_id": MODELS[model]["id"] if model in MODELS else ""}
        for dim in RATING_COLS.values():
            vals = [r[dim] for r in recs if dim in r]
            row[dim] = round(sum(vals) / len(vals), 3) if vals else None
        for fkey, ckey in (("entity_f1", "_ent_counts"), ("risk_f1", "_risk_counts")):
            counts = [r[ckey] for r in recs if ckey in r]
            if counts:
                tp = sum(c["tp"] for c in counts)
                fp = sum(c["fp"] for c in counts)
                fn = sum(c["fn"] for c in counts)
                row[fkey] = round(prf_from_counts(tp, fp, fn)["f1"], 3)
            else:
                row[fkey] = None
        lats = [r["latency_s"] for r in recs if "latency_s" in r]
        row["mean_latency_s"] = round(sum(lats) / len(lats), 2) if lats else None
        row["latency_label"] = next((r["latency_label"] for r in recs
                                     if r.get("latency_label")), "")
        rows.append(row)

    # ---- per-document win counts among candidates (ties: all tied win) ----
    candidates = [m for m in models if (MODELS.get(m, {}).get("role") == "candidate")]
    docs_all = sorted({d for _, d in per_doc})
    wins = {m: {dim: 0 for dim in DIMENSIONS} for m in candidates}
    for dim in DIMENSIONS:
        higher_better = True
        for doc in docs_all:
            scored = [(m, per_doc.get((m, doc), {}).get(dim)) for m in candidates]
            scored = [(m, v) for m, v in scored if v is not None]
            if len(scored) < 2:
                continue
            best = max(v for _, v in scored) if higher_better else min(v for _, v in scored)
            for m, v in scored:
                if v == best:
                    wins[m][dim] += 1

    return rows, wins, human_filled, gt_filled, len(blind), len(gt_rows)


PAPER_COLS = [
    ("model", "Model"),
    ("coherence", "Coherence"),
    ("consistency", "Consistency"),
    ("fluency", "Fluency"),
    ("relevance", "Relevance"),
    ("entity_f1", "Entity F1"),
    ("risk_f1", "Risk F1"),
    ("mean_latency_s", "Mean latency (s)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    args = ap.parse_args()
    rows, wins, human_filled, gt_filled, n_blind, n_gt = aggregate(args.cache_dir)

    if human_filled < n_blind:
        print(f"PENDING: {human_filled}/{n_blind} blind rows have human ratings.")
    if gt_filled == 0:
        print("PENDING: ground_truth_key.csv empty/absent -- entity/risk F1 unavailable.")

    df = pd.DataFrame(rows)
    cand = df[df["role"] == "candidate"].sort_values(
        "risk_f1", ascending=False, na_position="last")
    ref = df[df["role"] != "candidate"]

    def table(d):
        return d[[c for c, _ in PAPER_COLS]].rename(columns=dict(PAPER_COLS)) \
            .to_markdown(index=False, floatfmt=".3f")

    latency_label = next((r["latency_label"] for r in rows if r.get("latency_label")), "gpu")
    md_parts = [
        f"## Candidates (self-hostable, sorted by Risk F1; latency = {latency_label})",
        "", table(cand), "",
        f"## Reference — {REFERENCE_ROLE}", "",
        table(ref) if len(ref) else "_(reference model not yet run)_", "",
        "## Per-document wins among candidates (n=10; ties count for all tied)",
        "",
    ]
    wins_rows = [{"model": m, **w} for m, w in wins.items()]
    wins_df = pd.DataFrame(wins_rows).sort_values("risk_f1", ascending=False) \
        if wins_rows else pd.DataFrame()
    md_parts.append(wins_df.to_markdown(index=False) if len(wins_df) else "_(pending)_")
    md = "\n".join(md_parts) + "\n"

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.cache_dir / "results.csv", index=False)
    if len(wins_df):
        wins_df.to_csv(args.cache_dir / "wins.csv", index=False)
    (args.cache_dir / "results.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote results.csv, results.md"
          + (", wins.csv" if len(wins_df) else "") + f" -> {args.cache_dir}")


if __name__ == "__main__":
    main()
