#!/usr/bin/env python3
"""Prepare human ground-truth authoring materials (no LLM, no API key).

Produces, for the deterministic 10-doc sample (seed 42):
  ground_truth_key_template.csv   one BLANK row per document for YOU to fill with
                                  the true parties/dates/monetary_amounts/
                                  obligations/risk_clauses. Nothing is pre-filled
                                  or LLM-generated -- human-authored ground truth
                                  is what makes the evaluation defensible.
  doc_texts/NN_<stem>.txt         readable dump of each document's OHR-Bench
                                  ground-truth text (exactly the text the models
                                  are given), so you can read it to fill the key.
  doc_texts/INDEX.txt             number -> full document stem.

Fill the template, save it as ground_truth_key.csv, and aggregate_analysis.py
scores each model's extracted entities/risks against YOUR key.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .data import DOC_CHAR_CAP, load_docs, sample_stems

# Canonical ground-truth key columns (single source of truth; imported elsewhere).
GT_KEY_COLS = ["parties", "dates", "monetary_amounts", "obligations", "risk_clauses"]

TEMPLATE_HELP = {
    "parties": "every named party/signatory/org (';'-separated)",
    "dates": "effective/execution/deadline/term dates as written (';'-separated)",
    "monetary_amounts": "each amount with context, e.g. '$4,162,000.00 grant' (';'-separated)",
    "obligations": "each obligation as 'who must do what' (';'-separated)",
    "risk_clauses": "clauses a lawyer should flag: indemnification, auto-renewal, etc. (';'-separated)",
}


def write_ground_truth_template(cache_dir: Path, docs: dict[str, str]) -> Path:
    """Strictly blank template: one empty row per document. Columns match exactly
    what aggregate_analysis.py reads, so a filled copy needs no reshaping.
    Column guidance lives in ground_truth_key_INSTRUCTIONS.txt (kept out of the CSV
    so nothing can pollute the filled key)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "ground_truth_key_template.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_reference"] + GT_KEY_COLS)
        w.writeheader()
        for stem in sorted(docs):
            w.writerow({"doc_reference": stem, **{c: "" for c in GT_KEY_COLS}})
    (cache_dir / "ground_truth_key_INSTRUCTIONS.txt").write_text(
        "Fill ground_truth_key_template.csv from the readable dumps in doc_texts/,\n"
        "then save it as ground_truth_key.csv (same folder).\n"
        "Each cell is a ';'-separated list. Leave blank if the document states nothing.\n\n"
        + "\n".join(f"- {c}: {h}" for c, h in TEMPLATE_HELP.items())
        + "\n\nThis key is hand-authored by you — never LLM-generated.\n",
        encoding="utf-8",
    )
    return path


def _safe_name(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:100]


def write_doc_dumps(cache_dir: Path, docs: dict[str, str]) -> list[Path]:
    out_dir = cache_dir / "doc_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(docs)
    written = []
    index_lines = ["LexChain analysis — 10 sampled documents (seed 42)\n"]
    for i, stem in enumerate(stems, 1):
        text = docs[stem]
        truncated = len(text) >= DOC_CHAR_CAP
        path = out_dir / f"{i:02d}_{_safe_name(stem)}.txt"
        header = [
            "=" * 78,
            "LexChain document-analysis — ground-truth authoring dump",
            f"Document {i}/10: {stem}",
            f"Characters (model-visible): {len(text)}"
            + ("  [TRUNCATED to first 96,000 chars for analysis]" if truncated else ""),
            "Fill this document's row in ground_truth_key_template.csv "
            "(parties/dates/monetary_amounts/obligations/risk_clauses).",
            "=" * 78,
            "",
        ]
        path.write_text("\n".join(header) + text + "\n", encoding="utf-8")
        written.append(path)
        index_lines.append(f"{i:2d}. {stem}  ({len(text)} chars"
                           + (", truncated)" if truncated else ")"))
    (out_dir / "INDEX.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    ap.add_argument("--limit-docs", type=int, default=10)
    args = ap.parse_args()

    all_docs = load_docs()
    if not all_docs:
        raise SystemExit("no GT docs found (run download_data.py first)")
    stems = sample_stems(all_docs.keys(), n=args.limit_docs)
    docs = {s: all_docs[s] for s in stems}

    tmpl = write_ground_truth_template(args.cache_dir, docs)
    dumps = write_doc_dumps(args.cache_dir, docs)
    print(f"Wrote {tmpl}")
    print(f"Wrote {len(dumps)} readable dumps -> {args.cache_dir}/doc_texts/ (+ INDEX.txt)")
    print("\nNext: read the dumps, fill ground_truth_key_template.csv, and save it as")
    print(f"  {args.cache_dir}/ground_truth_key.csv")
    print("Nothing here is LLM-generated — the key is yours to hand-author.")


if __name__ == "__main__":
    main()
