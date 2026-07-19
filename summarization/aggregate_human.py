#!/usr/bin/env python3
"""Merge the filled blind human ratings with the auto proxy metrics into the
final results table.

Joins blind_eval.csv (filled by raters) with unblinding_key.csv on
presentation_id to recover the approach, averages the human scores per approach,
and combines them with summ_auto_summary.csv into:

  approach | ROUGE-L | BERTScore | avg human coverage | avg human accuracy |
           avg human fluency | mean length | compression ratio

Human columns show "(pending)" for any approach with no ratings yet, so this can
be run before or after the humans finish. Writes summ_final_results.{csv,md}.

The markdown carries the standing note: human scores are the primary basis for
the conclusion; ROUGE is a flagged proxy (inflated for extractive methods).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONCLUSION_NOTE = (
    "**Human scores are the primary basis for the conclusion.** ROUGE is a flagged "
    "PROXY only — extractive methods (TextRank, LexRank) inflate ROUGE by reusing "
    "source sentences verbatim, so a high ROUGE does not mean a better summary. "
    "BERTScore-vs-source is likewise a reference-free proxy (truncated to 512 tokens "
    "on long docs). Read the human coverage / accuracy / fluency columns as the verdict."
)

HUMAN_COLS = ["coverage", "factual_accuracy", "fluency"]


def human_means(cache: Path) -> pd.DataFrame | None:
    blind_path = cache / "blind_eval.csv"
    key_path = cache / "unblinding_key.csv"
    if not (blind_path.exists() and key_path.exists()):
        return None
    blind = pd.read_csv(blind_path)
    key = pd.read_csv(key_path)
    merged = blind.merge(key[["presentation_id", "approach"]], on="presentation_id")
    for c in HUMAN_COLS:
        merged[c] = pd.to_numeric(merged.get(c), errors="coerce")
    if merged[HUMAN_COLS].notna().to_numpy().sum() == 0:
        return None  # nothing rated yet
    means = merged.groupby("approach", as_index=False)[HUMAN_COLS].mean()
    means["n_rated"] = merged.dropna(subset=HUMAN_COLS, how="all").groupby(
        "approach")["presentation_id"].count().reindex(means["approach"]).values
    return means.round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/summarization"
                                 if on_colab else ".cache_summ"))
    args = ap.parse_args()
    cache = args.cache_dir

    auto = pd.read_csv(cache / "summ_auto_summary.csv")
    human = human_means(cache)

    table = auto[["approach", "rougeL", "bertscore_f1", "mean_length", "compression"]].copy()
    if human is not None:
        table = table.merge(human, on="approach", how="left")
        for c in HUMAN_COLS:
            table[c] = table[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "(pending)")
        pending = False
    else:
        for c in HUMAN_COLS:
            table[c] = "(pending)"
        pending = True

    table = table.rename(columns={
        "approach": "Approach", "rougeL": "ROUGE-L (proxy)",
        "bertscore_f1": "BERTScore (proxy)", "coverage": "Human coverage",
        "factual_accuracy": "Human accuracy", "fluency": "Human fluency",
        "mean_length": "Mean length (words)", "compression": "Compression ratio",
    })
    ordered = ["Approach", "ROUGE-L (proxy)", "BERTScore (proxy)", "Human coverage",
               "Human accuracy", "Human fluency", "Mean length (words)", "Compression ratio"]
    table = table[ordered]

    table.to_csv(cache / "summ_final_results.csv", index=False)
    md = ["# LexChain summarization comparison — final results", "",
          table.to_markdown(index=False, floatfmt=".4f"), "", CONCLUSION_NOTE]
    if pending:
        md += ["", "_Human columns are pending: fill blind_eval.csv and re-run._"]
    (cache / "summ_final_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nWrote {cache}/summ_final_results.csv and summ_final_results.md")


if __name__ == "__main__":
    main()
