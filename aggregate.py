#!/usr/bin/env python3
"""Aggregate per-config results into paper-ready outputs:

  matrix.csv           all 9 configs, all metrics
  matrix.md            full 9-row table
  chunkers_table.md    chunker marginals (averaged over embedders)
  embedders_table.md   embedder marginals (averaged over chunkers)

plus an interaction-effect note on Recall@5: for each cell,
residual = cell - (chunker mean + embedder mean - grand mean); cells with
|residual| > --interaction-threshold over/under-perform their marginals.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

METRIC_COLS = ["recall@1", "recall@5", "mrr@5", "ndcg@10", "lcs@5", "coverage"]
OPS_COLS = ["n_chunks", "tokens_p50", "chunk_time_s", "chunks_per_s", "index_mb"]


def load_results(cache: Path, suffix: str) -> pd.DataFrame:
    rows = []
    for path in sorted((cache / "results").glob("*.json")):
        name = path.stem
        if suffix:
            if not name.endswith(suffix):
                continue
        elif re.search(r"_n\d+$", name):
            continue  # skip smoke-test slices when aggregating the full run
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no results under {cache}/results (suffix={suffix or 'none'})")
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench" if on_colab
                                 else ".cache_bench"))
    ap.add_argument("--suffix", default="", help="e.g. _n5 for the smoke slice")
    ap.add_argument("--interaction-threshold", type=float, default=0.02)
    args = ap.parse_args()

    df = load_results(args.cache_dir, args.suffix)
    df = df.sort_values(["chunker", "embedder"]).reset_index(drop=True)

    matrix = df[["chunker", "embedder"] + METRIC_COLS + OPS_COLS]
    chunker_tbl = (
        df.groupby("chunker", as_index=False)[METRIC_COLS + OPS_COLS].mean(numeric_only=True)
        .round(4)
    )
    embedder_tbl = (
        df.groupby("embedder", as_index=False)[METRIC_COLS + ["chunks_per_s", "index_mb"]]
        .mean(numeric_only=True).round(4)
    )

    # interaction effects on Recall@5
    pivot = df.pivot(index="chunker", columns="embedder", values="recall@5")
    grand = pivot.values.mean()
    expected = (
        pivot.mean(axis=1).values[:, None] + pivot.mean(axis=0).values[None, :] - grand
    )
    residual = pivot - expected
    flags = []
    for ch in residual.index:
        for em in residual.columns:
            r = residual.loc[ch, em]
            if abs(r) > args.interaction_threshold:
                direction = "OVER" if r > 0 else "UNDER"
                flags.append(f"- **{ch} x {em} {direction}-performs its marginals "
                             f"on Recall@5 by {r:+.3f}**")
    interaction = ["## Interaction effects (Recall@5 vs additive marginals)", ""]
    if flags:
        interaction += flags
    else:
        interaction.append(f"No chunker x embedder pairing deviates from its marginals "
                           f"by more than {args.interaction_threshold} -- effects are "
                           f"additive; the tables below can be read independently.")
    interaction_md = "\n".join(interaction)

    out = args.cache_dir
    matrix.to_csv(out / f"matrix{args.suffix}.csv", index=False)
    matrix_md = md(matrix) + "\n\n" + interaction_md + "\n"
    (out / f"matrix{args.suffix}.md").write_text(matrix_md, encoding="utf-8")
    (out / f"chunkers_table{args.suffix}.md").write_text(md(chunker_tbl) + "\n", encoding="utf-8")
    (out / f"embedders_table{args.suffix}.md").write_text(md(embedder_tbl) + "\n", encoding="utf-8")

    n_q = df["n_questions"].iloc[0]
    print(f"=== Full matrix ({len(df)} configs, {n_q} questions each) ===\n")
    print(matrix_md)
    print("\n=== Chunkers (averaged over embedders) ===\n")
    print(md(chunker_tbl))
    print("\n=== Embedders (averaged over chunkers) ===\n")
    print(md(embedder_tbl))
    print(f"\nConfigs:\n" + "\n".join(
        f"  {r.chunker}: {r.chunker_config}" for r in
        df.drop_duplicates("chunker").itertuples()))
    print("\n".join(
        f"  {r.embedder}: {r.embedder_model}" for r in
        df.drop_duplicates("embedder").itertuples()))
    print(f"\nWrote matrix{args.suffix}.csv/.md, chunkers_table{args.suffix}.md, "
          f"embedders_table{args.suffix}.md -> {out}")
    print("Note: relevance = doc+page gate AND word-LCS >= "
          f"{df['relevance_threshold'].iloc[0]}; lcs@5 is OHR-Bench's unthresholded "
          "metric; all Law QAs count in every denominator.")


if __name__ == "__main__":
    main()
