#!/usr/bin/env python3
"""Automated PROXY metrics for the summaries (Layer 1 of the evaluation).

For every cached summary, computes against the SOURCE document (reference-free,
since there is no gold summary):
  ROUGE-1 / ROUGE-2 / ROUGE-L   (rouge-score)
  BERTScore F1                  (bert-score, roberta-large)
  summary length (words), compression ratio (summary_words / source_words)

============================  READ THIS  ====================================
These are PROXY metrics, NOT a quality verdict:
  * Extractive methods (TextRank, LexRank) copy source sentences verbatim, so
    their ROUGE-vs-source is INFLATED by construction. ROUGE must NOT be read
    as the quality ranking of extractive vs abstractive.
  * BERTScore is computed against the full source but the encoder truncates to
    512 tokens, so for long docs it mainly reflects overlap with the opening.
The blind HUMAN evaluation (build_blind_eval.py -> aggregate_human.py) is the
primary basis for the conclusion.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DISCLAIMER = (
    "PROXY METRICS — not a quality verdict. Extractive methods inflate ROUGE by "
    "reusing source sentences verbatim; BERTScore-vs-source truncates long docs "
    "to 512 tokens. Note also that at sentence_count=8 the extractive summaries "
    "run much longer than the ~180-word LLM summaries (legal sentences are long) "
    "— see the mean_length/compression columns; longer extracts raise ROUGE "
    "further. Human evaluation is the primary basis for the conclusion."
)


def load_summaries(cache: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((cache / "summaries").glob("*__*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no summaries under {cache}/summaries (run summarize.py)")
    return pd.DataFrame(rows)


def load_sources(data_dir: Path) -> dict[str, str]:
    from summarize import load_docs

    return load_docs(data_dir)


def compute_rouge(df: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for row in df.itertuples():
        s = scorer.score(sources[row.doc], row.summary)  # (target=source, prediction=summary)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    df["rouge1"], df["rouge2"], df["rougeL"] = r1, r2, rl
    return df


def compute_bertscore(df: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("bert-score not installed; skipping BERTScore (install for full metrics)")
        df["bertscore_f1"] = float("nan")
        return df
    cands = df["summary"].tolist()
    refs = [sources[d] for d in df["doc"]]
    _, _, f1 = bert_score(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
    df["bertscore_f1"] = f1.tolist()
    return df


def summarize_table(df: pd.DataFrame) -> pd.DataFrame:
    df["compression"] = df["summary_words"] / df["source_words"].clip(lower=1)
    agg = df.groupby("approach", as_index=False).agg(
        rouge1=("rouge1", "mean"), rouge2=("rouge2", "mean"), rougeL=("rougeL", "mean"),
        bertscore_f1=("bertscore_f1", "mean"),
        mean_length=("summary_words", "mean"),
        compression=("compression", "mean"),
    )
    return agg.round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/summarization"
                                 if on_colab else ".cache_summ"))
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip BERTScore (no torch needed)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    df = load_summaries(args.cache_dir)
    sources = load_sources(args.data_dir)
    df = compute_rouge(df, sources)
    if args.no_bertscore:
        df["bertscore_f1"] = float("nan")
    else:
        df = compute_bertscore(df, sources)

    per_doc_cols = ["doc", "approach", "summary_words", "source_words",
                    "rouge1", "rouge2", "rougeL", "bertscore_f1"]
    df[per_doc_cols].to_csv(args.cache_dir / "summ_auto_metrics.csv", index=False)

    table = summarize_table(df)
    table.to_csv(args.cache_dir / "summ_auto_summary.csv", index=False)
    print("\n" + "=" * 78)
    print(DISCLAIMER)
    print("=" * 78 + "\n")
    print(table.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote {args.cache_dir}/summ_auto_metrics.csv and summ_auto_summary.csv")


if __name__ == "__main__":
    main()
