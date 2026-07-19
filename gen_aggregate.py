#!/usr/bin/env python3
"""Aggregate the RAG generation results into the paper deliverables:

  gen_matrix.csv          per-model summary (all metrics + tokens + cost)
  gen_results_table.md    model | OHR-Bench F1 | accuracy (EM) | mean latency |
                          cost/1k  -- winner (max F1) marked as the system result,
                          plus a per-evidence_source F1 breakdown
  human_review.csv        every (model, question) row with auto scores + BLANK
                          human_correct/notes columns; ~40 stratified rows flagged
                          spot_check=yes for the team to validate

`--with-human filled.csv` computes auto-vs-human agreement (% + Cohen's kappa)
once the human_correct column is filled in.

PRICING is Groq pay-as-you-go $/1M tokens -- APPROXIMATE, verify against
groq.com/pricing before quoting; cost/1k is derived from measured token usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# $/1M tokens (input, output). VERIFY at groq.com/pricing -- values approximate.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.05, 0.20),
    "qwen/qwen3.6-27b": (0.15, 0.60),
}
SPOT_CHECK_ROWS = 40
AUTO_CORRECT_F1 = 0.5  # auto_correct := em==1 OR f1 >= this


def slug(model):
    return model.replace("/", "__").replace(":", "_")


def load_records(cache: Path, suffix: str) -> list[dict]:
    recs = []
    for path in sorted((cache / "gen_results").glob(f"*{suffix}.jsonl")):
        # when suffix=="" (full run), skip the _smoke/_n sample files
        if suffix == "" and ("_smoke" in path.stem or "_n" in path.stem):
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def cost_per_1k(model, mean_prompt, mean_completion) -> float | None:
    if model not in PRICING:
        return None
    pin, pout = PRICING[model]
    return 1000 * (mean_prompt * pin + mean_completion * pout) / 1e6


def summarize(recs: list[dict]) -> pd.DataFrame:
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)
    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        mean = lambda key: sum(r[key] for r in rs) / n
        mp, mc = mean("prompt_tokens"), mean("completion_tokens")
        rows.append({
            "model": model,
            "n": n,
            "ohr_f1": round(mean("f1"), 4),
            "accuracy_em": round(mean("em"), 4),
            "acc_contains": round(mean("em_contains"), 4),
            "mean_latency_s": round(mean("latency_s"), 3),
            "mean_prompt_tok": round(mp, 1),
            "mean_completion_tok": round(mc, 1),
            "cost_per_1k_usd": (round(cost_per_1k(model, mp, mc), 4)
                               if cost_per_1k(model, mp, mc) is not None else None),
        })
    df = pd.DataFrame(rows).sort_values("ohr_f1", ascending=False).reset_index(drop=True)
    return df


def breakdown_by_source(recs: list[dict]) -> pd.DataFrame:
    cells = defaultdict(lambda: defaultdict(list))
    sources = []
    for r in recs:
        src = r.get("evidence_source", "unknown")
        cells[r["model"]][src].append(r["f1"])
        if src not in sources:
            sources.append(src)
    sources = sorted(sources)
    rows = []
    for model in sorted(cells):
        row = {"model": model}
        for src in sources:
            vals = cells[model].get(src, [])
            row[f"F1[{src}] (n={len(vals)})"] = round(sum(vals) / len(vals), 3) if vals else None
        rows.append(row)
    return pd.DataFrame(rows)


def write_review_csv(recs: list[dict], path: Path):
    rows = []
    for r in recs:
        auto_correct = int(r["em"] == 1 or r["f1"] >= AUTO_CORRECT_F1)
        rows.append({
            "model": r["model"],
            "qid": r["qid"],
            "evidence_source": r.get("evidence_source"),
            "answer_form": r.get("answer_form"),
            "question": r["question"],
            "ground_truth_answer": r["ground_truth"],
            "model_answer": r["answer"],
            "auto_f1": r["f1"],
            "auto_em": r["em"],
            "auto_correct": auto_correct,
            "human_correct": "",   # <- team fills 1/0
            "notes": "",
        })
    # flag ~40 stratified spot-check rows (by evidence_source), deterministic
    by_src = defaultdict(list)
    for i, row in enumerate(rows):
        by_src[row["evidence_source"]].append(i)
    rng = random.Random(SAMPLE_SEED := 42)
    flagged = set()
    srcs = sorted(by_src)
    per = max(1, SPOT_CHECK_ROWS // max(1, len(srcs)))
    for src in srcs:
        idx = by_src[src][:]
        rng.shuffle(idx)
        flagged.update(idx[:per])
    for i in list(rng.sample(range(len(rows)), min(len(rows), SPOT_CHECK_ROWS))):
        if len(flagged) >= SPOT_CHECK_ROWS:
            break
        flagged.add(i)
    for i, row in enumerate(rows):
        row["spot_check"] = "yes" if i in flagged else ""
    fields = ["model", "qid", "evidence_source", "answer_form", "question",
              "ground_truth_answer", "model_answer", "auto_f1", "auto_em",
              "auto_correct", "human_correct", "notes", "spot_check"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return sum(1 for r in rows if r["spot_check"] == "yes")


def agreement(path: Path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("human_correct", "").strip() != ""]
    if not rows:
        print("no human_correct values filled in yet -- nothing to compare")
        return
    a = [int(r["auto_correct"]) for r in rows]
    h = [int(float(r["human_correct"])) for r in rows]
    n = len(rows)
    agree = sum(x == y for x, y in zip(a, h)) / n
    pa = agree
    pe = (sum(a) / n) * (sum(h) / n) + (1 - sum(a) / n) * (1 - sum(h) / n)
    kappa = (pa - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"auto-vs-human on {n} reviewed rows: {agree:.1%} agreement, "
          f"Cohen's kappa = {kappa:.3f}")


def md(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench" if on_colab
                                 else ".cache_bench"))
    ap.add_argument("--suffix", default="", help="_smokeK or _nN; default full run")
    ap.add_argument("--with-human", type=Path, default=None,
                    help="a filled human_review.csv -> print auto-vs-human agreement")
    args = ap.parse_args()

    if args.with_human:
        agreement(args.with_human)
        return

    recs = load_records(args.cache_dir, args.suffix)
    if not recs:
        raise SystemExit(f"no gen_results/*{args.suffix}.jsonl under {args.cache_dir}")

    df = summarize(recs)
    winner = df.iloc[0]["model"]
    out = args.cache_dir
    df.to_csv(out / f"gen_matrix{args.suffix}.csv", index=False)

    table = df[["model", "ohr_f1", "accuracy_em", "mean_latency_s", "cost_per_1k_usd"]].copy()
    table.columns = ["Model", "OHR-Bench F1", "Accuracy (EM)", "Mean latency (s)", "Cost / 1k ($)"]
    table["Model"] = [f"**{m}** ← end-to-end system result" if m == winner else m
                      for m in table["Model"]]
    src_tbl = breakdown_by_source(recs)

    n_reviewed = write_review_csv(recs, out / f"human_review{args.suffix}.csv")

    body = (
        f"# RAG generation benchmark ({len(recs)} answers, "
        f"{df['n'].iloc[0]} questions/model)\n\n"
        "Fixed pipeline: LangChain RecursiveCharacterTextSplitter (512/50) + "
        "e5-base-v2, top-k retrieval. Only the generation LLM varies. Metric: "
        "OHR-Bench F1 (headline) + EM accuracy, vendored from OHR-Bench.\n\n"
        "## Model comparison\n\n" + md(table) + "\n\n"
        "## OHR-Bench F1 by evidence source\n\n" + md(src_tbl) + "\n"
    )
    (out / f"gen_results_table{args.suffix}.md").write_text(body, encoding="utf-8")

    print(body)
    print(f"Winner (max OHR-Bench F1): {winner}")
    print(f"\nWrote gen_matrix{args.suffix}.csv, gen_results_table{args.suffix}.md, "
          f"human_review{args.suffix}.csv ({n_reviewed} rows flagged spot_check) -> {out}")
    print("Note: accuracy=EM is strict normalized equality; acc_contains "
          "(in gen_matrix.csv) is the forgiving variant. Cost/1k uses the "
          "editable PRICING dict -- verify against groq.com/pricing.")


if __name__ == "__main__":
    main()
