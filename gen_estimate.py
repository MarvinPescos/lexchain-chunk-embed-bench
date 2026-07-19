#!/usr/bin/env python3
"""Project full-run time & cost from smoke-test generation results.

Reads the smoke jsonl, then for each target size prints:
  - paid-tier cost (from measured token usage x PRICING)
  - free-tier calendar-day estimate (bounded by the per-model daily token cap)
  - compute time (sum of latencies, and the 30 RPM floor)

Free-tier daily token caps are approximate defaults; the smoke run also logs the
real `x-ratelimit-remaining-*` headroom per model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gen_aggregate import PRICING, cost_per_1k

# approximate Groq free-tier tokens-per-day per model class (verify via headers)
FREE_TPD = {
    "openai/gpt-oss-120b": 200_000,
    "openai/gpt-oss-20b": 500_000,
    "qwen/qwen3.6-27b": 200_000,
}
FREE_RPM = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench" if on_colab
                                 else ".cache_bench"))
    ap.add_argument("--smoke-suffix", default=None,
                    help="e.g. _smoke8; default: the first gen_sample_smoke*.json")
    ap.add_argument("--targets", default="1142,200")
    args = ap.parse_args()

    gr = args.cache_dir / "gen_results"
    if args.smoke_suffix:
        files = sorted(gr.glob(f"*{args.smoke_suffix}.jsonl"))
    else:
        files = sorted(gr.glob("*_smoke*.jsonl"))
    if not files:
        raise SystemExit(f"no smoke jsonl under {gr} (run rag_generate.py --smoke K)")

    by_model = defaultdict(list)
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_model[r["model"]].append(r)

    targets = [int(t) for t in args.targets.split(",")]
    print(f"=== Generation run projection (from {sum(len(v) for v in by_model.values())} "
          f"smoke answers) ===\n")
    grand_cost = {t: 0.0 for t in targets}
    for model, rs in by_model.items():
        n = len(rs)
        lat = sum(r["latency_s"] for r in rs) / n
        mp = sum(r["prompt_tokens"] for r in rs) / n
        mc = sum(r["completion_tokens"] for r in rs) / n
        tot = mp + mc
        c1k = cost_per_1k(model, mp, mc)
        tpd = FREE_TPD.get(model, 200_000)
        print(f"{model}: {lat:.2f}s/answer, ~{tot:.0f} tok/answer "
              f"({mp:.0f} in + {mc:.0f} out)"
              + (f", ${c1k:.4f}/1k" if c1k is not None else ""))
        for t in targets:
            compute_min = max(t * lat, t / FREE_RPM * 60) / 60
            free_days = (t * tot) / tpd
            cost = (c1k or 0) * t / 1000
            grand_cost[t] += cost
            print(f"    {t:>5} answers: paid ~${cost:.2f} | "
                  f"free-tier ~{free_days:.1f} day(s) @ {tpd//1000}K tok/day | "
                  f"~{compute_min:.0f} min compute")
        print()

    print("Across all models:")
    for t in targets:
        print(f"  {t} x {len(by_model)} models: paid ~${grand_cost[t]:.2f} total")
    print("\nFree-tier note: the per-model daily token cap is the binding limit; a "
          "run spanning several days is fine (every answer is checkpointed, rerun "
          "resumes). Adding a few $ of Groq credit removes the caps.")


if __name__ == "__main__":
    main()
