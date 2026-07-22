#!/usr/bin/env python3
"""Build the blind human-scoring scaffold from the cached analyses (50 rows).

GUARDRAIL: this refuses to run until <cache>/ground_truth_key.csv exists and has
at least one filled row -- the ground-truth key must be hand-authored BLIND to
model outputs, and this file is the first place model outputs are rendered.
(--force exists for the CPU test-suite only.)

Outputs (in <cache>):
  blind_eval.csv    one row per (doc, model) = 10 x 5 = 50. Model identity is
                    hidden behind a per-doc randomized label (Output A..E) and
                    rows are shuffled. Human rating columns use the SummEval
                    dimensions: coherence, consistency, fluency, relevance
                    (1-5 Likert), plus hallucination notes.
  unblinding_key.csv presentation_id -> doc, label, model, role
                    (role marks the 70B "reference (non-deployable)" row).
Seeded (42) so blinding is reproducible. Do NOT open the key until rating is done.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .data import MODELS
from .prompt import ENTITY_CATEGORIES

BLIND_SEED = 42
LABELS = ["Output A", "Output B", "Output C", "Output D", "Output E"]

HUMAN_COLS = [
    "coherence_1to5",     # SummEval: collective quality / organization of the summary
    "consistency_1to5",   # SummEval: factual alignment with the source document
    "fluency_1to5",       # SummEval: grammatical / readable sentences
    "relevance_1to5",     # SummEval: captures the important content
    "hallucinations_notes",
]


def _fmt_entities(entities: dict) -> str:
    parts = []
    for cat in ENTITY_CATEGORIES:
        vals = entities.get(cat) or []
        parts.append(f"{cat}: " + ("; ".join(vals) if vals else "(none)"))
    return "\n".join(parts)


def _fmt_risks(risk_flags: list) -> str:
    """Checklist rendering: present categories with their quotes, then absents."""
    present = [f for f in risk_flags if f.get("present")]
    absent = [f["category"] for f in risk_flags if not f.get("present")]
    lines = [f"- {f['category']}: \"{f.get('quote', '')}\"" for f in present] or ["(none present)"]
    if absent:
        lines.append("absent: " + ", ".join(absent))
    return "\n".join(lines)


def load_analyses(cache_dir: Path) -> dict[str, dict[str, dict]]:
    """doc -> model -> parsed analysis (from checkpoints).

    Ignores checkpoints for models no longer in the registry, so stale outputs
    from a previous model set can never leak into the blind sheet."""
    out: dict[str, dict[str, dict]] = {}
    skipped = set()
    for path in sorted((cache_dir / "analyses").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["model"] not in MODELS:
            skipped.add(rec["model"])
            continue
        parsed = rec.get("parsed") or {"summary": "", "entities": {}, "risk_flags": []}
        out.setdefault(rec["doc"], {})[rec["model"]] = parsed
    if skipped:
        print(f"[build_blind_eval] ignoring checkpoints for de-registered models: "
              f"{sorted(skipped)} (delete them to reclaim space)")
    return out


def gt_key_ready(cache_dir: Path) -> bool:
    """True iff ground_truth_key.csv exists with >=1 row that has any filled cell."""
    path = cache_dir / "ground_truth_key.csv"
    if not path.exists():
        return False
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(v.strip() for k, v in row.items() if k != "doc_reference" and v):
                return True
    return False


def build(cache_dir: Path, force: bool = False) -> tuple[int, int]:
    if not force and not gt_key_ready(cache_dir):
        raise SystemExit(
            "GUARDRAIL: ground_truth_key.csv is missing or empty in "
            f"{cache_dir}.\nAuthor the key first (from doc_texts/, blind to model "
            "outputs), save it as ground_truth_key.csv, then re-run. "
            "The blind sheet renders model outputs and must come second."
        )
    analyses = load_analyses(cache_dir)
    if not analyses:
        raise SystemExit(f"no analyses in {cache_dir}/analyses (run analyze.py first)")
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
                "role": MODELS[model]["role"] if model in MODELS else "candidate",
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
                                          "output_label", "model", "role"])
        w.writeheader()
        w.writerows(sorted(key_rows, key=lambda r: r["presentation_id"]))

    return len(blind_rows), len(key_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_colab = Path("/content").exists()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/content/drive/MyDrive/lexchain_bench/analysis"
                                 if on_colab else ".cache_analysis"))
    ap.add_argument("--force", action="store_true",
                    help="bypass the ground-truth-first guardrail (tests only)")
    args = ap.parse_args()
    n_blind, _ = build(args.cache_dir, force=args.force)
    print(f"Wrote blind_eval.csv ({n_blind} rows) and unblinding_key.csv -> {args.cache_dir}")
    print("SummEval rating columns (1-5, 5 best): coherence / consistency / fluency / relevance,")
    print("plus hallucinations_notes. Do NOT open unblinding_key.csv until rating is done.")


if __name__ == "__main__":
    main()
