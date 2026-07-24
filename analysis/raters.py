"""Multi-rater loading, validation, and inter-rater agreement.

Rating sources (auto-detected, in priority order, all in the cache dir):
  1. blind_eval.xlsx          one sheet per rater (sheet name = rater id)  [preferred]
  2. blind_eval_<rater>.csv   one CSV per rater (e.g. blind_eval_author_1.csv)
  3. blind_eval.csv           single rater (backward compatible)

Every rating is validated on load (1-5, no blanks) and, with >1 rater, the
presentation_id sets must match EXACTLY -- otherwise we fail loudly.

Scores used downstream are the per-(presentation_id, dimension) MEAN across
raters. Agreement is reported separately: exact-match %, within-1 %, per-rater
means, and Pearson r -- with r suppressed/flagged when rating variance is near
zero (ceiling effects make r meaningless; read the agreement % instead).
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

RATING_COLS = {  # blind-sheet column -> short name (SummEval dimensions)
    "coherence_1to5": "coherence",
    "consistency_1to5": "consistency",
    "fluency_1to5": "fluency",
    "relevance_1to5": "relevance",
}
SUMM_DIMENSIONS = list(RATING_COLS.values())

RATING_MIN, RATING_MAX = 1, 5
# below this per-rater standard deviation the ratings are effectively constant
# (ceiling), so Pearson r is unstable/meaningless
LOW_VARIANCE_STD = 0.5


# ------------------------------------------------------------------ loading

def _rows_from_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rater_tables(cache_dir: Path) -> tuple[dict[str, list[dict]], str]:
    """Returns ({rater_id: rows}, source_description)."""
    xlsx = cache_dir / "blind_eval.xlsx"
    if xlsx.exists():
        import pandas as pd

        sheets = pd.read_excel(xlsx, sheet_name=None)
        tables = {
            str(name): df.to_dict("records")
            for name, df in sheets.items() if len(df)
        }
        if tables:
            return tables, f"blind_eval.xlsx (sheets: {', '.join(sorted(tables))})"

    per_rater_csvs = sorted(cache_dir.glob("blind_eval_*.csv"))
    if per_rater_csvs:
        tables = {p.stem.replace("blind_eval_", ""): _rows_from_csv(p)
                  for p in per_rater_csvs}
        return tables, f"{len(tables)} CSVs ({', '.join(p.name for p in per_rater_csvs)})"

    single = cache_dir / "blind_eval.csv"
    if single.exists():
        return {"rater_1": _rows_from_csv(single)}, "blind_eval.csv (single rater)"
    return {}, "(no rating file found)"


def _clean_rating(value) -> float | None:
    """Parse one rating cell; None if blank/unparseable."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_ratings(cache_dir: Path, valid_pids: set[str] | None = None):
    """Load + validate all raters. Returns (per_rater, averaged, source).

    per_rater: {rater: {presentation_id: {dimension: score}}}
    averaged:  {presentation_id: {dimension: mean score across raters}}
    Raises SystemExit listing every problem found.
    """
    tables, source = load_rater_tables(cache_dir)
    if not tables:
        return {}, {}, source

    problems: list[str] = []
    per_rater: dict[str, dict[str, dict[str, float]]] = {}

    for rater, rows in tables.items():
        scores: dict[str, dict[str, float]] = {}
        for i, row in enumerate(rows, start=2):  # +2 ~ spreadsheet row number
            pid = str(row.get("presentation_id", "")).strip()
            if not pid or pid.lower() == "nan":
                problems.append(f"{rater} row {i}: missing presentation_id")
                continue
            dims = {}
            for col, dim in RATING_COLS.items():
                v = _clean_rating(row.get(col))
                if v is None:
                    problems.append(f"{rater} {pid}: blank/unparseable {col}")
                elif not (RATING_MIN <= v <= RATING_MAX):
                    problems.append(f"{rater} {pid}: {col}={v} outside {RATING_MIN}-{RATING_MAX}")
                else:
                    dims[dim] = v
            if pid in scores:
                problems.append(f"{rater}: duplicate presentation_id {pid}")
            scores[pid] = dims
        per_rater[rater] = scores

    # all raters must cover exactly the same presentations
    pid_sets = {r: set(s) for r, s in per_rater.items()}
    if len(pid_sets) > 1:
        reference_rater, reference = next(iter(pid_sets.items()))
        for rater, pids in pid_sets.items():
            if pids != reference:
                missing = sorted(reference - pids)[:5]
                extra = sorted(pids - reference)[:5]
                problems.append(
                    f"rater '{rater}' presentation_ids differ from '{reference_rater}': "
                    f"missing={missing} extra={extra}"
                )
    # ratings must correspond to real presentations
    if valid_pids is not None:
        for rater, pids in pid_sets.items():
            unknown = sorted(pids - valid_pids)[:5]
            if unknown:
                problems.append(f"rater '{rater}': presentation_ids not in "
                                f"unblinding_key.csv: {unknown}")

    if problems:
        raise SystemExit("RATING VALIDATION FAILED (" + source + "):\n  "
                         + "\n  ".join(problems[:40]))

    averaged: dict[str, dict[str, float]] = {}
    all_pids = set().union(*pid_sets.values()) if pid_sets else set()
    for pid in all_pids:
        averaged[pid] = {}
        for dim in SUMM_DIMENSIONS:
            vals = [per_rater[r][pid][dim] for r in per_rater if dim in per_rater[r][pid]]
            if vals:
                averaged[pid][dim] = sum(vals) / len(vals)
    return per_rater, averaged, source


# ---------------------------------------------------------------- agreement

def _std(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return None  # zero variance -> undefined
    return num / (da * db)


def agreement_stats(per_rater: dict[str, dict[str, dict[str, float]]]) -> list[dict]:
    """Per-dimension agreement. With >2 raters, pairwise metrics are averaged."""
    raters = sorted(per_rater)
    if len(raters) < 2:
        return []
    pids = sorted(set.intersection(*(set(per_rater[r]) for r in raters)))
    rows = []
    for dim in SUMM_DIMENSIONS:
        vectors = {r: [per_rater[r][p][dim] for p in pids] for r in raters}
        exact, within1, rs = [], [], []
        for r1, r2 in itertools.combinations(raters, 2):
            a, b = vectors[r1], vectors[r2]
            exact.append(sum(x == y for x, y in zip(a, b)) / len(a))
            within1.append(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a))
            r = _pearson(a, b)
            if r is not None:
                rs.append(r)
        stds = {r: _std(v) for r, v in vectors.items()}
        min_std = min(stds.values())
        if min_std == 0:
            pearson, note = None, "undefined - zero variance (ceiling); use agreement %"
        elif min_std < LOW_VARIANCE_STD:
            pearson = sum(rs) / len(rs) if rs else None
            note = "UNSTABLE - near-zero variance (ceiling); use agreement %"
        else:
            pearson = sum(rs) / len(rs) if rs else None
            note = ""
        row = {
            "dimension": dim,
            "n": len(pids),
            "exact_match_pct": round(100 * sum(exact) / len(exact), 1),
            "within1_pct": round(100 * sum(within1) / len(within1), 1),
        }
        for r in raters:
            row[f"mean_{r}"] = round(sum(vectors[r]) / len(vectors[r]), 3)
            row[f"sd_{r}"] = round(stds[r], 3)
        row["pearson_r"] = round(pearson, 3) if pearson is not None else None
        row["r_note"] = note
        rows.append(row)
    return rows
