"""Entity/risk matching for precision/recall/F1 against the filled ground-truth key.

Matching rules (documented, tunable):
- parties / obligations / risk : rapidfuzz token_set_ratio >= FUZZY_THRESHOLD (85)
- dates                        : normalized to a canonical form, exact match
- monetary_amounts             : matched on the numeric value(s) they contain
Each predicted item matches at most one gold item (greedy best-first).
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 85.0

# category -> matcher name; used by score_category
FUZZY_CATS = {"parties", "obligations", "risk_clauses"}
DATE_CATS = {"dates"}
MONEY_CATS = {"monetary_amounts"}

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


# Common legal org-suffix variants collapsed to a canonical token so that e.g.
# "Acme Corporation" and "Acme Corp." match. Applied token-wise after norm.
_LEGAL_SUFFIX = {
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "l.l.c": "llc", "l.l.c.": "llc",
    "corp.": "corp", "inc.": "inc", "co.": "co", "ltd.": "ltd",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\s+", " ", re.sub(r"[^\w\s./-]", " ", s)).strip()
    return " ".join(_LEGAL_SUFFIX.get(tok, tok) for tok in s.split())


def _date_keys(s: str) -> set[str]:
    """Canonical (year, month, day) tokens found in a date string, for exact-ish match."""
    s = str(s).lower()
    keys = set()
    # ISO / numeric: 2020-06-02, 06/02/2020, 6-2-20
    for m in re.finditer(r"\b(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})\b", s):
        a, b, c = m.groups()
        year = a if len(a) == 4 else (c if len(c) == 4 else None)
        if year:
            nums = sorted(int(x) for x in (a, b, c) if x != year)
            keys.add(f"{year}:{':'.join(map(str, nums))}")
    # "June 2, 2020" / "2 June 2020"
    for m in re.finditer(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?", s):
        mon, day, year = m.groups()
        mon3 = mon[:4] if mon[:4] in _MONTHS else mon[:3]
        if mon3 in _MONTHS:
            keys.add(f"{year or '?'}:{_MONTHS[mon3]}:{int(day)}")
    for m in re.finditer(r"\b(\d{4})\b", s):  # bare year fallback
        keys.add(f"{m.group(1)}")
    return keys


def _money_keys(s: str) -> set[str]:
    """Numeric values in a monetary string (commas/decimals normalized)."""
    keys = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", str(s)):
        num = m.group(0).replace(",", "")
        try:
            keys.add(str(round(float(num), 2)))
        except ValueError:
            pass
    return keys


def _match(category: str, a: str, b: str) -> bool:
    if category in DATE_CATS:
        return bool(_date_keys(a) & _date_keys(b))
    if category in MONEY_CATS:
        return bool(_money_keys(a) & _money_keys(b))
    return fuzz.token_set_ratio(_norm(a), _norm(b)) >= FUZZY_THRESHOLD


def score_category(category: str, predicted: list[str], gold: list[str]) -> dict:
    """Greedy one-to-one matching -> tp/fp/fn (+ precision/recall/f1)."""
    pred = [p for p in (predicted or []) if str(p).strip()]
    gold = [g for g in (gold or []) if str(g).strip()]
    used = [False] * len(gold)
    tp = 0
    for p in pred:
        for j, g in enumerate(gold):
            if not used[j] and _match(category, p, g):
                used[j] = True
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold) - tp
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    return _prf(tp, fp, fn)
