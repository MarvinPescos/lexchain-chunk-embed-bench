"""The single FROZEN prompt + JSON schema for the document-analysis benchmark.

PROMPT_VERSION 2.0-cuad-checklist: the risk section is a fixed checklist of
clause categories derived from CUAD's expert-annotated clause categories
(Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract
Review", 2021). For every category the model must state present/absent and
quote the clause verbatim when present.

The SAME prompt is sent to all 5 models (4 Ollama candidates + the NIM 70B
reference). Do not edit without bumping PROMPT_VERSION -- results across
versions are not comparable.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "2.0-cuad-checklist"

SCHEMA_KEYS = ["summary", "entities", "risk_flags"]
ENTITY_CATEGORIES = ["parties", "dates", "monetary_amounts", "obligations"]

# (category, what to look for) -- subset of CUAD clause categories relevant to
# LexChain's risk schema, hardcoded and frozen.
RISK_CATEGORIES: list[tuple[str, str]] = [
    ("indemnification", "a party must indemnify / hold harmless the other"),
    ("limitation_of_liability", "a cap or limit on either party's liability"),
    ("uncapped_liability", "liability expressly uncapped, or carve-outs that remove the cap"),
    ("auto_renewal", "the term renews automatically unless notice is given"),
    ("unilateral_termination", "a party may terminate for convenience / without cause"),
    ("liquidated_damages", "penalty or pre-agreed damages amount for breach or delay"),
    ("exclusivity", "exclusive dealing, supply, licensing or purchase restriction"),
    ("non_compete", "restriction on competing business activity"),
    ("governing_law", "governing law and/or jurisdiction / venue"),
    ("anti_assignment", "consent required to assign the agreement or its rights"),
    ("change_of_control", "rights or obligations triggered by merger / acquisition / change of control"),
    ("insurance_obligations", "a party must obtain or maintain insurance"),
]

RISK_CATEGORY_NAMES = [c for c, _ in RISK_CATEGORIES]

SYSTEM_PROMPT = (
    "You are a legal document-analysis engine for LexChain. You read a legal "
    "document and return a single JSON object describing it. Be faithful: never "
    "invent facts, parties, dates, amounts, or clauses that are not in the text. "
    "Return only JSON, with no additional prose or reasoning text."
)

_CHECKLIST_LINES = "\n".join(f'  - "{name}": {desc}' for name, desc in RISK_CATEGORIES)

# The frozen user prompt. {document} is filled with the document text.
USER_PROMPT_TEMPLATE = """Analyze the following legal document and return ONLY a JSON object with exactly these keys:

{{
  "summary": "a faithful, self-contained summary of about 8 sentences (~180 words) covering the key parties, dates, obligations, amounts and conditions",
  "entities": {{
    "parties": ["every named party / signatory / organization or person bound by the document"],
    "dates": ["every meaningful date: effective date, execution date, deadlines, term dates (quote as written)"],
    "monetary_amounts": ["every monetary amount with its context, e.g. '$4,162,000.00 grant'"],
    "obligations": ["each distinct obligation or covenant, phrased as 'who must do what'"]
  }},
  "risk_flags": [
    {{"category": "<category name from the checklist below>", "present": true or false, "quote": "verbatim excerpt of the clause if present, else an empty string"}}
  ]
}}

Risk checklist -- "risk_flags" MUST contain exactly one object for EACH of these 12 categories, in this order:
{checklist}

Rules:
- Return ONLY the JSON object: no prose, no markdown fences, no reasoning text.
- "risk_flags" must have exactly 12 entries, one per checklist category, each with "category", "present", and "quote".
- A "quote" must be copied verbatim from the document (an excerpt is fine); use "" when the category is absent.
- Use [] for any entity list with no items; never omit a key.
- Do not invent content; if the document does not state something, mark it absent / leave the list empty.

DOCUMENT:
\"\"\"
{document}
\"\"\"
"""


def build_messages(document: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                checklist=_CHECKLIST_LINES, document=document
            ),
        },
    ]


def prompt_overhead_text() -> str:
    """The full prompt text minus the document -- for context-budget accounting."""
    return SYSTEM_PROMPT + USER_PROMPT_TEMPLATE.format(checklist=_CHECKLIST_LINES, document="")


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (Qwen3 etc.). Also drops an
    unterminated trailing <think> block defensively."""
    text = _THINK_BLOCK.sub("", text)
    lone = text.lower().find("<think>")
    if lone != -1:
        text = text[:lone]
    return text.strip()


_FENCE_ANY = re.compile(r"```(?:json)?", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# llama3.1:8b drops the "quote": key for absent risk categories and writes a
# bare "" value: {"category":"exclusivity","present":false, ""}. Restore the key.
# (`, ""}` only occurs in this malformed shape; a legit `"quote": ""}` has the
# key before the comma is consumed, so this never corrupts valid JSON.)
_BARE_EMPTY_QUOTE = re.compile(r',\s*""\s*}')


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip())


def _repair_json(text: str) -> str:
    """Best-effort cleanup for small-model JSON quirks (only used as a fallback
    when strict parsing fails): restore a dropped "quote": key, remove markdown
    fences anywhere, and drop trailing commas before } or ]. Safe on valid JSON."""
    text = _FENCE_ANY.sub("", text)
    text = _BARE_EMPTY_QUOTE.sub(', "quote": ""}', text)
    text = _TRAILING_COMMA.sub(r"\1", text)
    return text.strip()


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced {...} block (models sometimes add stray prose)."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "present", "1")


def normalize_analysis(obj: dict) -> dict:
    """Coerce a parsed object into the canonical schema shape: fill missing keys,
    normalize types, and align risk_flags to the fixed 12-category checklist."""
    out = {"summary": "", "entities": {}, "risk_flags": []}
    if isinstance(obj.get("summary"), str):
        out["summary"] = obj["summary"].strip()
    ent = obj.get("entities") or {}
    for cat in ENTITY_CATEGORIES:
        vals = ent.get(cat) if isinstance(ent, dict) else None
        if isinstance(vals, str):
            vals = [vals]
        out["entities"][cat] = (
            [str(v).strip() for v in vals if str(v).strip()] if isinstance(vals, list) else []
        )

    # index whatever the model emitted by (normalized) category name
    emitted: dict[str, dict] = {}
    flags = obj.get("risk_flags") or []
    if isinstance(flags, list):
        for f in flags:
            if not isinstance(f, dict):
                continue
            cat = str(f.get("category") or f.get("risk") or "").strip().lower().replace(" ", "_")
            if cat:
                emitted[cat] = f
    # canonical order: exactly one entry per checklist category
    for cat in RISK_CATEGORY_NAMES:
        f = emitted.get(cat, {})
        present = _to_bool(f.get("present", False))
        quote = str(f.get("quote") or "").strip()
        out["risk_flags"].append(
            {"category": cat, "present": present, "quote": quote if present else ""}
        )
    return out


def validate_schema(parsed: dict) -> list[str]:
    """Post-parse schema validation. Returns a list of problems ([] = valid)."""
    problems = []
    if not parsed.get("summary"):
        problems.append("empty summary")
    ent = parsed.get("entities", {})
    for cat in ENTITY_CATEGORIES:
        if cat not in ent:
            problems.append(f"missing entities.{cat}")
    flags = parsed.get("risk_flags", [])
    if [f.get("category") for f in flags] != RISK_CATEGORY_NAMES:
        problems.append("risk_flags do not cover the 12 checklist categories in order")
    for f in flags:
        if f.get("present") and not f.get("quote"):
            problems.append(f"risk '{f.get('category')}' marked present without a quote")
    return problems


def parse_analysis(text: str) -> dict | None:
    """Parse a model response into the canonical schema, or None if unparseable.

    Tries, in order of increasing aggressiveness: think-stripped raw, fence-
    stripped, first balanced {...} object, then repaired variants of each
    (fences removed anywhere + trailing commas dropped)."""
    text = strip_thinking(text)
    candidates: list[str] = []
    for base in (text, _strip_fences(text)):
        candidates.append(base)
        inner = _first_json_object(base)
        if inner:
            candidates.append(inner)
    candidates += [_repair_json(c) for c in list(candidates)]

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return normalize_analysis(obj)
        except json.JSONDecodeError:
            continue
    return None
