"""The single fixed prompt + JSON schema for the document-analysis comparison.

The SAME prompt is sent to all three models (Llama 3.1 70B, Qwen2.5 72B,
Mixtral 8x22B) so the only variable is the model. The task mirrors LexChain's
own analysis schema: a summary, structured entities, and risk flags.
"""

from __future__ import annotations

import json
import re

# Output schema (documented for the report + used by the parser/aggregator).
# entities is an object of four string-lists; risk_flags a list of objects.
SCHEMA_KEYS = ["summary", "entities", "risk_flags"]
ENTITY_CATEGORIES = ["parties", "dates", "monetary_amounts", "obligations"]

SYSTEM_PROMPT = (
    "You are a legal document-analysis engine for LexChain. You read a legal "
    "document and return a single JSON object describing it. Be faithful: never "
    "invent facts, parties, dates, amounts, or clauses that are not in the text."
)

# The fixed user prompt. {document} is filled with the document text.
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
    {{"risk": "short description of a clause a lawyer should watch (e.g. indemnification, unlimited liability, auto-renewal, termination for convenience, governing law, penalty)", "severity": "low | medium | high"}}
  ]
}}

Rules:
- Return ONLY the JSON object, no prose, no markdown fences.
- Use [] for any list with no items; never omit a key.
- Do not invent content; if the document does not state something, leave that list empty.

DOCUMENT:
\"\"\"
{document}
\"\"\"
"""


def build_messages(document: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document=document)},
    ]


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip())


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


def normalize_analysis(obj: dict) -> dict:
    """Coerce a parsed object into the canonical schema shape (fill missing keys,
    normalize types) so downstream code can rely on it."""
    out = {"summary": "", "entities": {}, "risk_flags": []}
    if isinstance(obj.get("summary"), str):
        out["summary"] = obj["summary"].strip()
    ent = obj.get("entities") or {}
    for cat in ENTITY_CATEGORIES:
        vals = ent.get(cat) if isinstance(ent, dict) else None
        if isinstance(vals, str):
            vals = [vals]
        out["entities"][cat] = [str(v).strip() for v in vals if str(v).strip()] if isinstance(vals, list) else []
    flags = obj.get("risk_flags") or []
    norm_flags = []
    if isinstance(flags, list):
        for f in flags:
            if isinstance(f, dict):
                risk = str(f.get("risk") or f.get("description") or "").strip()
                sev = str(f.get("severity") or "").strip().lower()
            else:
                risk, sev = str(f).strip(), ""
            if risk:
                norm_flags.append({"risk": risk, "severity": sev})
    out["risk_flags"] = norm_flags
    return out


def parse_analysis(text: str) -> dict | None:
    """Parse a model response into the canonical schema, or None if unparseable.
    Tries: direct json, fence-stripped, first balanced object."""
    for candidate in (text, _strip_fences(text), _first_json_object(text) or ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return normalize_analysis(obj)
        except json.JSONDecodeError:
            continue
    return None
