#!/usr/bin/env python3
"""Probe OpenRouter models for zero-data-retention (ZDR) routing.

For each candidate model id, make ONE minimal chat call with the exact provider
preferences the benchmark uses (data.PROVIDER_PREFS = data_collection:deny,
allow_fallbacks:false, zdr:true). Report which route to a ZDR-compliant provider
(OK) vs which fail (e.g. 404 "no endpoints found matching your data policy").

Also prints each model's catalog context window so you can see, at a glance,
which candidates are even usable given our ~21.6k-token worst-case document
(a model whose context is below that is unusable regardless of ZDR).

Run on Colab (or anywhere OPENROUTER_API_KEY is set):
  python -m analysis.probe_zdr
  python -m analysis.probe_zdr --models mistralai/mistral-large-2407,google/gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

from .data import CTX_SAFETY_MARGIN, MAX_OUTPUT_TOKENS, PROVIDER_PREFS

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default candidates for the third-model slot (context-viable, current ids).
# gemma-2-27b-it and deepseek-r1-distill-llama-70b are omitted: 8k context < our
# worst-case document, so they are unusable no matter their data policy.
DEFAULT_CANDIDATES = [
    "mistralai/mistral-large-2407",   # Mistral, 123B dense (above 70B tier)
    "mistralai/mistral-large-2512",   # Mistral, newer/larger
    "deepseek/deepseek-chat-v3.1",    # DeepSeek, large MoE instruct
    "google/gemma-3-27b-it",          # Google, 27B (below 70B tier)
]

# our worst-case prompt budget (largest doc + prompt + output + margin)
WORST_CASE_TOKENS = 18_444 + 619 + MAX_OUTPUT_TOKENS + CTX_SAFETY_MARGIN  # ~21.6k


def catalog_context() -> dict[str, int]:
    try:
        with urllib.request.urlopen(OPENROUTER_BASE_URL + "/models", timeout=30) as r:
            return {m["id"]: m.get("context_length") for m in json.load(r)["data"]}
    except Exception:
        return {}


def probe(model_id: str, client) -> tuple[str, str]:
    """Returns (status, detail). status in {'OK','FAILED'}."""
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0,
            max_tokens=5,
            extra_body={"provider": dict(PROVIDER_PREFS)},
        )
        text = (resp.choices[0].message.content or "").strip()
        provider = getattr(resp, "provider", None) or "?"
        return "OK", f"routed via {provider}; reply={text!r}"
    except Exception as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
        if isinstance(body, dict):
            body = body.get("message") or json.dumps(body)
        return "FAILED", f"HTTP {status}: {str(body)[:200]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(DEFAULT_CANDIDATES))
    args = ap.parse_args()
    ids = [m.strip() for m in args.models.split(",") if m.strip()]

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (Colab: add it as a secret)")
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    ctx = catalog_context()

    print(f"ZDR probe with provider prefs = {PROVIDER_PREFS}")
    print(f"(usable requires context >= our worst case ~{WORST_CASE_TOKENS:,} tokens)\n")
    print(f"{'model':40s} {'ctx':>9s} {'usable':>7s}  ZDR")
    results = {}
    for mid in ids:
        c = ctx.get(mid)
        usable = "yes" if (c and c >= WORST_CASE_TOKENS) else "NO"
        status, detail = probe(mid, client)
        results[mid] = status
        flag = "OK " if status == "OK" else "404"
        print(f"{mid:40s} {str(c):>9s} {usable:>7s}  {flag}  {detail}")

    ok = [m for m, s in results.items() if s == "OK"]
    print(f"\nZDR-OK: {ok or '(none)'}")
    print("Pick a ZDR-OK model whose context clears the worst case, then add it "
          "to analysis/data.py MODELS as the third model.")


if __name__ == "__main__":
    main()
