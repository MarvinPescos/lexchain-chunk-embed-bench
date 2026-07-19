"""Groq chat client (OpenAI-compatible) with rate-limit-aware, resumable backoff.

Groq free tier is tightly capped (a 70B-class model is ~100K tokens/day). The
runner checkpoints after every question, and this client is careful about 429s:

  - per-minute limit (RPM/TPM): short wait honoring `retry-after`, then retry
  - per-day limit (RPD/TPD):    raise DailyLimitError so the caller can flush and
                                stop cleanly; rerunning resumes where it left off

`chat()` returns a GenResult with the raw text, token usage, latency, and the
parsed `x-ratelimit-*` headers so the smoke test can report real headroom.

Only stdlib (urllib) is used, so nothing new is pinned. `MockGroqClient` mirrors
the interface for offline tests.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class DailyLimitError(Exception):
    """Raised when a 429 looks like a daily (not per-minute) exhaustion."""

    def __init__(self, model: str, retry_after: float, message: str):
        self.model = model
        self.retry_after = retry_after
        super().__init__(message)


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ratelimit: dict = field(default_factory=dict)


def _parse_ratelimit(headers) -> dict:
    out = {}
    for k in headers.keys():
        lk = k.lower()
        if lk.startswith("x-ratelimit") or lk == "retry-after":
            out[lk] = headers.get(k)
    return out


def _retry_after_seconds(headers, body: str, default: float) -> float:
    ra = headers.get("retry-after") if headers else None
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    # Groq sometimes embeds "try again in 2m30s" / "in 12.5s" in the body
    import re

    m = re.search(r"in ((?:\d+m)?[\d.]+s|\d+m[\d.]*s?)", body or "")
    if m:
        txt = m.group(1)
        secs = 0.0
        mm = re.search(r"(\d+)m", txt)
        ss = re.search(r"([\d.]+)s", txt)
        if mm:
            secs += int(mm.group(1)) * 60
        if ss:
            secs += float(ss.group(1))
        if secs:
            return secs
    return default


# a 429 whose reset is longer than this (or that names a daily quota) is treated
# as a daily exhaustion rather than something worth blocking the run on
DAILY_LIMIT_THRESHOLD_S = 20 * 60


class GroqClient:
    def __init__(self, api_key: str, *, max_retries: int = 6,
                 base_backoff: float = 2.0, timeout: float = 120.0,
                 max_tokens: int = 256, temperature: float = 0.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty")
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, model: str, system: str, user: str) -> GenResult:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        attempt = 0
        while True:
            req = urllib.request.Request(
                GROQ_URL, data=payload,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    latency = time.time() - t0
                    data = json.loads(resp.read().decode("utf-8"))
                    usage = data.get("usage", {})
                    return GenResult(
                        text=data["choices"][0]["message"]["content"] or "",
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_s=latency,
                        ratelimit=_parse_ratelimit(resp.headers),
                    )
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 429:
                    wait = _retry_after_seconds(e.headers, body,
                                                self.base_backoff * (2 ** attempt))
                    daily = wait > DAILY_LIMIT_THRESHOLD_S or "day" in body.lower()
                    if daily:
                        raise DailyLimitError(model, wait, body[:300])
                    attempt += 1
                    if attempt > self.max_retries:
                        raise
                    time.sleep(min(wait, 120))
                    continue
                if e.code in (500, 502, 503, 520, 524):
                    attempt += 1
                    if attempt > self.max_retries:
                        raise
                    time.sleep(self.base_backoff * (2 ** (attempt - 1)))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(self.base_backoff * (2 ** (attempt - 1)))


class MockGroqClient:
    """Offline stand-in. `answer_fn(model, user)->str` produces the raw text;
    `fail_plan` is a list of exceptions to raise on successive calls before
    succeeding (to exercise backoff / daily-limit handling)."""

    def __init__(self, answer_fn=None, fail_plan=None, latency=0.01):
        self.answer_fn = answer_fn or (lambda model, user: "<response>mock</response>")
        self.fail_plan = list(fail_plan or [])
        self.latency = latency
        self.calls = 0

    def chat(self, model: str, system: str, user: str) -> GenResult:
        self.calls += 1
        if self.fail_plan:
            exc = self.fail_plan.pop(0)
            if isinstance(exc, Exception):
                raise exc
        raw = self.answer_fn(model, user)
        return GenResult(text=raw, prompt_tokens=len(system + user) // 4,
                         completion_tokens=max(1, len(raw) // 4),
                         latency_s=self.latency,
                         ratelimit={"x-ratelimit-remaining-requests": "999"})
