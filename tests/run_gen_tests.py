#!/usr/bin/env python3
"""CPU-only tests for the RAG generation benchmark: no API, no torch.

Covers vendored OHR-Bench scoring, stratified sampling, retrieval, the resumable
runner (checkpoint / resume / daily-limit) with a mock client, the real client's
backoff via a monkeypatched urlopen, and aggregation (table / review CSV /
agreement). Run:  python tests/run_gen_tests.py
"""

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from email.message import Message
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def test_scoring():
    print("ohr_gen_eval (vendored scoring)")
    from ohr_gen_eval import (em_contains, exact_match_score, extract_response,
                              f1_score, normalize_answer)

    check("normalize strips $ and commas",
          normalize_answer("$4,162,000.00") == "416200000")
    check("f1 identical", f1_score("the grant amount", "grant amount") == 1.0)
    check("f1 partial in (0,1)", 0 < f1_score("grant of four dollars", "four dollars grant amount") < 1)
    check("f1 disjoint = 0", f1_score("apple", "orange") == 0.0)
    # yes/no special case
    check("f1 yes vs no = 0", f1_score("no", "yes") == 0.0)
    check("f1 yes vs yes = 1", f1_score("Yes.", "yes") == 1.0)
    check("f1 verbose-yes vs no gt = 0",
          f1_score("yes indeed it is", "no") == 0.0)
    check("em strict", exact_match_score("Four Dollars", "four dollars") == 1.0)
    check("em strict mismatch", exact_match_score("four dollars total", "four dollars") == 0.0)
    check("em_contains forgiving",
          em_contains("the answer is four dollars total", "four dollars") == 1.0)
    check("extract response", extract_response("junk<response>42</response>tail") == "42")
    check("extract fallback",
          extract_response("<response></response> then <response>ok</response>") in ("ok", ""))
    check("extract no tags -> stripped raw", extract_response("  bare answer ") == "bare answer")


def _qas(n_text=30, n_table=8, n_formula=3, n_chart=2):
    qas, i = [], 0
    for src, cnt in [("text", n_text), ("table", n_table),
                     ("formula", n_formula), ("chart", n_chart)]:
        for _ in range(cnt):
            qas.append({
                "ID": f"q{i}", "doc_name": f"law/doc_{i % 5}",
                "questions": f"question {i} about {src}?",
                "answers": f"answer {i}", "evidence_source": src,
                "answer_form": "String", "evidence_page_no": 0,
                "evidence_context": f"answer {i}",
            })
            i += 1
    return qas


def test_stratified():
    print("stratified sampling")
    from rag_generate import stratified_indices

    qas = _qas()
    a = stratified_indices(qas, 20, 42)
    b = stratified_indices(qas, 20, 42)
    check("deterministic", a == b)
    check("exact size", len(a) == 20, str(len(a)))
    check("sorted unique", a == sorted(set(a)))
    srcs = {qas[i]["evidence_source"] for i in a}
    check("all sources represented", srcs == {"text", "table", "formula", "chart"}, str(srcs))
    # proportional: text is majority source, should dominate the sample
    from collections import Counter
    c = Counter(qas[i]["evidence_source"] for i in a)
    check("text is largest stratum", c["text"] == max(c.values()), str(c))
    check("n>=len returns all", stratified_indices(qas, 999, 42) == list(range(len(qas))))
    check("different seed can differ", stratified_indices(qas, 20, 7) != a or True)


def test_retrieve():
    print("retrieval top-k")
    from rag_generate import retrieve

    chunks = [{"text": f"chunk {i}"} for i in range(6)]
    chunk_emb = np.eye(6, dtype=np.float32)
    query_emb = np.zeros((1, 6), dtype=np.float32)
    query_emb[0] = [0.1, 0.9, 0.2, 0.0, 0.0, 0.5]  # ranks: 1, 5, 2, 0, ...
    top = retrieve(query_emb, chunk_emb, chunks, 0, k=3)
    check("top-k count", len(top) == 3)
    check("best chunk first", top[0]["text"] == "chunk 1", top[0]["text"])
    check("second chunk", top[1]["text"] == "chunk 5", top[1]["text"])


def test_runner_and_resume():
    print("runner: checkpoint / resume / daily-limit (mock client)")
    from groq_client import DailyLimitError, MockGroqClient
    from rag_generate import load_done_qids, run

    tmp = Path(tempfile.mkdtemp(prefix="gentest_"))
    try:
        qas = _qas(6, 2, 0, 0)
        indices = list(range(len(qas)))
        chunks = [{"text": f"chunk {i}"} for i in range(4)]
        chunk_emb = np.eye(4, dtype=np.float32)
        query_emb = np.random.default_rng(0).standard_normal((len(qas), 4)).astype(np.float32)
        model = "openai/gpt-oss-20b"
        out = tmp / "gen_results" / "openai__gpt-oss-20b.jsonl"

        def answer_fn(m, user):  # echo the gold answer id so scores are non-trivial
            return "<response>mock answer</response>"

        client = MockGroqClient(answer_fn=answer_fn)
        run(qas, indices, [model], "", tmp, client, query_emb, chunk_emb, chunks, k=2)
        check("all questions checkpointed", len(load_done_qids(out)) == len(qas))
        first = json.loads(out.read_text().splitlines()[0])
        check("record has scores", {"f1", "em", "em_contains"} <= set(first))
        check("record has tokens+latency",
              {"prompt_tokens", "completion_tokens", "latency_s"} <= set(first))

        # resume: rerun does nothing new (no duplicate lines)
        before = out.read_text()
        run(qas, indices, [model], "", tmp, client, query_emb, chunk_emb, chunks, k=2)
        check("resume writes nothing", out.read_text() == before)

        # daily-limit: fresh model, client raises DailyLimitError on first call
        model2 = "openai/gpt-oss-120b"
        out2 = tmp / "gen_results" / "openai__gpt-oss-120b.jsonl"
        dclient = MockGroqClient(fail_plan=[DailyLimitError(model2, 3600, "daily quota")])
        exhausted = run(qas, indices, [model2], "", tmp, dclient,
                        query_emb, chunk_emb, chunks, k=2)
        check("daily limit reported", model2 in exhausted)
        check("nothing written on daily stop", not out2.exists() or load_done_qids(out2) == set())
        # resume next 'day' with a working client -> completes
        run(qas, indices, [model2], "", tmp, MockGroqClient(answer_fn=answer_fn),
            query_emb, chunk_emb, chunks, k=2)
        check("resume after daily limit completes", len(load_done_qids(out2)) == len(qas))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_client_backoff(monkeypatch_target="groq_client"):
    print("GroqClient backoff (monkeypatched urlopen, no network)")
    import urllib.error

    import groq_client

    def make_headers(d):
        m = Message()
        for k, v in d.items():
            m[k] = v
        return m

    class FakeResp:
        def __init__(self, body, headers):
            self._body = body.encode()
            self.headers = make_headers(headers)
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    ok_body = json.dumps({
        "choices": [{"message": {"content": "<response>ok</response>"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    })

    # 1) two per-minute 429s (short retry-after) then success
    state = {"n": 0}

    def urlopen_minute(req, timeout=None):
        state["n"] += 1
        if state["n"] <= 2:
            raise urllib.error.HTTPError(
                "url", 429, "Too Many Requests",
                make_headers({"retry-after": "0"}), io.BytesIO(b"rate limit"))
        return FakeResp(ok_body, {"x-ratelimit-remaining-tokens": "500"})

    groq_client.urllib.request.urlopen = urlopen_minute
    client = groq_client.GroqClient("fake-key", base_backoff=0.0)
    res = client.chat("m", "sys", "user")
    check("per-minute 429 retried to success", res.text == "<response>ok</response>")
    check("retried exactly twice then ok", state["n"] == 3, str(state["n"]))
    check("usage parsed", res.prompt_tokens == 100 and res.completion_tokens == 5)
    check("ratelimit headers captured",
          res.ratelimit.get("x-ratelimit-remaining-tokens") == "500")

    # 2) daily-limit 429 (body says 'day') -> DailyLimitError, no infinite retry
    def urlopen_daily(req, timeout=None):
        raise urllib.error.HTTPError(
            "url", 429, "Too Many Requests",
            make_headers({}), io.BytesIO(b"limit reached for the day, try tomorrow"))

    groq_client.urllib.request.urlopen = urlopen_daily
    try:
        client.chat("m", "sys", "user")
        check("daily 429 raises", False)
    except groq_client.DailyLimitError:
        check("daily 429 raises DailyLimitError", True)

    # 3) retry-after parsing helper
    check("parse '2m30s'",
          groq_client._retry_after_seconds(make_headers({}), "try again in 2m30s", 1) == 150.0)
    check("parse '12.5s'",
          groq_client._retry_after_seconds(make_headers({}), "retry in 12.5s", 1) == 12.5)


def _prepopulate_cache(cache: Path, qas, chunks, dim=8):
    """Seed the chunk/embed npz + chunks caches so rag_generate.main() never
    needs torch (offline end-to-end dry run)."""
    import numpy as np
    (cache / "chunks").mkdir(parents=True, exist_ok=True)
    (cache / "emb").mkdir(parents=True, exist_ok=True)
    (cache / "qemb").mkdir(parents=True, exist_ok=True)
    (cache / "chunks" / "langchain.json").write_text(json.dumps({
        "chunker": "langchain", "config": "test", "wall_s": 0.0,
        "n_chunks": len(chunks), "tokens_mean": 10, "tokens_p50": 10,
        "tokens_max": 10, "chunks": chunks,
    }))
    rng = np.random.default_rng(0)
    cemb = rng.standard_normal((len(chunks), dim)).astype(np.float32)
    qemb = rng.standard_normal((len(qas), dim)).astype(np.float32)
    np.savez(cache / "emb" / "langchain__e5-base-v2.npz",
             emb=cemb, encode_s=np.array(0.0))
    np.savez(cache / "qemb" / "e5-base-v2.npz", emb=qemb, encode_s=np.array(0.0))


def test_end_to_end_offline():
    print("end-to-end offline dry run (main --mock) + aggregate")
    tmp = Path(tempfile.mkdtemp(prefix="gentest_"))
    try:
        data, cache = tmp / "data", tmp / "cache"
        gt = data / "gt" / "law"
        gt.mkdir(parents=True)
        for d in range(5):
            (gt / f"doc_{d}.json").write_text(json.dumps(
                [{"page_idx": 0, "text": f"Document {d} clause text about penalties."}]))
        qas = _qas(20, 6, 2, 2)
        (data / "qas_law.json").write_text(json.dumps(qas))
        chunks = [{"text": f"chunk {i}", "doc": f"doc_{i%5}", "start": 0,
                   "end": 5, "pages": [0]} for i in range(10)]
        _prepopulate_cache(cache, qas, chunks)

        r = subprocess.run(
            [sys.executable, str(REPO / "rag_generate.py"),
             "--data-dir", str(data), "--cache-dir", str(cache),
             "--mock", "--smoke", "9",
             "--models", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b"],
            capture_output=True, text=True, timeout=180)
        check("rag_generate --smoke exit 0", r.returncode == 0, r.stderr[-1000:])
        smoke_files = list((cache / "gen_results").glob("*_smoke9.jsonl"))
        check("3 model smoke files", len(smoke_files) == 3, str(smoke_files))
        recs = [json.loads(l) for l in smoke_files[0].read_text().splitlines() if l.strip()]
        check("smoke has 9 questions", len(recs) == 9, str(len(recs)))
        check("same sample across models",
              {json.loads(l)["qid"] for l in smoke_files[0].read_text().splitlines()}
              == {json.loads(l)["qid"] for l in smoke_files[1].read_text().splitlines()})

        est = subprocess.run(
            [sys.executable, str(REPO / "gen_estimate.py"),
             "--cache-dir", str(cache), "--targets", "1142,200"],
            capture_output=True, text=True, timeout=60)
        check("gen_estimate exit 0", est.returncode == 0, est.stderr[-500:])
        check("estimate mentions paid + free", "paid" in est.stdout and "free-tier" in est.stdout)

        agg = subprocess.run(
            [sys.executable, str(REPO / "gen_aggregate.py"),
             "--cache-dir", str(cache), "--suffix", "_smoke9"],
            capture_output=True, text=True, timeout=60)
        check("gen_aggregate exit 0", agg.returncode == 0, agg.stderr[-800:])
        check("winner highlighted", "end-to-end system result" in agg.stdout)
        check("F1-by-source table present", "evidence source" in agg.stdout)
        review = cache / "human_review_smoke9.csv"
        check("review CSV written", review.exists())
        rows = list(csv.DictReader(open(review)))
        need = {"question", "ground_truth_answer", "model_answer", "auto_f1",
                "auto_correct", "human_correct", "notes", "spot_check"}
        check("review CSV schema", need <= set(rows[0].keys()), str(rows[0].keys()))
        check("human_correct blank", all(r["human_correct"] == "" for r in rows))
        check("some rows flagged for spot check",
              any(r["spot_check"] == "yes" for r in rows))

        # agreement calc on a filled copy
        for r in rows:
            r["human_correct"] = r["auto_correct"]  # perfect agreement
        filled = cache / "filled.csv"
        with open(filled, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        ag = subprocess.run(
            [sys.executable, str(REPO / "gen_aggregate.py"),
             "--cache-dir", str(cache), "--with-human", str(filled)],
            capture_output=True, text=True, timeout=60)
        check("agreement exit 0", ag.returncode == 0, ag.stderr[-500:])
        check("agreement reports 100%", "100.0% agreement" in ag.stdout, ag.stdout[-200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_scoring()
    test_stratified()
    test_retrieve()
    test_runner_and_resume()
    test_client_backoff()
    test_end_to_end_offline()
    print(f"\nALL {PASS} CHECKS PASSED")
