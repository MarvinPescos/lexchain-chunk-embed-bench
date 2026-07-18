#!/usr/bin/env python3
"""CPU-only tests: matching (vendored OHR-Bench LCS), chunker adapters,
ranking metrics, and the resumable bench pipeline with the fake embedder.

Run:  python tests/run_tests.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def test_matching():
    print("matching (vendored OHR-Bench functions)")
    from matching import evidence_pages, evidence_text, lcs_score, normalize_answer

    check("normalize lowercases/strips punct",
          normalize_answer("The Court's ORDER, dated 2021!") == "courts order dated 2021")
    check("lcs full containment", lcs_score("p q r s t u", "q r s") == 1.0)
    check("lcs half", lcs_score("x q y s z", "p q r s") == 0.5)
    check("lcs order-sensitive", lcs_score("s r q p", "p q r s") == 0.25)
    check("lcs empty gold -> 0.5 (OHR convention)", lcs_score("anything", "") == 0.5)
    check("evidence list joined",
          evidence_text({"evidence_context": ["a", "b"]}) == "a\nb")
    check("evidence str passthrough",
          evidence_text({"evidence_context": "a"}) == "a")
    check("pages int -> set", evidence_pages({"evidence_page_no": 3}) == {3})
    check("pages list -> set", evidence_pages({"evidence_page_no": [1, 2]}) == {1, 2})


def _synthetic_doc(n_pages=8):
    # each "page" is a single long paragraph (> chunk budget), so splitters must
    # split inside it — the only case where LangChain's recursive splitter
    # actually carries chunk_overlap (whole small splits are never re-included)
    pages = []
    for i in range(n_pages):
        pages.append(
            f"Section {i}. This is paragraph number {i} of the synthetic legal "
            f"agreement, containing obligations, covenants and representations "
            f"specific to clause {i} which the parties hereby acknowledge. " * 30
        )
    return pages


def test_chunkers():
    print("chunkers (real libraries, comparable configs)")
    from chunkers import CHUNKERS, chunk_corpus, count_tokens

    docs = {"synth_doc": _synthetic_doc()}
    for name in CHUNKERS:
        chunks, config, wall = chunk_corpus(name, docs)
        check(f"{name}: produced chunks", len(chunks) > 3, str(len(chunks)))
        sizes = [count_tokens(c["text"]) for c in chunks]
        check(f"{name}: sizes near budget", max(sizes) <= 512 + 64,
              f"max {max(sizes)}")
        check(f"{name}: page attribution present",
              all(c["pages"] for c in chunks))
        pages_covered = set().union(*(c["pages"] for c in chunks))
        check(f"{name}: all pages covered", len(pages_covered) == 8,
              str(len(pages_covered)))
        check(f"{name}: config documented", "512" in config, config)
        joined = " ".join(c["text"] for c in chunks)
        check(f"{name}: text preserved", "paragraph number 7" in joined)
    # overlap sanity where the library supports it natively
    for name in ("langchain", "llamaindex"):
        chunks, _, _ = chunk_corpus(name, docs)
        overlaps = sum(
            1 for a, b in zip(chunks, chunks[1:])
            if a["doc"] == b["doc"] and b["start"] < a["end"]
        )
        check(f"{name}: consecutive chunks overlap", overlaps > 0)


def test_metrics_ranking():
    print("ranking metrics on synthetic rankings")
    import numpy as np

    from bench import score_config

    # 4 chunks in one doc/page; evidence = chunk 2's text
    chunks = [
        {"text": f"unrelated filler text {i} nothing here", "doc": "d", "pages": [0]}
        for i in range(3)
    ]
    chunks.insert(2, {"text": "the grant amount was four million dollars exactly",
                      "doc": "d", "pages": [0]})
    qa = {"doc_name": "law/d", "questions": "q",
          "evidence_context": "grant amount was four million dollars",
          "evidence_page_no": 0}
    rels = [{2}]

    # craft embeddings so ranking is [2, 0, 1, 3] -> first relevant at rank 1
    chunk_emb = np.eye(4, dtype=np.float32)
    q = np.array([[0.4, 0.3, 0.9, 0.1]], dtype=np.float32)
    q /= np.linalg.norm(q)
    m = score_config(chunks, [qa], rels, chunk_emb, q, top_k=10)
    check("recall@1 hit", m["recall@1"] == 1.0, str(m))
    check("mrr@5 = 1", m["mrr@5"] == 1.0)
    check("ndcg@10 = 1 (single relevant at rank 1)", abs(m["ndcg@10"] - 1.0) < 1e-9)
    check("lcs@5 counts gated chunks", m["lcs@5"] == 1.0, str(m))
    check("coverage", m["coverage"] == 1.0)

    # push relevant chunk to rank 2 -> r@1=0, mrr=0.5, ndcg = 1/log2(3)
    q2 = np.array([[0.9, 0.1, 0.5, 0.1]], dtype=np.float32)
    q2 /= np.linalg.norm(q2)
    m2 = score_config(chunks, [qa], rels, chunk_emb, q2, top_k=10)
    check("recall@1 miss", m2["recall@1"] == 0.0)
    check("recall@5 still hit", m2["recall@5"] == 1.0)
    check("mrr@5 = 0.5", m2["mrr@5"] == 0.5)
    import math
    check("ndcg@10 = 1/log2(3)", abs(m2["ndcg@10"] - 1 / math.log2(3)) < 1e-9)


def _write_synth_data(data_dir: Path, n_docs=6):
    gt_dir = data_dir / "gt" / "law"
    gt_dir.mkdir(parents=True)
    qas = []
    for d in range(n_docs):
        stem = f"doc_{d:02d}"
        pages = []
        for p in range(3):
            pages.append({"page_idx": p,
                          "text": f"Filler provisions for {stem} page {p}. " * 30
                                  + f"The unique penalty clause of {stem} page {p} "
                                    f"imposes a fine of {d}{p} thousand dollars."})
        (gt_dir / f"{stem}.json").write_text(json.dumps(pages))
        qas.append({
            "doc_name": f"law/{stem}", "ID": f"q{d}",
            "questions": f"What fine does the penalty clause of {stem} page 1 impose?",
            "answers": f"{d}1 thousand dollars", "doc_type": "law",
            "answer_form": "String", "evidence_source": "text",
            "evidence_context": f"unique penalty clause of {stem} page 1 imposes "
                                f"a fine of {d}1 thousand dollars",
            "evidence_page_no": 1,
        })
    (data_dir / "qas_law.json").write_text(json.dumps(qas))


def test_bench_e2e_resume():
    print("bench e2e with fake embedder + resume + smoke-suffix isolation")
    tmp = Path(tempfile.mkdtemp(prefix="cebench_"))
    try:
        data, cache = tmp / "data", tmp / "cache"
        _write_synth_data(data)

        def run(extra):
            return subprocess.run(
                [sys.executable, str(REPO / "bench.py"), "--data-dir", str(data),
                 "--cache-dir", str(cache), "--embedders", "fake",
                 "--chunkers", "langchain,llamaindex"] + extra,
                capture_output=True, text=True, timeout=300,
            )

        r = run([])
        check("bench exit 0", r.returncode == 0, r.stderr[-800:])
        results = sorted((cache / "results").glob("*.json"))
        check("2 config results", len(results) == 2, str(results))
        res = json.loads(results[0].read_text())
        check("fake retrieval works (R@5 high)", res["recall@5"] >= 0.5, str(res))
        check("coverage recorded", 0 < res["coverage"] <= 1.0)
        check("ops fields present",
              all(k in res for k in ("chunk_time_s", "chunks_per_s", "index_mb")))

        # resume: delete ONE results file; rerun must only recompute that one
        results[0].unlink()
        emb_mtime = {p: p.stat().st_mtime_ns for p in (cache / "emb").glob("*.npz")}
        r2 = run([])
        check("resume exit 0", r2.returncode == 0, r2.stderr[-500:])
        check("deleted config recomputed", results[0].exists())
        check("embeddings reused, not recomputed",
              all(p.stat().st_mtime_ns == emb_mtime[p]
                  for p in (cache / "emb").glob("*.npz")))
        check("other config skipped", "results cached, skipping" in r2.stdout)

        # smoke suffix isolation
        r3 = run(["--limit-docs", "2"])
        check("smoke exit 0", r3.returncode == 0, r3.stderr[-500:])
        check("smoke files suffixed",
              (cache / "results" / "langchain__fake_n2.json").exists())
        check("full results untouched by smoke",
              len(list((cache / "results").glob("*.json"))) == 4)

        # aggregate on both slices
        for suffix, n_expected in (("", 2), ("_n2", 2)):
            r4 = subprocess.run(
                [sys.executable, str(REPO / "aggregate.py"), "--cache-dir", str(cache),
                 "--suffix", suffix],
                capture_output=True, text=True, timeout=120,
            )
            check(f"aggregate suffix='{suffix}' exit 0", r4.returncode == 0,
                  r4.stderr[-500:])
            check(f"matrix{suffix}.csv written",
                  (cache / f"matrix{suffix}.csv").exists())
        out = (cache / "matrix.md").read_text()
        check("interaction section present", "Interaction effects" in out)
        check("marginal tables written",
              (cache / "chunkers_table.md").exists()
              and (cache / "embedders_table.md").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_matching()
    test_chunkers()
    test_metrics_ranking()
    test_bench_e2e_resume()
    print(f"\nALL {PASS} CHECKS PASSED")
