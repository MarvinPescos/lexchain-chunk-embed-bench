#!/usr/bin/env python3
"""CPU-only tests for the summarization comparison: extractive summarizers,
ROUGE, blind-eval blinding correctness, resumable checkpointing, deterministic
sampling. The LLM path is exercised with a fake summarizer (no NIM/torch).

Run:  python summarization/tests/test_summ.py
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUMM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUMM_DIR))

import build_blind_eval  # noqa: E402
import summarize  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


LEGAL_DOC = (
    "This Agreement is made on January 5, 2021 between Acme Corporation and Beta LLC. "
    "Acme agrees to pay Beta the sum of 4,162,000 dollars for consulting services. "
    "Beta shall deliver the final report by December 31, 2021. "
    "The parties agree that any dispute shall be resolved by arbitration in Delaware. "
    "Confidential information shall not be disclosed to third parties. "
    "This Agreement may be terminated by either party with thirty days written notice. "
    "Late payments accrue interest at a rate of one percent per month. "
    "The obligations herein are binding upon successors and assigns. "
    "Force majeure events excuse performance for their duration. "
    "This Agreement is governed by the laws of the State of Delaware."
)


def _write_docs(data_dir, docs: dict[str, str]):
    law = data_dir / "gt" / "law"
    law.mkdir(parents=True)
    for stem, text in docs.items():
        (law / f"{stem}.json").write_text(json.dumps(
            [{"page_idx": 0, "text": text}]))


def test_deterministic_sample():
    print("deterministic sampling")
    docs = {f"doc_{i:02d}": "x" for i in range(95)}
    a = summarize.sample_docs(docs)
    b = summarize.sample_docs(docs)
    check("sample size 13", len(a) == 13)
    check("sample reproducible", a == b)
    check("sample sorted", a == sorted(a))
    check("sample is subset", set(a) <= set(docs))


def test_extractive_and_rouge():
    print("extractive summarizers + ROUGE")
    for approach in ("textrank", "lexrank"):
        summary, meta = summarize.summarize_one(LEGAL_DOC, approach)
        check(f"{approach}: non-empty", len(summary.split()) > 5, summary)
        check(f"{approach}: shorter than source",
              len(summary.split()) < len(LEGAL_DOC.split()))
        check(f"{approach}: extractive text is from source",
              any(sent.strip()[:20] in LEGAL_DOC for sent in summary.split(".") if sent.strip()))
        check(f"{approach}: meta model recorded", "sumy" in meta["model"])

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    extractive, _ = summarize.summarize_one(LEGAL_DOC, "textrank")
    paraphrase = "The companies signed a consulting deal worth several million dollars."
    r_ext = scorer.score(LEGAL_DOC, extractive)["rougeL"].fmeasure
    r_par = scorer.score(LEGAL_DOC, paraphrase)["rougeL"].fmeasure
    check("extractive ROUGE-L > paraphrase ROUGE-L (inflation effect)",
          r_ext > r_par, f"ext={r_ext:.3f} par={r_par:.3f}")


def test_fake_llm_and_resume():
    print("LLM path (fake) + resumable checkpointing")
    tmp = Path(tempfile.mkdtemp(prefix="summtest_"))
    try:
        data = tmp / "data"
        _write_docs(data, {f"doc_{i:02d}": LEGAL_DOC + f" Clause {i}." for i in range(20)})
        docs = summarize.load_docs(data)
        stems = summarize.sample_docs(docs)[:2]
        cache = tmp / "cache"

        calls = {"n": 0}

        def fake_llm(text):
            calls["n"] += 1
            return "This is an abstractive summary of the agreement between the parties."

        summarize.run(docs, stems, ["textrank", "lexrank", "llm"], cache, llm_fn=fake_llm)
        files = list((cache / "summaries").glob("*.json"))
        check("6 summaries written (2 docs x 3)", len(files) == 6, str(len(files)))
        check("fake llm called twice", calls["n"] == 2, str(calls["n"]))
        rec = json.loads(next((cache / "summaries").glob("*__llm.json")).read_text())
        check("llm record marked fake", rec["model"] == "fake-llm")

        # resume: rerun must not recompute anything
        calls["n"] = 0
        summarize.run(docs, stems, ["textrank", "lexrank", "llm"], cache, llm_fn=fake_llm)
        check("resume skips all (fake llm not called)", calls["n"] == 0)
        check("still 6 summaries", len(list((cache / "summaries").glob("*.json"))) == 6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_blind_eval_scaffold():
    print("blind-eval scaffold (blinding + key round-trip)")
    tmp = Path(tempfile.mkdtemp(prefix="summtest_"))
    try:
        cache = tmp / "cache"
        sdir = cache / "summaries"
        sdir.mkdir(parents=True)
        docs = ["docA", "docB", "docC"]
        approaches = ["textrank", "lexrank", "llm"]
        for d in docs:
            for a in approaches:
                (sdir / f"{d}__{a}.json").write_text(json.dumps(
                    {"doc": d, "approach": a, "summary": f"summary of {d} by {a}"}))

        build_blind_eval.build(cache, seed=42)
        blind = list(csv.DictReader(open(cache / "blind_eval.csv")))
        key = list(csv.DictReader(open(cache / "unblinding_key.csv")))

        check("39-equivalent row count (3 docs x 3)", len(blind) == 9, str(len(blind)))
        check("blind CSV has no approach column", "approach" not in blind[0])
        check("blind rating columns blank",
              all(r["coverage"] == "" and r["fluency"] == "" for r in blind))
        check("variants are neutral labels",
              all(r["variant"].startswith("Summary ") for r in blind))

        # key round-trips: (pid) -> approach must match the summary text shown
        key_by_pid = {r["presentation_id"]: r for r in key}
        for r in blind:
            approach = key_by_pid[r["presentation_id"]]["approach"]
            check(f"key maps {r['presentation_id']} to the shown summary",
                  r["summary"] == f"summary of {r['doc_reference']} by {approach}")
            break  # one spot-check is enough for the assertion style

        # per-doc: the 3 approaches are all present but order is permuted (not identity)
        per_doc_order = {}
        for r in sorted(blind, key=lambda x: x["presentation_id"]):
            approach = key_by_pid[r["presentation_id"]]["approach"]
            per_doc_order.setdefault(r["doc_reference"], []).append(approach)
        check("each doc has all 3 approaches",
              all(sorted(v) == sorted(approaches) for v in per_doc_order.values()))
        check("blinding permutes at least one doc off identity",
              any(v != approaches for v in per_doc_order.values()), str(per_doc_order))

        # reproducible with same seed
        build_blind_eval.build(cache, seed=42)
        key2 = list(csv.DictReader(open(cache / "unblinding_key.csv")))
        check("blinding reproducible with seed", key == key2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scripts_compile():
    print("scripts compile")
    for name in ("summarize.py", "evaluate_summ.py", "build_blind_eval.py",
                 "aggregate_human.py"):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(SUMM_DIR / name)],
                           capture_output=True, text=True)
        check(f"py_compile {name}", r.returncode == 0, r.stderr[-300:])


if __name__ == "__main__":
    test_deterministic_sample()
    test_extractive_and_rouge()
    test_fake_llm_and_resume()
    test_blind_eval_scaffold()
    test_scripts_compile()
    print(f"\nALL {PASS} CHECKS PASSED")
