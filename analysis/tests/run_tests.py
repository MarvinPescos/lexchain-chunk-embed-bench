#!/usr/bin/env python3
"""CPU-only tests for the 4+1 model analysis benchmark (fake LLM, no network).

Run:  python analysis/tests/run_tests.py
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from analysis.prompt import (  # noqa: E402
    RISK_CATEGORY_NAMES,
    build_messages,
    normalize_analysis,
    parse_analysis,
    strip_thinking,
    validate_schema,
)
from analysis.data import MODELS, REFERENCE_ROLE, context_check  # noqa: E402
from analysis.matching_entities import score_category  # noqa: E402
from analysis.analyze import run_analyses  # noqa: E402
from analysis import aggregate_analysis, build_blind_eval, prepare_ground_truth  # noqa: E402

PASS = 0
ALL_MODELS = list(MODELS)  # 4 candidates + 1 reference
CANDIDATES = [m for m in ALL_MODELS if MODELS[m]["role"] == "candidate"]


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def _risk_flags(present_map=None):
    present_map = present_map or {}
    return [
        {"category": c, "present": c in present_map,
         "quote": present_map.get(c, "")}
        for c in RISK_CATEGORY_NAMES
    ]


def _good_obj(present_map=None):
    return {
        "summary": "The parties entered an agreement with obligations and fees.",
        "entities": {
            "parties": ["Acme Corp", "Beta LLC"],
            "dates": ["2020-06-02"],
            "monetary_amounts": ["$100"],
            "obligations": ["Acme must pay Beta"],
        },
        "risk_flags": _risk_flags(present_map),
    }


def test_prompt_parsing():
    print("prompt: checklist schema, parsing, think-stripping")
    msgs = build_messages("DOC")
    check("prompt contains all 12 categories",
          all(c in msgs[1]["content"] for c in RISK_CATEGORY_NAMES))

    good = json.dumps(_good_obj({"auto_renewal": "renews automatically each year"}))
    p = parse_analysis(good)
    check("parses clean json", p is not None)
    check("12 checklist entries in order",
          [f["category"] for f in p["risk_flags"]] == RISK_CATEGORY_NAMES)
    ar = next(f for f in p["risk_flags"] if f["category"] == "auto_renewal")
    check("present category keeps quote", ar["present"] and "renews" in ar["quote"])
    check("schema valid", validate_schema(p) == [])

    think = "<think>Let me reason...\nblah</think>\n" + good
    check("qwen think block stripped", parse_analysis(think) is not None)
    check("strip_thinking removes block", "<think>" not in strip_thinking(think))
    check("unterminated think dropped",
          strip_thinking("prefix <think> endless reasoning") == "prefix")

    fenced = "```json\n" + good + "\n```"
    check("parses fenced json", parse_analysis(fenced) is not None)
    check("unparseable -> None", parse_analysis("not json") is None)

    # normalization: missing/misordered categories get realigned; extras dropped
    partial = _good_obj()
    partial["risk_flags"] = [{"category": "governing_law", "present": True,
                              "quote": "laws of Virginia"}]
    n = normalize_analysis(partial)
    check("missing categories filled as absent",
          len(n["risk_flags"]) == 12 and
          sum(f["present"] for f in n["risk_flags"]) == 1)
    bad = _good_obj({"indemnification": ""})  # present without quote
    bad["risk_flags"][0]["quote"] = ""
    bad["risk_flags"][0]["present"] = True
    problems = validate_schema(normalize_analysis(bad))
    check("present-without-quote flagged", any("without a quote" in p for p in problems))


def test_matching():
    print("entity/risk matching + F1")
    s = score_category("parties", ["Acme Corp", "Beta LLC"], ["ACME CORP.", "Beta, LLC"])
    check("fuzzy party match tp=2", s["tp"] == 2 and s["f1"] == 1.0, str(s))
    s = score_category("parties", ["Acme Corp", "Ghost Inc"], ["Acme Corporation"])
    check("party precision/recall", s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0, str(s))
    s = score_category("dates", ["June 2, 2020"], ["2020-06-02"])
    check("date format-invariant match", s["tp"] == 1, str(s))
    s = score_category("monetary_amounts", ["$4,162,000.00 grant"], ["4162000 dollars"])
    check("money numeric match", s["tp"] == 1, str(s))


def test_context_check():
    print("context safety (fail loudly, never truncate)")
    small = {"doc": "hello " * 100}
    report = context_check(small, ["llama3.1-8b"])
    check("small doc fits", report["per_model"]["llama3.1-8b"]["limit"] == 32768)
    huge = {"doc": "word " * 40000}  # ~40k tokens > every num_ctx
    try:
        context_check(huge, ["llama3.1-8b"])
        check("oversized doc rejected", False)
    except SystemExit as e:
        check("oversized doc rejected", "CONTEXT CHECK FAILED" in str(e))
    # phi4's 16k window: a ~18k-token doc must be rejected for phi4 specifically
    doc18k = {"doc": "word " * 18000}
    try:
        context_check(doc18k, ["phi4-14b"])
        check("phi4 16k limit enforced", False)
    except SystemExit as e:
        check("phi4 16k limit enforced", "phi4-14b" in str(e))


def test_prepare_ground_truth():
    print("prepare_ground_truth: blank template + readable dumps")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "PARTY ONE and PARTY TWO agree. " * 20,
                "doc_beta": "Effective 2021-01-01. Fee is $500."}
        tmpl = prepare_ground_truth.write_ground_truth_template(tmp, docs)
        rows = list(csv.DictReader(open(tmpl)))
        check("template one blank row per doc", len(rows) == 2, str(len(rows)))
        check("template cols == aggregator's GT_LIST_COLS",
              prepare_ground_truth.GT_KEY_COLS == aggregate_analysis.GT_LIST_COLS)
        check("template gt cells are blank",
              all(r[c] == "" for r in rows for c in prepare_ground_truth.GT_KEY_COLS))
        dumps = prepare_ground_truth.write_doc_dumps(tmp, docs)
        check("one dump per doc + index", len(dumps) == 2
              and (tmp / "doc_texts" / "INDEX.txt").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# fake deterministic LLM: model-specific quality so aggregation is testable.
# llama3.1-8b gets everything right; others miss the auto_renewal risk.
def _fake_call_fn(short, messages):
    if short == "llama3.1-8b":
        obj = _good_obj({"auto_renewal": "renews automatically each year",
                         "governing_law": "laws of Virginia"})
    else:
        obj = _good_obj({"governing_law": "laws of Virginia"})
    raw = json.dumps(obj)
    if short == "qwen3-14b":  # simulate thinking-mode leakage
        raw = "<think>step by step...</think>" + raw
    return raw, {"completion_tokens": 42}, 0.5 if short != "llama-3.1-70b" else 2.0


def test_runner_resume():
    print("runner: checkpoint + resume + metadata")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_a": "text a", "doc_b": "text b"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)
        files = list((tmp / "analyses").glob("*.json"))
        check("10 checkpoints (2 docs x 5 models)", len(files) == 10, str(len(files)))
        rec = json.loads((tmp / "analyses" / "doc_a__qwen3-14b.json").read_text())
        check("think-leaked output parsed ok", rec["ok"], str(rec.get("schema_problems")))
        check("checkpoint has prompt_version+role+latency label",
              rec["prompt_version"] and rec["role"] == "candidate"
              and rec["latency_label"] == "colab_gpu")
        ref = json.loads((tmp / "analyses" / "doc_a__llama-3.1-70b.json").read_text())
        check("reference role recorded", ref["role"] == REFERENCE_ROLE)

        calls = []
        def counting_fn(short, msgs):
            calls.append(short)
            return _fake_call_fn(short, msgs)
        run_analyses(docs, ALL_MODELS, tmp, counting_fn)
        check("resume skips completed", calls == [], f"made {len(calls)} calls")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_gt_key(tmp, docs):
    with open(tmp / "ground_truth_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_reference"] + aggregate_analysis.GT_LIST_COLS)
        w.writeheader()
        for d in docs:
            w.writerow({"doc_reference": d, "parties": "Acme Corp; Beta LLC",
                        "dates": "2020-06-02", "monetary_amounts": "$100",
                        "obligations": "Acme must pay Beta",
                        "risk_clauses": "auto renewal each year; governing law Virginia"})


def test_blind_gate_and_aggregate():
    print("GT-first guardrail + blind sheet (A-E) + aggregate + wins")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "t", "doc_beta": "t"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)

        # guardrail: refuses before the key exists
        try:
            build_blind_eval.build(tmp)
            check("guardrail blocks without GT key", False)
        except SystemExit as e:
            check("guardrail blocks without GT key", "GUARDRAIL" in str(e))
        _write_gt_key(tmp, docs)
        n_blind, _ = build_blind_eval.build(tmp)  # passes now
        check("blind rows = docs x 5 models", n_blind == 10, str(n_blind))

        blind = list(csv.DictReader(open(tmp / "blind_eval.csv")))
        key = {r["presentation_id"]: r for r in csv.DictReader(open(tmp / "unblinding_key.csv"))}
        check("labels span A-E",
              {r["output_label"] for r in blind} == {f"Output {c}" for c in "ABCDE"})
        check("blind rows hide model+role",
              all("model" not in r and "role" not in r for r in blind))
        check("SummEval columns present",
              all(c in blind[0] for c in ("coherence_1to5", "consistency_1to5",
                                          "fluency_1to5", "relevance_1to5",
                                          "hallucinations_notes")))
        check("key carries reference role",
              sum(r["role"] == REFERENCE_ROLE for r in key.values()) == 2)

        # fill ratings: give llama3.1-8b top scores, others lower
        model_of = {r["presentation_id"]: r["model"] for r in key.values()}
        for r in blind:
            base = 5 if model_of[r["presentation_id"]] == "llama3.1-8b" else 3
            for c in ("coherence_1to5", "consistency_1to5", "fluency_1to5", "relevance_1to5"):
                r[c] = base
        with open(tmp / "blind_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=blind[0].keys())
            w.writeheader(); w.writerows(blind)

        rows, wins, human_filled, gt_filled, n_bl, _ = aggregate_analysis.aggregate(tmp)
        check("all rows rated", human_filled == 10, str(human_filled))
        by_model = {r["model"]: r for r in rows}
        check("SummEval means un-blinded",
              by_model["llama3.1-8b"]["coherence"] == 5.0
              and by_model["phi4-14b"]["coherence"] == 3.0)
        check("risk F1 rewards the complete model",
              by_model["llama3.1-8b"]["risk_f1"] == 1.0
              and by_model["phi4-14b"]["risk_f1"] < 1.0,
              str({m: by_model[m]["risk_f1"] for m in by_model}))
        check("entity F1 = 1.0 all (fake entities match key)",
              all(by_model[m]["entity_f1"] == 1.0 for m in by_model))
        check("latency recorded", by_model["llama-3.1-70b"]["mean_latency_s"] == 2.0)
        check("wins exclude reference", "llama-3.1-70b" not in wins)
        check("llama3.1-8b wins risk_f1 on both docs",
              wins["llama3.1-8b"]["risk_f1"] == 2, str(wins["llama3.1-8b"]))
        check("ties count for all (entity_f1)",
              all(wins[m]["entity_f1"] == 2 for m in CANDIDATES), str(wins))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_prompt_parsing()
    test_matching()
    test_context_check()
    test_prepare_ground_truth()
    test_runner_resume()
    test_blind_gate_and_aggregate()
    print(f"\nALL {PASS} CHECKS PASSED")
