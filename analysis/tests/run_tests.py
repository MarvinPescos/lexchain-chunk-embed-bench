#!/usr/bin/env python3
"""CPU-only tests for the analysis comparison (fake LLM, no network, no key).

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

from analysis.prompt import build_messages, parse_analysis, normalize_analysis  # noqa: E402
from analysis.matching_entities import score_category  # noqa: E402
from analysis.analyze import run_analyses  # noqa: E402
from analysis import build_blind_eval, aggregate_analysis  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def test_prompt_parsing():
    print("prompt parsing / schema normalization")
    good = '{"summary":"S","entities":{"parties":["A","B"],"dates":["2020-06-02"],' \
           '"monetary_amounts":["$100"],"obligations":["A must pay"]},' \
           '"risk_flags":[{"risk":"auto-renewal","severity":"high"}]}'
    p = parse_analysis(good)
    check("parses clean json", p is not None)
    check("summary kept", p["summary"] == "S")
    check("entities parsed", p["entities"]["parties"] == ["A", "B"])
    check("risk parsed", p["risk_flags"][0]["risk"] == "auto-renewal")

    fenced = "```json\n" + good + "\n```"
    check("parses fenced json", parse_analysis(fenced) is not None)
    noisy = "Sure! Here it is:\n" + good + "\nHope that helps."
    check("parses json with surrounding prose", parse_analysis(noisy) is not None)
    check("unparseable -> None", parse_analysis("not json at all") is None)

    # normalization fills missing keys / coerces string->list
    n = normalize_analysis({"summary": "x", "entities": {"parties": "solo"}})
    check("missing keys filled", n["entities"]["obligations"] == [] and n["risk_flags"] == [])
    check("scalar entity coerced to list", n["entities"]["parties"] == ["solo"])


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
    s = score_category("parties", [], ["Acme"])
    check("all missed -> recall 0", s["tp"] == 0 and s["fn"] == 1 and s["f1"] == 0.0)


def _fake_call_fn(model_id, messages):
    """Deterministic per-model output; encodes model identity so blind/F1 tests can check."""
    tag = model_id.split("/")[-1]
    obj = {
        "summary": f"Summary from {tag}. Acme Corp and Beta LLC entered an agreement.",
        "entities": {
            "parties": ["Acme Corp", "Beta LLC"],
            "dates": ["2020-06-02"],
            "monetary_amounts": ["$100"],
            "obligations": ["Acme must pay Beta"],
        },
        "risk_flags": [{"risk": "auto-renewal clause", "severity": "high"}],
    }
    return json.dumps(obj), {"completion_tokens": 42}, 0.01


def test_runner_resume():
    print("runner: checkpoint + resume")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_a": "text a", "doc_b": "text b"}
        models = {"llama-3.1-70b": "meta/llama-3.1-70b-instruct",
                  "qwen2.5-72b": "qwen/qwen2.5-72b-instruct"}
        run_analyses(docs, models, tmp, _fake_call_fn)
        files = list((tmp / "analyses").glob("*.json"))
        check("4 checkpoints written", len(files) == 4, str(len(files)))
        rec = json.loads((tmp / "analyses" / "doc_a__llama-3.1-70b.json").read_text())
        check("checkpoint parsed ok", rec["ok"] and rec["parsed"]["entities"]["parties"])

        calls = []
        def counting_fn(mid, msgs):
            calls.append(mid)
            return _fake_call_fn(mid, msgs)
        run_analyses(docs, models, tmp, counting_fn)  # resume -> no new calls
        check("resume skips completed", calls == [], f"made {len(calls)} calls")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_blind_and_aggregate(monkeypatch_docs=True):
    print("blind sheet build + un-blind aggregate + F1")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        # 2 docs x 3 models fake analyses
        docs = {"doc_alpha": "t", "doc_beta": "t"}
        models = {"llama-3.1-70b": "meta/llama-3.1-70b-instruct",
                  "qwen2.5-72b": "qwen/qwen2.5-72b-instruct",
                  "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1"}
        run_analyses(docs, models, tmp, _fake_call_fn)

        # build_blind_eval / load_docs reads real GT dir; stub it to our docs
        import analysis.build_blind_eval as bbe
        orig = bbe.load_docs
        bbe.load_docs = lambda *a, **k: {d: "source text " + d for d in docs}
        try:
            n_blind, n_key = bbe.build(tmp)
        finally:
            bbe.load_docs = orig
        check("30-style blind rows (2x3=6)", n_blind == 6, str(n_blind))

        blind = list(csv.DictReader(open(tmp / "blind_eval.csv")))
        key = {r["presentation_id"]: r for r in csv.DictReader(open(tmp / "unblinding_key.csv"))}
        check("blind rows hide model", all("model" not in r for r in blind))
        check("labels are neutral", all(r["output_label"].startswith("Output") for r in blind))
        check("key inverts every row", all(r["presentation_id"] in key for r in blind))
        check("key covers all 6", len(key) == 6)
        # each doc's 3 labels map to the 3 distinct models
        per_doc = {}
        for r in key.values():
            per_doc.setdefault(r["doc_reference"], set()).add(r["model"])
        check("each doc has all 3 models", all(len(v) == 3 for v in per_doc.values()))

        gt_template = list(csv.DictReader(open(tmp / "ground_truth_key_template.csv")))
        check("gt template one row per doc", len(gt_template) == 2, str(len(gt_template)))

        # simulate the human filling ratings (give llama higher scores) + gt key
        model_of = {r["presentation_id"]: r["model"] for r in key.values()}
        for r in blind:
            m = model_of[r["presentation_id"]]
            base = 5 if m == "llama-3.1-70b" else 3
            r["summary_coverage_1to5"] = base
            r["summary_accuracy_1to5"] = base
            r["summary_fluency_1to5"] = base
        cols = blind[0].keys()
        with open(tmp / "blind_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(blind)

        with open(tmp / "ground_truth_key.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["doc_reference"] + aggregate_analysis.GT_LIST_COLS)
            w.writeheader()
            for d in docs:
                w.writerow({"doc_reference": d, "parties": "Acme Corp; Beta LLC",
                            "dates": "2020-06-02", "monetary_amounts": "$100",
                            "obligations": "Acme must pay Beta",
                            "risk_clauses": "auto-renewal clause"})

        rows, human_filled, gt_filled, n_bl, n_gt = aggregate_analysis.aggregate(tmp)
        check("all rows rated", human_filled == 6, str(human_filled))
        check("gt filled for 2 docs", gt_filled == 2, str(gt_filled))
        by_model = {r["model"]: r for r in rows}
        check("llama higher coverage (un-blinded)",
              by_model["llama-3.1-70b"]["coverage"] == 5.0
              and by_model["qwen2.5-72b"]["coverage"] == 3.0, str(by_model))
        check("entity F1 = 1.0 (fake matches gt)",
              by_model["llama-3.1-70b"]["entity_f1"] == 1.0, str(by_model["llama-3.1-70b"]))
        check("risk F1 = 1.0", by_model["mixtral-8x22b"]["risk_f1"] == 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_prompt_parsing()
    test_matching()
    test_runner_resume()
    test_blind_and_aggregate()
    print(f"\nALL {PASS} CHECKS PASSED")
