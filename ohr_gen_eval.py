"""OHR-Bench generation scoring, vendored verbatim so our numbers match theirs.

Source (Apache-2.0): https://github.com/opendatalab/OHR-Bench
  - src/metric/common.py         -> normalize_answer, f1_score, exact_match_score, em
  - src/tasks/quest_answer.py    -> QA prompt usage + <response> extraction
  - src/prompts/QA_prompt.txt    -> QA_PROMPT

Metrics (see README / plan for the full explanation):
  f1_score          token-overlap F1 after normalization; OHR-Bench's headline
                    generation metric. yes/no/noanswer answers must match exactly
                    or score 0.
  exact_match_score strict normalized equality -> reported as accuracy (EM).
  em_contains       normalized(gold) is a substring of normalized(pred); a more
                    forgiving accuracy for extractive short answers (their `em`).

We deliberately skip their BLEU/ROUGE/BERTScore (need `evaluate` + HF cache and
aren't requested). Chinese (jieba) path is irrelevant for the English Law set;
we guard it so the dependency is never imported.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

# OHR-Bench's QA_prompt.txt, verbatim.
QA_PROMPT = (
    "You are an expert, you have been provided with a question and documents "
    "retrieved based on that question. Your task is to search the content and "
    "answer these questions using both the retrieved information. \n\n"
    "You **MUST** answer the questions briefly with one or two words or very "
    "short sentences, devoid of additional elaborations.\n\n"
    "Write the answers within <response></response>. If you cannot find answer "
    'from retrieved Documents, say: "Not answerable".'
)


def normalize_answer(s: str) -> str:
    """Vendored: lowercase, strip punctuation, remove articles, fix whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _has_chn_character(s: str) -> bool:
    for char in s:
        try:
            if "CJK" in unicodedata.name(char):
                return True
        except ValueError:
            continue
    return False


def f1_score(prediction: str, ground_truth: str) -> float:
    """Vendored OHR-Bench word-overlap F1 (the headline generation metric)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0.0

    if (normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    if _has_chn_character(prediction) or _has_chn_character(ground_truth):
        import jieba  # lazy: never needed for the English Law set

        prediction_tokens = jieba.lcut(normalized_prediction)
        ground_truth_tokens = jieba.lcut(normalized_ground_truth)
    else:
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Vendored: strict normalized equality. Reported as accuracy (EM)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def em_contains(prediction: str, ground_truth: str) -> float:
    """Vendored `em`: normalized(gold) is contained in normalized(pred).

    A more forgiving accuracy for extractive short-answer QA.
    """
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


_RESPONSE_FALLBACK = re.compile(r"</?response>(.*?)</?response>", re.DOTALL)


def extract_response(raw: str) -> str:
    """OHR-Bench's answer extraction from the model's raw text."""
    real = raw.split("<response>")[-1].split("</response>")[0]
    if real.strip() == "":
        m = _RESPONSE_FALLBACK.search(raw)
        if m:
            real = m.group(1).strip()
    return real.strip()
