"""
Experiment 2 — sense-assignment assertion (promptfoo `type: python`).

The Exp 2 workflow ran annotation FIRST (Argilla → curated gold), so gold IS
known at run time: eval_set.csv supplies a gold_sense_id per record and
exp2_tests.py → exp2_transform.py resolves it to ``gold_number`` in the test
vars. The assertion therefore scores correctness the Exp-1 way. Its job:

  1. **Parse the predicted sense number** from the model's output, using the
     same three-way detection as exp1_assert.py (JSON-style sense_number
     field, "SENSE: N" line, bare integer).

  2. **Surface the prediction in the assertion `reason`** (``predicted: <N>``
     / ``correct (predicted: N)`` / ``wrong: expected G, got N``) so it can be
     regex-extracted from the promptfoo SQLite store without re-parsing every
     raw output.

  3. **Score against gold** (``context['vars']['gold_number']``): pass when
     predicted == gold, fail with ``wrong:`` otherwise; a format failure (no
     parseable number) is a ``format_failure:`` fail.

A pass-through fallback is preserved for the rare row that carries no
``gold_number`` (empty / unresolvable gold sid): it passes with
``predicted: N`` in the reason and is scored downstream. The authoritative
IAA / accuracy / novelty analysis lives in analysis/exp2_iaa.ipynb.
"""
import re

# Same regex set as exp1_assert.py — kept inlined (not imported) so a future
# change to Exp 1's parser does not silently shift Exp 2's parsing behaviour.
# Order matters: more specific patterns first.
_SENSE_LINE_RE = re.compile(r'(?:^|\n)\s*SENSE\s*:\s*"?(\d+)"?', re.IGNORECASE)
_JSON_RE = re.compile(r'"sense_number"\s*:\s*"?(\d+)"?', re.IGNORECASE)
_INT_RE = re.compile(r"\b\d+\b")
_STRIP_CHARS = "\"'`*. \t\r\n"


def _parse_predicted(raw: str) -> str | None:
    """Return the predicted sense number as a string, or None if unparseable."""
    # 1. justified-plain two-line: "SENSE: N" line at start of line.
    m = _SENSE_LINE_RE.search(raw)
    if m:
        return m.group(1)
    # 2. JSON output (if a v2 ever adds it back): "sense_number" field.
    m = _JSON_RE.search(raw)
    if m:
        return m.group(1)
    # 3. Plain output that IS just a number (the expected zeroshot shape).
    cleaned = "".join(ch for ch in raw if ch not in _STRIP_CHARS)
    if cleaned.isdigit():
        return cleaned
    # 4. Loose: first standalone integer anywhere in prose.
    loose = _INT_RE.search(raw)
    if loose:
        return loose.group(0)
    return None


def get_assert(output, context):
    raw = str(output or "").strip()
    gold_raw = (context or {}).get("vars", {}).get("gold_number")
    gold = str(gold_raw).strip() if gold_raw not in (None, "") else None

    predicted = _parse_predicted(raw)

    if predicted is None:
        return {
            "pass": False,
            "score": 0,
            "reason": f"format_failure: no sense number in output | {raw[:120]}",
        }

    # Run-time path: no gold. Pass-through with the predicted number in the
    # reason so the analysis notebook can extract it from grading_result.
    if gold is None:
        return {
            "pass": True,
            "score": 1,
            "reason": f"predicted: {predicted}",
        }

    # Replay-time path: gold present (post-annotation re-scoring or dev
    # fixture). Score correctness Exp-1 style.
    if predicted == gold:
        return {"pass": True, "score": 1, "reason": f"correct (predicted: {predicted})"}
    return {
        "pass": False,
        "score": 0,
        "reason": f"wrong: expected {gold}, got {predicted}",
    }
