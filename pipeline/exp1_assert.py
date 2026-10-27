"""
Experiment 1 — sense-assignment assertion (promptfoo `type: python`).

Senses are presented to the model as a NUMBERED list (1..N); the model answers
with a number (or "0" = no sense fits). This compares the predicted number to
``context['vars']['gold_number']``.

Three-way outcome (PLAN.md §4.1): correct / wrong / format_failure. A model that
refuses or emits prose is a different failure mode than one that picks the wrong
sense — they must not be pooled in accuracy. promptfoo only records pass/fail,
so a format failure is returned as a fail with a ``format_failure:`` reason
prefix; the analysis notebook separates the three categories on the prefix.

Handles three output formats:
  - JSON templates           -> {"justification": "...", ..., "sense_number": 3}
                                (field order is rationale-first per
                                 PROMPT_RECOMMENDATIONS.md §1.2; the parser is
                                 order-independent — it looks up the
                                 ``sense_number`` key by name)
  - justified-plain template -> two lines:
                                  "JUSTIFICATION: ...\\nSENSE: 3"
                                (output_justified_plain block; introduced to
                                 replace a broken task_justified + output_plain
                                 pairing — see PROMPT_RECOMMENDATIONS.md §1.1)
  - plain templates          -> just the number, e.g. 3   (or "0")

promptfoo calls ``get_assert(output, context)`` and accepts a
{pass, score, reason} dict (a GradingResult).
"""
import re

# Order matters: more specific patterns are tried first so that an output that
# matches several is attributed to its true template.
_SENSE_LINE_RE = re.compile(r'(?:^|\n)\s*SENSE\s*:\s*"?(\d+)"?', re.IGNORECASE)
_JSON_RE = re.compile(r'"sense_number"\s*:\s*"?(\d+)"?', re.IGNORECASE)
_INT_RE = re.compile(r"\b\d+\b")
_STRIP_CHARS = "\"'`*. \t\r\n"


def get_assert(output, context):
    # Promptfoo may pass a parsed object for structured providers.  Preserve
    # the field semantics before falling back to text parsing.
    output_dict = output if isinstance(output, dict) else None
    raw = str(output or "").strip()
    gold = str((context or {}).get("vars", {}).get("gold_number", "") or "").strip()

    predicted = None
    # 1. justified-plain two-line format: a "SENSE: N" line at start of line.
    m = _SENSE_LINE_RE.search(raw)
    if m:
        predicted = m.group(1)
    else:
        # 2. JSON output: structured responses must contain an intact
        # sense_number field.  Never use the prose/first-integer fallback for
        # JSON-shaped output: confidence and justification commonly contain
        # integers and are not predictions.
        if output_dict is not None and "sense_number" in output_dict:
            predicted = str(output_dict["sense_number"])
        else:
            m = _JSON_RE.search(raw)
            json_shaped = raw.lstrip().startswith(("{", "[", "```json", "```JSON"))
            if m:
                predicted = m.group(1)
            elif json_shaped:
                predicted = None
            else:
                cleaned = "".join(ch for ch in raw if ch not in _STRIP_CHARS)
                if cleaned.isdigit():
                    # 3. Plain output that IS just a number (the expected format).
                    predicted = cleaned
                else:
                    # 4. Loose: the first standalone integer anywhere in prose.
                    loose = _INT_RE.search(raw)
                    if loose:
                        predicted = loose.group(0)

    if predicted is None:
        return {
            "pass": False,
            "score": 0,
            "reason": f"format_failure: no sense number in output | {raw[:120]}",
        }

    if predicted == gold:
        return {"pass": True, "score": 1, "reason": "correct"}
    return {"pass": False, "score": 0, "reason": f"wrong: expected {gold}, got {predicted}"}
