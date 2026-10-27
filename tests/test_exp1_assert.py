"""Regression tests for the public Exp. 1 assertion."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp1_assert import get_assert


def test_truncated_json_does_not_use_confidence_as_prediction():
    output = '{"confidence": 5, "justification": "truncated'
    result = get_assert(output, {"vars": {"gold_number": "5"}})
    assert result["reason"].startswith("format_failure:")


def test_parsed_json_reads_sense_number_field():
    output = {"confidence": 5, "sense_number": 2}
    result = get_assert(output, {"vars": {"gold_number": "2"}})
    assert result["pass"] is True
