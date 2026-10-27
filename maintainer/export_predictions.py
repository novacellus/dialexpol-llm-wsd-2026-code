#!/usr/bin/env python3
"""Export safe, parsed prediction tables from the private promptfoo store.

The exporter is intentionally streaming and read-only. It never exports raw
responses, rationales, prompt text, provider configuration, costs, latency, or
token usage. It uses the same outcome and number parsing helpers as the public
analysis loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai" / "analysis"))
from _eval_loader import classify_exp2_template, classify_outcome, classify_template, parse_predicted_number  # noqa: E402
sys.path.insert(0, str(ROOT / "ai" / "scripts"))
from transform import numbered_senses  # noqa: E402
from exp2_transform import _resolve_inv_lemma  # noqa: E402
from exp1_assert import get_assert as exp1_assert  # noqa: E402
from exp2_assert import get_assert as exp2_assert  # noqa: E402


FINAL_RUNS = {
    "phase_a_plain_restricted": "exp1",
    "phase_a_json_restricted": "exp1",
    "phase_b_plain": "exp1",
    "exp2_core": "exp2",
    "exp2_extended": "exp2",
}

EXP1_FIELDS = [
    "run", "eval_id", "provider", "template", "task_type", "granularity",
    "def_language", "show_metadata", "lemma", "quote_idx", "gold_number",
    "gold_sid", "predicted_number", "predicted_sid", "response_status", "outcome_stored", "outcome_corrected",
]
EXP2_FIELDS = [
    "run", "eval_id", "provider", "template", "task_type", "def_language",
    "context_length", "show_metadata", "key", "record_id", "siglum",
    "gold_number", "gold_sense_id", "predicted_number", "predicted_sid", "response_status", "outcome_stored", "outcome_corrected",
]


def db_uri(path: Path) -> str:
    return f"file:{path}?mode=ro&immutable=1"


def open_db(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_uri(path), uri=True)


def final_eval_ids(registry_path: Path) -> dict[str, tuple[str, str]]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    result: dict[str, tuple[str, str]] = {}
    for run, kind in FINAL_RUNS.items():
        record = registry.get(run, {})
        eval_id = record.get("id")
        if not eval_id:
            raise SystemExit(f"final run has no eval id: {run}")
        result[eval_id] = (run, kind)
    return result


def query_for(kind: str) -> str:
    if kind == "exp1":
        return """
        SELECT json_extract(prompt,'$.label'), json_extract(prompt,'$.raw'), json_extract(provider,'$.label'),
               json_extract(test_case,'$.vars.lemma'), json_extract(test_case,'$.vars.granularity'),
               json_extract(test_case,'$.vars.def_language'), json_extract(test_case,'$.vars.show_metadata'),
               json_extract(test_case,'$.vars.gold_number'), json_extract(test_case,'$.vars.gold'),
               json_extract(test_case,'$.metadata.quote_idx'), success,
               json_extract(grading_result,'$.reason'), json_extract(response,'$.output')
        FROM eval_results WHERE eval_id=?
        """
    return """
    SELECT json_extract(prompt,'$.label'), json_extract(prompt,'$.raw'), json_extract(provider,'$.label'),
           json_extract(test_case,'$.metadata.key'), json_extract(test_case,'$.metadata.record_id'),
           json_extract(test_case,'$.metadata.siglum'), json_extract(test_case,'$.vars.def_language'),
           json_extract(test_case,'$.vars.context_length'), json_extract(test_case,'$.vars.show_metadata'),
           json_extract(test_case,'$.vars.gold_number'), json_extract(test_case,'$.metadata.gold_sense_id'),
           success, json_extract(grading_result,'$.reason'), json_extract(response,'$.output')
    FROM eval_results WHERE eval_id=?
    """


def export_run(con: sqlite3.Connection, eval_id: str, run: str, kind: str, writer: csv.DictWriter) -> int:
    seen: set[tuple[str, ...]] = set()
    count = 0
    for row in con.execute(query_for(kind), (eval_id,)):
        if kind == "exp1":
            prompt, prompt_raw, provider, lemma, granularity, language, metadata, gold_number, gold_sid, quote_idx, success, reason, output = row
            assert_no_prompt_echo(prompt_raw, output, run)
            outcome_stored = classify_outcome(bool(success), reason)
            outcome_corrected = outcome_stored if run != "phase_a_json_restricted" or output is None else corrected_outcome(output, gold_number, exp1_assert)
            if metadata is None:
                raise SystemExit(f"{run}: missing show_metadata for quote {quote_idx}")
            seed = normalise_quote_seed(quote_idx)
            template = classify_template(prompt)
            predicted_number = parse_release_number(output, template)
            predicted_sid = exp1_sid(lemma, granularity, language, seed, predicted_number)
            ordered = numbered_senses(str(lemma).upper(), granularity, language, sense_order_seed=seed)
            number_to_sid = {n: sid for n, sid, _ in ordered}
            gold_number_expected = next((n for n, sid, _ in ordered if sid == gold_sid), None)
            if gold_number_expected is not None and str(gold_number) != str(gold_number_expected):
                raise SystemExit(f"{run}: gold-number mismatch for {lemma}/{quote_idx}")
            if outcome_corrected == "correct" and predicted_sid != gold_sid:
                raise SystemExit(f"{run}: correct row decoded to wrong sid for {lemma}/{quote_idx}")
            record = {
                "run": run, "eval_id": eval_id, "provider": provider,
                "template": template, "task_type": template.removesuffix("_json"),
                "granularity": granularity, "def_language": language,
                "show_metadata": bool(metadata), "lemma": lemma, "quote_idx": quote_idx,
                "gold_number": gold_number, "gold_sid": gold_sid,
                "predicted_number": predicted_number,
                "predicted_sid": predicted_sid,
                "response_status": "responded" if output is not None else "no_response",
                "outcome_stored": outcome_stored, "outcome_corrected": outcome_corrected,
            }
            identity = (str(provider), str(record["template"]), str(granularity), str(language), str(metadata), str(quote_idx))
        else:
            prompt, prompt_raw, provider, key, record_id, siglum, language, context_length, metadata, gold_number, gold_sid, success, reason, output = row
            assert_no_prompt_echo(prompt_raw, output, run)
            if metadata is None:
                raise SystemExit(f"{run}: missing show_metadata for {key}")
            lemma = str(key).split("::", 1)[0]
            outcome_stored = classify_outcome(bool(success), reason)
            outcome_corrected = outcome_stored
            template = classify_exp2_template(prompt)
            predicted_sid = exp2_sid(key, record_id, lemma, gold_sid, language, output)
            ordered = numbered_senses(_resolve_inv_lemma(str(lemma).upper(), gold_sid), "l1", language, sense_order_seed=record_id)
            number_to_sid = {n: sid for n, sid, _ in ordered}
            number_to_sid[0] = "0"
            gold_number_expected = next((n for n, sid, _ in ordered if sid == gold_sid), 0 if str(gold_sid) == "0" else None)
            if gold_number_expected is not None and str(gold_number) != str(gold_number_expected):
                raise SystemExit(f"{run}: gold-number mismatch for {key}")
            if outcome_corrected == "correct" and predicted_sid != ("0" if str(gold_sid) == "0" else gold_sid):
                raise SystemExit(f"{run}: correct row decoded to wrong sid for {key}")
            record = {
                "run": run, "eval_id": eval_id, "provider": provider,
                "template": template, "task_type": template,
                "def_language": language, "context_length": context_length,
                "show_metadata": bool(metadata), "key": key, "record_id": record_id,
                "siglum": siglum, "gold_number": gold_number, "gold_sense_id": gold_sid,
                "predicted_number": parse_predicted_number(output),
                "predicted_sid": predicted_sid,
                "response_status": "responded" if output is not None else "no_response",
                "outcome_stored": outcome_stored, "outcome_corrected": outcome_corrected,
            }
            identity = (str(provider), str(template), str(language), str(context_length), str(metadata), str(key))
        if identity in seen:
            raise SystemExit(f"duplicate prediction identity in {run}: {identity}")
        seen.add(identity)
        writer.writerow(record)
        count += 1
    print(f"{run}: exported {count} rows", file=sys.stderr)
    return count


def normalise_quote_seed(value) -> str:
    """Match the CSV DictReader convention used by exp1_tests.py."""
    text = str(value)
    if text.isdigit():
        return text + ".0"
    return text


def corrected_outcome(output, gold_number, assertion) -> str:
    result = assertion(output, {"vars": {"gold_number": "" if gold_number is None else str(gold_number)}})
    reason = result.get("reason", "")
    if result.get("pass") is True:
        return "correct"
    if reason.startswith("format_failure"):
        return "format_failure"
    return "wrong"


def parse_release_number(output, template: str) -> int | None:
    """Parse a released prediction using the template's format contract."""
    if template.endswith("_json"):
        if output is None:
            return None
        match = re.search(r'"sense_number"\s*:\s*"?(\d+)"?', str(output), re.IGNORECASE)
        return int(match.group(1)) if match else None
    return parse_predicted_number(output)


def assert_no_prompt_echo(prompt_raw, output, run: str) -> None:
    """Fail if a stored response contains distinctive rendered-prompt text."""
    if prompt_raw is None or output is None:
        return
    text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
    prompt = str(prompt_raw)
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    markers = [first_line[:80], "Candidate senses:"]
    if any(marker and marker in text for marker in markers):
        raise SystemExit(f"{run}: response appears to contain rendered prompt text")


def exp1_sid(lemma: str, granularity: str, language: str, quote_idx, number: int | None) -> str | None:
    if number is None:
        return None
    ordered = numbered_senses(str(lemma).upper(), granularity, language, sense_order_seed=str(quote_idx))
    return next((sid for n, sid, _ in ordered if n == number), None)


def exp2_sid(key: str, record_id: str, lemma: str, gold_sid: str, language: str, output) -> str | None:
    number = parse_predicted_number(output)
    if number is None:
        return None
    inventory_lemma = _resolve_inv_lemma(str(lemma).upper(), gold_sid)
    ordered = numbered_senses(inventory_lemma, "l1", language, sense_order_seed=record_id)
    mapping = {n: sid for n, sid, _ in ordered}
    mapping[0] = "0"
    return mapping.get(number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".promptfoo" / "promptfoo.db")
    parser.add_argument("--registry", type=Path, default=ROOT / "ai" / "analysis" / "eval_registry.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "ai" / "release" / "generated")
    parser.add_argument("--skip-count-check", action="store_true")
    args = parser.parse_args()

    ids = final_eval_ids(args.registry)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"exp1": args.out_dir / "exp1_predictions.csv", "exp2": args.out_dir / "exp2_predictions.csv"}
    temp_paths = {kind: path.with_suffix(path.suffix + ".tmp") for kind, path in paths.items()}
    files = {}
    writers = {}
    counts = {"exp1": 0, "exp2": 0}
    try:
        for kind, path in paths.items():
            handle = temp_paths[kind].open("w", newline="", encoding="utf-8")
            files[kind] = handle
            fields = EXP1_FIELDS if kind == "exp1" else EXP2_FIELDS
            writers[kind] = csv.DictWriter(handle, fieldnames=fields)
            writers[kind].writeheader()
        with open_db(args.db) as con:
            for eval_id, (run, kind) in ids.items():
                counts[kind] += export_run(con, eval_id, run, kind, writers[kind])
    finally:
        for handle in files.values():
            handle.close()

    if not args.skip_count_check:
        expected = {"exp2_core": 33480, "exp2_extended": 100440}
        # Per-run checks are performed below by reading the exported run labels;
        # Exp-1 counts are intentionally registry-derived and reported here.
        import pandas as pd
        exp2 = pd.read_csv(temp_paths["exp2"], usecols=["run"])
        for run, wanted in expected.items():
            got = int((exp2["run"] == run).sum())
            if run == "exp2_extended" and got == 0:
                raise SystemExit("exp2_extended produced no rows; run may still be incomplete")
            if got != wanted:
                raise SystemExit(f"{run}: expected {wanted} rows, got {got}")
        for kind, path in temp_paths.items():
            import pandas as pd
            frame = pd.read_csv(path)
            axes = ["provider", "template", "granularity", "def_language", "show_metadata"] if kind == "exp1" else ["provider", "template", "def_language", "context_length", "show_metadata"]
            cell_counts = frame.groupby(axes, dropna=False).size()
            # Phase A/B and plain/JSON runs intentionally have different
            # scopes; balance is required within each run only.
            for run_name, run_frame in frame.groupby("run"):
                run_counts = run_frame.groupby(axes, dropna=False).size()
                if run_counts.nunique() != 1:
                    raise SystemExit(f"{run_name}: inconsistent provider/template/axis cell counts: {run_counts.to_dict()}")
                if run_name != "phase_a_json_restricted":
                    if not (run_frame["outcome_stored"] == run_frame["outcome_corrected"]).all():
                        raise SystemExit(f"{run_name}: stored and corrected outcomes differ")
    for kind, path in temp_paths.items():
        os.replace(path, paths[kind])
    print(f"export complete: exp1={counts['exp1']} exp2={counts['exp2']}")


if __name__ == "__main__":
    main()
