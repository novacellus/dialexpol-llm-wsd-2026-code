#!/usr/bin/env python3
"""
promptfoo test-case generator for Experiment 2.

Reads the single canonical runner input ``neolatin_concordances/concordance_lines.csv`` —
one row per analyzable gold record, produced by the curation pipeline
(``build_exp2_eval_set.py``; see analysis/exp2_pipeline.md). Emits the Exp 2
runtime matrix: one promptfoo test case per

    (analyzable gold record) × def_language × context_length

Each test case is fully formed: raw vars are enriched here by
``exp2_transform.py:get_vars`` (``senses``, ``examples``; the ``sentence_wsd``
and ``gold_number`` come straight from eval_set), so promptfoo needs no
separate transform step.

Why eval_set.csv (not the raw concordance sample CSVs the earlier draft read):
the sample CSVs are the PRE-annotation 988-line pool with no gold and the
un-restored sentence text. eval_set.csv is the post-annotation gold set —
839 analyzable records, gold known, sentences RESTORED, and the <WSD> target
already marked by the curation pipeline. The runner must use it so it (a) runs
only analyzable records, (b) honours the restored text + curated WSD mark, and
(c) scores against real gold. See analysis/exp2_pipeline.md step 8/9.

Stage matrix (analysis/exp2_design_decisions.md "Core vs extended scenarios"):
  axes = 4 templates × 2 def_languages × 2 context_lengths × 2 show_metadata.
  CORE     = kwic_only × meta off = 8 cells/provider (exp1-analogous headline).
  EXTENDED = the 24 remaining cells (any +prev and/or meta on), disjoint.
  FULL     = the whole 32-cell grid (CORE ∪ EXTENDED).
  5 providers × 837 records: CORE ≈ 33K calls, FULL ≈ 134K.

promptfoo entry points:
  generate_smoke()    — civitas, first 3 records, full axis grid (24 cases) —
                        pipeline smoke test, no LLM credits at scale
  generate_pilot()    — one Tier B lemma (bellum) × full axis grid —
                        single-lemma feasibility check
  generate_core()     — CORE scenario (promptfooconfig.exp2_core.yaml)
  generate_extended() — EXTENDED scenario (promptfooconfig.exp2_extended.yaml)
  generate_full()     — the whole 32-cell grid in one run
"""
from __future__ import annotations
import os

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp2_transform import get_vars  # noqa: E402

DATA = Path(os.environ["DIALEXPOL_DATA_DIR"])
EVAL_SET = DATA / "neolatin_concordances" / "concordance_lines.csv"

# Lemma allocation (analysis/exp2_design_decisions.md §"Lemma selection").
# Mirrors scripts/deploy_experiment_2.sh PRIORITY/SECONDARY constants — the
# two must stay aligned so the LLM eval scope matches the annotator scope.
TIER_A = ["civitas", "populus", "dignitas", "imperator",
          "natio", "episcopus", "comitia", "praefectus"]
TIER_B = ["officium", "lex", "princeps", "libertas", "bellum", "dux"]
EXP2_LEMMAS = TIER_A + TIER_B

# Run-time axes (exp2_design_decisions.md §"LLM evaluation").
# v1 drops Exp 1's granularity, show_metadata, and yaml_kind axes — see the
# design doc for the rationale. context_length is Exp 2-specific.
DEF_LANGUAGES = ["la", "pl"]
CONTEXT_LENGTHS = ["kwic_only", "kwic_plus_prev"]
# show_metadata axis (EXTENDED scenario). CORE pins meta off; see
# analysis/exp2_design_decisions.md "Core vs extended scenarios".
SHOW_METADATA = [False, True]

# Analysis covariates carried through to results (not used by templates).
# These are the eval_set columns the post-hoc analysis (exp2_iaa.ipynb,
# exp2_analysis_plan.md) stratifies/filters on.
_COVARIATES = ["gold_category", "resolution", "restored",
               "wsd_status", "context_truncated", "n_annotators",
               "min_conf", "mean_conf", "genuine_ambiguity"]


# ----------------------------------------------------------------------------
# loader

def _load_rows() -> list[dict]:
    if not EVAL_SET.exists():
        sys.exit(f"eval_set.csv not found at {EVAL_SET} — run "
                 f"scripts/build_exp2_eval_set.py first (see exp2_pipeline.md).")
    with open(EVAL_SET, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------
# generator

def generate(
    lemmas: list[str] | None = None,
    def_languages: list[str] = DEF_LANGUAGES,
    context_lengths: list[str] = CONTEXT_LENGTHS,
    show_metadatas: list[bool] = SHOW_METADATA,
    max_per_lemma: int | None = None,
    exclude_core: bool = False,
) -> list[dict]:
    """Emit promptfoo test cases.

    lemmas          : subset of EXP2_LEMMAS, or None = all 14.
    def_languages   : la / pl subset.
    context_lengths : kwic_only / kwic_plus_prev subset.
    show_metadatas  : [False] / [True] / both — the show_metadata axis.
    max_per_lemma   : optional per-lemma truncation for pilot / smoke runs
                      (eval_set.csv is lemma-contiguous, so this slices the
                      first N records of each lemma).
    exclude_core    : when True, skip the CORE cell (context_length=="kwic_only"
                      AND show_metadata is False) so EXTENDED is disjoint from
                      CORE (no overlap, no double-pay).
    """
    want = set(lemmas) if lemmas is not None else None
    cases: list[dict] = []
    per_lemma: Counter = Counter()
    for r in _load_rows():
        lemma = r["lemma"]
        if want is not None and lemma not in want:
            continue
        if max_per_lemma is not None:
            if per_lemma[lemma] >= max_per_lemma:
                continue
            per_lemma[lemma] += 1
        for def_language in def_languages:
            for context_length in context_lengths:
                for show_metadata in show_metadatas:
                    # CORE cell = kwic_only + meta off. EXTENDED is its
                    # complement, so it skips that one combination.
                    if (exclude_core and context_length == "kwic_only"
                            and not show_metadata):
                        continue
                    case_vars = {
                        # source identifiers (analysis correlation keys). `key`
                        # (lemma::record_id) is the join key; record_id alone is
                        # NOT unique across lemmas (exp2_pipeline.md "Gotchas").
                        "key": r["key"],
                        "record_id": r["record_id"],
                        "lemma": lemma,
                        "siglum": r["siglum"],
                        # linguistic input — sentence_wsd is the RESTORED,
                        # curated, already-<WSD>-marked text; used verbatim.
                        "sentence": r["sentence"],
                        "sentence_wsd": r["sentence_wsd"],
                        "prev_sent": r.get("prev_sent", ""),
                        "next_sent": r.get("next_sent", ""),
                        # axis values
                        "def_language": def_language,
                        "context_length": context_length,
                        "show_metadata": show_metadata,
                        # Sense-order shuffle seed (PROMPT_RECOMMENDATIONS.md
                        # §2.1): tied to record_id so every (template, axis,
                        # model) cell sees the same shuffled order per record.
                        "sense_order_seed": r["record_id"],
                        # gold is KNOWN (annotation ran first). exp2_transform
                        # maps this to gold_number; "0" = novelty.
                        "gold_sid": r["gold_sense_id"],
                    }
                    get_vars(case_vars)
                    cases.append({
                        "vars": case_vars,
                        "metadata": {
                            "key": r["key"],
                            "record_id": r["record_id"],
                            "siglum": r["siglum"],
                            "gold_sense_id": r["gold_sense_id"],
                            **{c: r.get(c, "") for c in _COVARIATES},
                        },
                        "description": (
                            f"{lemma} {r['record_id']} "
                            f"[{def_language}/{context_length}"
                            f"/{'meta' if show_metadata else 'nometa'}]"
                        ),
                    })
    return cases


# ----------------------------------------------------------------------------
# named entry points

def generate_smoke(*_) -> list[dict]:
    """One lemma × first 3 records × full axis grid = 3×2×2×2 = 24 cases.

    Exercises both context-length variants, both definition languages, AND both
    show_metadata states (so the metadata block is actually rendered) without
    spending LLM credits at scale.
    """
    return generate(lemmas=["civitas"], max_per_lemma=3)


def generate_pilot(*_) -> list[dict]:
    """One Tier B lemma × all analyzable records × full axis grid.

    bellum chosen: low-sense anchor (few L1 senses), large N — a single-lemma
    full matrix to validate per-call cost + per-provider behaviour before
    committing to the 14-lemma run. (For novelty-`0` behaviour, pilot `civitas`
    instead — it carries all 13 gold-novelty records; bellum has none.)
    """
    return generate(lemmas=["bellum"])


def generate_core(*_) -> list[dict]:
    """CORE scenario (exp1-analogous headline): all 14 lemmas × every record ×
    {la,pl} × kwic_only × meta off = 8 cells/provider. See
    analysis/exp2_design_decisions.md "Core vs extended scenarios".
    """
    return generate(context_lengths=["kwic_only"], show_metadatas=[False])


def generate_extended(*_) -> list[dict]:
    """EXTENDED scenario: the 24 cells CORE does not cover (any kwic_plus_prev
    and/or show_metadata on), disjoint from CORE so the two run with no overlap.
    """
    return generate(exclude_core=True)


def generate_full(*_) -> list[dict]:
    """All 14 lemmas × every record × the full 4 × 2 × 2 × 2 = 32-cell grid
    (CORE ∪ EXTENDED in a single run)."""
    return generate()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    for name, fn in [("smoke", generate_smoke),
                     ("pilot", generate_pilot),
                     ("core", generate_core),
                     ("extended", generate_extended),
                     ("full", generate_full)]:
        cases = fn()
        by_ctx = Counter(c["vars"]["context_length"] for c in cases)
        by_lang = Counter(c["vars"]["def_language"] for c in cases)
        by_meta = Counter(c["vars"]["show_metadata"] for c in cases)
        lemmas = {c["vars"]["lemma"] for c in cases}
        n_novelty = sum(c["vars"].get("gold_number") == 0 for c in cases)
        print(f"{name:9s}: {len(cases):5d} cases | {len(lemmas):2d} lemmas | "
              f"la={by_lang['la']} pl={by_lang['pl']} | "
              f"kwic={by_ctx['kwic_only']} +prev={by_ctx['kwic_plus_prev']} | "
              f"meta_on={by_meta[True]} meta_off={by_meta[False]} | "
              f"novelty={n_novelty}")

    print("\nsample test case (first smoke case):")
    print(json.dumps(generate_smoke()[0], indent=2, ensure_ascii=False)[:1400])
