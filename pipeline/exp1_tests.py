#!/usr/bin/env python3
"""
promptfoo test-case generator for Experiment 1.

Reads ``dictionary_quotations/quotations.csv`` (the demo/test split with
reconciled gold) and emits the expanded screening matrix: one promptfoo test
case per

    (in-scope test quote) × granularity × def_language × show_metadata

Each test case is **fully formed**: the raw vars are enriched here by
``transform.py:get_vars`` (``senses``, ``quotation_wsd``, ``examples``,
``freeform_lang``, normalised ``show_metadata``), so promptfoo needs no
separate transform step. Out-of-scope quotes (no reconciled gold at a given
granularity) are skipped for that granularity. Demo-pool quotes are never
emitted as test cases — they are used by ``transform.py`` to build ``examples``.

promptfoo entry points:
  generate_phase_a()  — Phase A screening lemmas (PLAN.md §2.2)
  generate_phase_b()  — Phase B lemmas (10+ usable quotes, excluding Phase A)
  generate_all()      — every lemma

Wire into promptfooconfig.yaml as e.g.:
  tests: file://scripts/exp1_tests.py:generate_phase_a
"""
from __future__ import annotations
import os

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transform import get_vars  # noqa: E402  (sys.path adjusted just above)

DATA = Path(os.environ["DIALEXPOL_DATA_DIR"])
SPLIT_CSV = DATA / "dictionary_quotations" / "quotations.csv"

# Phase A screening lemmas. Derived 2026-05-15 by composite ranking — see
# analysis/exp1_quote_design.ipynb §6 (the decision-support cell) and
# analysis/exp1_design_decisions.md §"Phase A lemma selection". n_test = 422
# (84% of the PLAN §2.2 ~500-quote target; the shortfall is documented).
PHASE_A = ["LIBERTAS", "OFFICIUM", "AUCTORITAS", "DOMINUS", "IURISDICTIO",
           "COMMUNITAS", "DOMINIUM", "COMES", "CONSILIUM", "HONOR", "CONVENTIO"]
PHASE_B_MIN_QUOTES = 10  # PLAN.md §2.3 — "all lemmas with 10+ quotations"

# Run-time axes (PLAN.md §2.2 / prompts/exp1/conditions.yaml).
# Phase A runs the full matrix; Phase B restricts it — see PHASE_B_* below.
GRANULARITIES = ["l1", "l1l2"]
DEF_LANGUAGES = ["la", "pl"]
SHOW_METADATA = [True, False]

# Phase B axis restriction (methodology_log entry 2026-05-16a).
#   show_metadata: dropped True — metadata block carried no model-actionable
#                  signal in Phase A (eval-1Zw axis_marginals: +0.43 pp, the
#                  weakest of the three axes; analysis notebook §4).
#   granularity:   dropped l1l2 — two-level granularity deferred to a future
#                  study (no operational need for this paper).
#   def_language:  both la and pl retained (la↔pl is a kept comparison).
# Phase A data on the dropped axes is preserved in eval-1Zw; Phase B simply
# does not collect new data on them. The 4 templates × 2 def_languages = 8
# surviving configs are the §7b promoted set carried into Phase B.
PHASE_B_GRANULARITIES = ["l1"]
PHASE_B_DEF_LANGUAGES = DEF_LANGUAGES        # both retained
PHASE_B_SHOW_METADATA = [False]


def _load_rows() -> list[dict]:
    with open(SPLIT_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate(
    lemmas: set[str] | None = None,
    granularities: list[str] = GRANULARITIES,
    def_languages: list[str] = DEF_LANGUAGES,
    show_metadata_vals: list[bool] = SHOW_METADATA,
) -> list[dict]:
    """Emit promptfoo test cases for the given lemmas (None = all lemmas).

    Axis defaults are the full Phase A matrix; per-phase generators pass
    restricted subsets via the PHASE_*_ constants above (see generate_phase_b)."""
    cases: list[dict] = []
    for r in _load_rows():
        if r.get("split") != "test":
            continue
        if lemmas is not None and r["lemma"] not in lemmas:
            continue
        for gran in granularities:
            gold = r.get(f"gold_{gran}")
            gold_label = r.get(f"gold_{gran}_label")
            if not gold or not gold_label:        # out of scope at this granularity
                continue
            for def_language in def_languages:
                for show_metadata in show_metadata_vals:
                    case_vars = {
                        "lemma": r["lemma"],
                        "quotation_cleaned": r["quotation_cleaned"],
                        # The clean LLM-input form (sc notes stripped, brackets
                        # resolved, headword wrapped in <WSD>...</WSD>) computed
                        # once at data-prep time by scripts/augment_split_with_wsd.py.
                        # transform.py.get_vars prefers this over runtime mark_wsd.
                        "quotation_plain_wsd": r.get("quotation_plain_wsd", ""),
                        "granularity": gran,
                        "def_language": def_language,
                        "show_metadata": show_metadata,
                        "source": r["source"],
                        "page": r["page"],
                        "date": r["date"],
                        "gold": gold,
                        "gold_label": gold_label,
                        # Sense-order shuffle seed (PROMPT_RECOMMENDATIONS.md §2.1):
                        # tied to quote_idx so every (template, axis, model) config
                        # sees the same shuffled order for the same quote.
                        "sense_order_seed": r["quote_idx"],
                    }
                    get_vars(case_vars)  # enrich in place: senses, quotation_wsd, examples, ...
                    if case_vars.get("gold_number") is None:
                        # gold sense carries no presentable definition -> no
                        # number -> not scorable; treat as out of scope.
                        continue
                    cases.append({
                        "vars": case_vars,
                        # analysis covariates — carried through to results,
                        # not used by templates (axis values are in `vars`).
                        "metadata": {
                            "quote_idx": r["quote_idx"],
                            "n_words": r["n_words"],
                            "length_bin": r["length_bin"],
                            "phase": "A" if r["lemma"] in PHASE_A else "B",
                        },
                        "description": (
                            f"{r['lemma']} q{r['quote_idx']} "
                            f"[{gran}/{def_language}/"
                            f"{'meta' if show_metadata else 'nometa'}]"
                        ),
                    })
    return cases


def _phase_b_lemmas() -> set[str]:
    """Lemmas with >= PHASE_B_MIN_QUOTES usable quotes, excluding Phase A."""
    counts = Counter(r["lemma"] for r in _load_rows())   # all usable rows
    return {lemma for lemma, n in counts.items()
            if n >= PHASE_B_MIN_QUOTES and lemma not in PHASE_A}


def generate_phase_a(*_) -> list[dict]:
    return generate(set(PHASE_A))


def generate_phase_a_restricted(*_) -> list[dict]:
    """Phase A lemmas on Phase B's restricted axis matrix.

    Use to collect a Phase A JSON eval whose axis matrix matches Phase B
    (`granularity=l1`, `show_metadata=False`, both `def_language` values)
    so the paired plain↔json analysis runs on the 8 surviving configs only.
    Cuts cost by ~75 % vs full-matrix Phase A JSON; the dropped axes
    already have plain-side data in eval-1Zw.
    """
    return generate(
        set(PHASE_A),
        granularities=PHASE_B_GRANULARITIES,
        def_languages=PHASE_B_DEF_LANGUAGES,
        show_metadata_vals=PHASE_B_SHOW_METADATA,
    )


def generate_phase_b(*_) -> list[dict]:
    return generate(
        _phase_b_lemmas(),
        granularities=PHASE_B_GRANULARITIES,
        def_languages=PHASE_B_DEF_LANGUAGES,
        show_metadata_vals=PHASE_B_SHOW_METADATA,
    )


def generate_all(*_) -> list[dict]:
    return generate(None)


def generate_smoke(*_) -> list[dict]:
    """Tiny matrix for pipeline smoke-testing: AUCTORITAS quotes spanning
    several DISTINCT gold senses, all axis combinations (~3 quotes × 8 combos
    ≈ 24 cases). Exercises every code path — both granularities, both
    definition languages, metadata on/off, few-shot examples — without
    generating the full screening matrix.

    The gold spread matters: a naive "first N quotes" can all share one gold
    (AUCTORITAS's first two both reconcile to sense 1), which makes scoring
    degenerate — every answer key the same. This picks quotes that introduce
    distinct gold numbers so the assertion is genuinely exercised.
    """
    cases = generate({"AUCTORITAS"})
    seen_golds: set = set()
    keep_quotes: set = set()
    for c in cases:
        qi = c["metadata"]["quote_idx"]
        g = c["vars"].get("gold_number")
        if qi in keep_quotes:
            continue
        if g not in seen_golds and len(keep_quotes) < 3:
            seen_golds.add(g)
            keep_quotes.add(qi)
    return [c for c in cases if c["metadata"]["quote_idx"] in keep_quotes]


def generate_pilot(*_) -> list[dict]:
    """Pilot matrix per PLAN.md §2.2 'Cost guardrails': 1 lemma (AUCTORITAS) ×
    full variable axes (granularity × def_language × show_metadata = 8) ×
    all in-scope test quotes for that lemma. Sized for pipeline + per-call cost
    validation on the 5 mid-tier Phase A models.

    AUCTORITAS chosen for continuity (every smoke since 2026-05-14 used it;
    the lemma's data has not been used for prompt iteration, only pipeline
    debugging, so its quotes carry into Phase A unchanged per PLAN §2.2).
    """
    return generate({"AUCTORITAS"})


def generate_pilot_flagship(*_) -> list[dict]:
    """Flagship feasibility slice per PLAN.md §2.2 'Cost guardrails' and
    'Budget fallback': a fixed first-N slice of generate_pilot, just enough
    to validate the flagships handle this task (format compliance, real
    per-call cost, refusal patterns, rate limits) without spending the full
    flagship pilot budget.

    Stable across runs — slice by sorted (quote_idx, granularity, def_language,
    show_metadata) — so the same items run on both flagships.
    """
    SLICE_N = 8  # ~8 cases × 6 prompts × 2 flagships ≈ 96 flagship calls
    cases = generate({"AUCTORITAS"})
    cases.sort(key=lambda c: (c["metadata"]["quote_idx"],
                              c["vars"]["granularity"],
                              c["vars"]["def_language"],
                              c["vars"]["show_metadata"]))
    return cases[:SLICE_N]


if __name__ == "__main__":
    import json

    for name, fn in [("phase_a", generate_phase_a),
                     ("phase_b", generate_phase_b),
                     ("all", generate_all),
                     ("smoke", generate_smoke)]:
        cases = fn()
        by_gran = Counter(c["vars"]["granularity"] for c in cases)
        lemmas = {c["vars"]["lemma"] for c in cases}
        print(f"{name:9s}: {len(cases):5d} test cases | {len(lemmas):2d} lemmas | "
              f"l1={by_gran['l1']} l1l2={by_gran['l1l2']}")

    print("\nsample test case:")
    print(json.dumps(generate_phase_a()[0], indent=2, ensure_ascii=False))
