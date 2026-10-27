#!/usr/bin/env python3
"""
promptfoo transform for Experiment 2 — sense assignment on Neo-Latin
corpus concordance lines.

Sibling to scripts/transform.py (Exp 1). Reuses the sense-numbering /
inventory-loading / few-shot demo-selection machinery; differs only in the
input-presentation layer: Exp 2's inputs are KWIC sentences from the
Neo-Latin corpus sample (data/concordances/experiment/sample/), wrapped in
<WSD>…</WSD> via known char offsets, and optionally augmented with the
previous sentence (`context_length` axis).

INPUT CONTRACT — `test_vars` provided by exp2_tests.py, per test case.
Exp 2 ran annotation FIRST, so the source is eval_set.csv (one row per
analyzable gold record) and gold is known. Both the precomputed and the
raw-offset shapes are supported:
  lemma            str   lower-case headword as in eval_set.csv ("civitas")
  siglum           str   source-text siglum ("hartnoch_derepublica"); joins the
                         metadata.csv provenance row for source_meta
  show_metadata    bool  axis value; the template renders source_meta iff true
  sentence_wsd     str   PREFERRED — restored sentence with the target already
                         wrapped in <WSD>…</WSD> (from build_exp2_eval_set.py).
                         When present, used verbatim.
  sentence         str   the KWIC sentence (used only as the offset-marking
                         fallback when sentence_wsd is absent)
  prev_sent        str   the sentence immediately before, or "" if absent
  form             str   (offset fallback only) surface form at (start, end)
  match_start, match_end  int  (offset fallback only) char offsets in `sentence`
  context_length   str   "kwic_only" | "kwic_plus_prev"
  def_language     str   "la" | "pl"
  sense_order_seed str   per-record seed for inventory shuffle
  gold_sid         str   gold sense id from eval_set ("civitas.1.1", or "0" for
                         novelty); resolved to gold_number below

OUTPUT — get_vars returns `test_vars` augmented with:
  lemma            str   UPPER-cased to match the SLS entry convention
                         used by the Exp 1 inventory ("CIVITAS")
  sentence_wsd     str   sentence with the target form wrapped in <WSD>…</WSD>
  senses           str   numbered candidate-sense inventory ("1. <def>" / line)
  gold_number      int|None  the gold sense's presented number (0 = novelty);
                             omitted when gold_sid is empty/unresolvable, in
                             which case the assertion takes its pass-through
                             (no-gold) path.
  examples         str   sense-balanced few-shot demos (SLS quote pool;
                         see "Few-shot demos" below)
  source_meta      str   one-sentence provenance (show_metadata axis), joined
                         from metadata.csv on siglum; "" when no row matches

NOTE — few-shot demos
=====================
The demo pool is the Exp 1 SLS dictionary quote split (split=='demo'),
NOT Exp 2 corpus lines. This is a deliberate choice documented in
prompts/exp2/components.yaml `fewshot_examples`: per Min et al. EMNLP
2022, demos primarily teach format and label-space — which is identical
across the two experiments (numbered SLS senses, <WSD> marker, plain
output of a single integer). The SLS quote pool provides per-lemma
sense-balanced coverage that the Exp 2 corpus sample cannot match
(no gold yet, and Tier B lemmas have lopsided sense distributions).

The cross-paradigm match (dictionary citation vs naturally-bounded
corpus sentence) is the deliberate cost. The alternative — wait for
Argilla annotations to assemble an in-distribution demo pool — would
gate Exp 2 v1 on a long manual step for a benefit Min et al. predict
is small.

Data read once and cached:
  senses/senses_l1.csv  — keep-sense inventory (l1 only; Exp 2
                                   v1 does not exercise the granularity
                                   axis, per exp2_design_decisions.md)
  dictionary_quotations/quotations.csv  — SLS demo pool
"""
from __future__ import annotations
import os

import csv
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Reuse Exp 1's machinery directly — same sense inventory, same numbering
# convention, same demo-selection rule. Importing rather than copying keeps
# the two experiments in lockstep on shared mechanics.
from transform import (  # noqa: E402
    _LANG_NAME,
    _load_inventory,
    build_examples,
    format_senses,
    numbered_senses,
    gold_number as _gold_number,
)


# ----------------------------------------------------------------------------
# source metadata (show_metadata axis — EXTENDED scenario)

_METADATA_CSV = Path(os.environ["DIALEXPOL_DATA_DIR"]) / "neolatin_concordances" / "sources.csv"


@lru_cache(maxsize=1)
def _load_source_meta() -> dict:
    """siglum → one-sentence provenance string (the show_metadata block body).

    Source of truth: neolatin_concordances/sources.csv `prompt_sentence`
    column, keyed on `siglum` (joins eval_set rows). Frozen 2026-05-25; see
    analysis/exp2_design_decisions.md "Period/author metadata". Cached (4 rows).
    Returns {} if the file is absent — source_meta then renders empty and the
    show_metadata guard simply shows nothing.
    """
    if not _METADATA_CSV.exists():
        return {}
    with open(_METADATA_CSV, encoding="utf-8") as f:
        return {r["siglum"]: r.get("prompt_sentence", "")
                for r in csv.DictReader(f) if r.get("siglum")}


# ----------------------------------------------------------------------------
# inventory-key resolution (dictionary homonyms)

def _resolve_inv_lemma(lemma_uc: str, gold_sid: str) -> str:
    """Map (display lemma, gold sid) to the sense-inventory key.

    Most headwords key the inventory by the bare uppercased lemma (BELLUM,
    CIVITAS). A few are homonym-split, keyed with a hyphen suffix
    (COMITIA-1/COMITIA-2, POPULUS-1) with NO bare entry. The gold sid carries
    the homonym as an underscore prefix (`comitia_2.1` → COMITIA-2) — the
    dictionary entry the record's sample was drawn for — so we recover the key
    from it. Bare lemmas pass through unchanged.
    """
    inv = _load_inventory("l1")
    if lemma_uc in inv:
        return lemma_uc
    if gold_sid and gold_sid != "0":
        cand = gold_sid.split(".")[0].upper().replace("_", "-")
        if cand in inv:
            return cand
    return lemma_uc


# ----------------------------------------------------------------------------
# input-presentation: KWIC sentence with <WSD> marker via char offsets

def mark_sentence_wsd(sentence: str, match_start: int, match_end: int) -> str:
    """Wrap the (start, end) substring in <WSD>...</WSD>.

    Exp 2 has exact char offsets in the sample CSVs (`match_start`,
    `match_end`), so we don't need Exp 1's bracket/regex fallback. The
    offsets come from build_concordances.py which already locates the
    surface form unambiguously.
    """
    if not sentence:
        return ""
    try:
        s = int(match_start); e = int(match_end)
    except (TypeError, ValueError):
        return sentence
    if not (0 <= s < e <= len(sentence)):
        return sentence
    return f"{sentence[:s]}<WSD>{sentence[s:e]}</WSD>{sentence[e:]}"


# ----------------------------------------------------------------------------
# promptfoo entry point

def get_vars(test_vars: dict, *_) -> dict:
    """Derive the template variables for one Exp 2 test case.

    promptfoo passes `(test_vars, provider, prompt)` positionally; `*_`
    swallows provider/prompt — the transform is prompt-agnostic.
    """
    # Lemma normalisation: sample CSVs use lower-case ("civitas"); SLS inventory
    # uses upper-case ("CIVITAS"). Normalise to upper for both sense lookup and
    # the prompt's "Dictionary entry: {{lemma}}" line.
    lemma_lc = (test_vars.get("lemma") or "").strip()
    lemma_uc = lemma_lc.upper()
    test_vars["lemma"] = lemma_uc   # displayed headword (bare, no homonym suffix)

    def_language = test_vars.get("def_language", "la")
    sense_order_seed = test_vars.get("sense_order_seed")
    test_vars["sense_order_seed"] = sense_order_seed

    # Resolve the inventory key up front: needed before building senses, and it
    # depends on the gold sid for homonym-split headwords (see
    # _resolve_inv_lemma). The displayed `lemma` stays the bare form.
    gold_sid = (test_vars.pop("gold_sid", None) or "").strip()
    inv_lemma = _resolve_inv_lemma(lemma_uc, gold_sid)

    # senses (l1 only for Exp 2 v1)
    numbered = numbered_senses(inv_lemma, "l1", def_language, sense_order_seed)
    test_vars["senses"] = format_senses(numbered)

    # gold: now KNOWN at run time. The Exp 2 workflow ran annotation FIRST
    # (Argilla → curated gold), so eval_set.csv supplies a real gold_sense_id
    # per record via `gold_sid`. Resolve it to the presented number and write
    # `gold_number` so the assertion scores correctness (its replay path).
    #   - "0" is the novelty answer (no inventory sid): the prompt's literal
    #     "0. No matching sense" option, scored as gold_number 0 directly.
    #   - an empty / unresolvable sid omits the key entirely — promptfoo's
    #     `vars` validator rejects null values, and the assertion treats a
    #     missing key as the pass-through (no-gold) path.
    if gold_sid == "0":
        test_vars["gold_number"] = 0
    elif gold_sid:
        gnum = _gold_number(numbered, gold_sid)
        if gnum is not None:
            test_vars["gold_number"] = gnum

    # input presentation. Prefer a precomputed sentence_wsd: eval_set.csv
    # carries the RESTORED, curation-pipeline-marked text (the source of truth
    # for restored rows — see build_exp2_eval_set.py / wsd_mark.mark_target), so
    # re-marking from raw offsets here would mis-place the target on every
    # restored row. Fall back to offset-based marking only when no precomputed
    # value is supplied (the self-test and any caller passing raw sentence +
    # char offsets).
    precomputed_wsd = (test_vars.get("sentence_wsd") or "").strip()
    if not precomputed_wsd:
        sentence = test_vars.get("sentence", "") or ""
        test_vars["sentence_wsd"] = mark_sentence_wsd(
            sentence,
            test_vars.get("match_start", -1),
            test_vars.get("match_end", -1),
        )

    # context-length axis: prev_sent is included verbatim by the template's
    # guard ({% if context_length == "kwic_plus_prev" %}); we only need to
    # ensure the key exists and is a string.
    test_vars["prev_sent"] = (test_vars.get("prev_sent") or "").strip()

    # source metadata (show_metadata axis — EXTENDED scenario). The template
    # guard ({% if show_metadata %}) decides whether to render; we always set
    # the var so it is never undefined. Joined on siglum to
    # neolatin_concordances/sources.csv (prompt_sentence column).
    test_vars["source_meta"] = _load_source_meta().get(
        test_vars.get("siglum", ""), "")

    # few-shot demos — reuse the Exp 1 SLS-quote-pool demo builder (see
    # module docstring "few-shot demos" for the design choice).
    test_vars["examples"] = build_examples(
        inv_lemma, "l1", def_language, sense_order_seed,
    )

    test_vars["freeform_lang"] = _LANG_NAME.get(def_language, "Latin")
    return test_vars


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test against a real sample row
    sample = {
        "lemma": "civitas",
        "sentence": "Deinde JVDICIVM CAPITANEALE et CIVILE simul est, quando "
                    "Capitaneus vel ejus Officium Nobilem ob privatum Crimen "
                    "juxta Statutum Thorunense in Civitate captum cum Officio "
                    "civili judicat.",
        "prev_sent": "In Prussia nullum tale judicium reperitur, ut ex paulo "
                     "antea dictis manifestum est.",
        "form": "Civitate",
        "match_start": 141, "match_end": 149,
        "def_language": "pl",
        "context_length": "kwic_plus_prev",
        "sense_order_seed": "civitas:0",
        "siglum": "hartnoch_derepublica",
    }
    out = get_vars(dict(sample))
    print("sentence_wsd:", out["sentence_wsd"][:120], "…")
    print("gold_number :", out.get("gold_number", "<absent — run-time, no gold>"))
    print("source_meta :", out.get("source_meta", "")[:120], "…")
    print(f"\nsenses (first 200 chars):\n{out['senses'][:200]}…")
    print(f"\nexamples (first 200 chars):\n{out['examples'][:200]}…")
    assert "<WSD>Civitate</WSD>" in out["sentence_wsd"], "WSD marker did not land on the target form"
    assert out["senses"], "senses block is empty"
    assert out["examples"], "examples block is empty"
    assert out["lemma"] == "CIVITAS", "lemma was not upper-cased"
    assert out["source_meta"].startswith("This passage comes from"), \
        "source_meta did not join from metadata.csv"
    print("\nself-test OK")
