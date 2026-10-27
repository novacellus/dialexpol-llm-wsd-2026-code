#!/usr/bin/env python3
"""
promptfoo transform for Experiment 1 — sense assignment on SLS dictionary
quotations.

Called per test case as ``get_vars(test_vars, ...)``. It enriches the test-case
vars with the derived template variables the rendered prompts in
``prompts/exp1/rendered/`` expect. (In practice ``exp1_tests.py`` calls it at
generation time, so each emitted test case is already fully formed.)

INPUT CONTRACT — `test_vars` provided by exp1_tests.py, per test case:
  lemma              str   entry headword, e.g. "AUCTORITAS"
  quotation_cleaned  str   the quote with [..] headword-expansion brackets
  granularity        str   "l1" | "l1l2"  — which sense inventory to use
  def_language       str   "la" | "pl"    — definition language
  show_metadata      bool  toggles the source-metadata block
  source, page, date str   bibliographic metadata
  gold               str   reconciled gold sid for this granularity
  gold_label         str   human-readable gold label (kept for analysis only)
  (any other columns pass through untouched)

OUTPUT — get_vars returns `test_vars` augmented with:
  senses             str        numbered candidate-sense inventory ("1. <def>" / line)
  gold_number        int|None   the number (1..N) of the gold sense in `senses`
  quotation_wsd      str        quote with the target form wrapped <WSD>...</WSD>
  examples           str        few-shot demos in PLAIN format (quote -> N)
  examples_json      str        few-shot demos in JSON format, matching the
                                output_json schema (justification = the gold
                                sense's definition; confidence = 5; sense_number
                                = the shuffled position). Per Min et al. EMNLP
                                2022 demos teach format more than label
                                mappings, so the JSON template must see
                                JSON-shaped demos (PROMPT_RECOMMENDATIONS.md §2.4).
  freeform_lang      str        "Latin" | "Polish" — for the free-form condition
  show_metadata      bool       normalised to a real boolean
  sense_order_seed   str        echo of the seed actually used for shuffling

Senses are presented as a plain NUMBERED list (1..N) — not by hierarchical
label or sid: the model outputs a number, scoring is an integer match, and a
flat numbered list leaks no hierarchy structure.

Sense numbering is **seeded-random per test item**, with the seed shared
across configurations so all configs see the same shuffled order for the same
quote (PROMPT_RECOMMENDATIONS.md §2.1 — Pezeshkpour & Hruschka NAACL 2024,
Zheng et al. ICLR 2024, Zhang et al. RoDEval EMNLP 2025 DRoS). This isolates
position bias as item-level noise rather than as a between-configuration
confound. The seed is taken from `test_vars["sense_order_seed"]` (set by
exp1_tests.py from quote_idx); if absent, falls back to natural-sid order
for backwards compatibility with the diagnostic tools that call get_vars
directly. `gold` (sid) and `gold_label` stay in the vars so number→sid is
recoverable for analysis; few-shot demos use the SAME shuffled numbering as
the live test item so demos teach the inventory the model sees.

Data read once and cached:
  senses/senses_l1.csv, senses_exp2_l1l2.csv  — keep-sense inventories
  dictionary_quotations/quotations.csv            — demo pool (split=='demo')
"""
from __future__ import annotations
import os

import csv
import json
import random
import re
from pathlib import Path

DATA = Path(os.environ["DIALEXPOL_DATA_DIR"])
INVENTORY_CSV = {
    "l1": DATA / "senses" / "senses_l1.csv",
    "l1l2": DATA / "senses" / "senses_l1l2.csv",
}
SPLIT_CSV = DATA / "dictionary_quotations" / "quotations.csv"

FEWSHOT_K = 3          # PLAN.md §2.2 — few-shot demonstrations per prompt
DEMO_SEED = 20260514   # reproducible sense-balanced demo selection

_LANG_NAME = {"la": "Latin", "pl": "Polish"}

# ----------------------------------------------------------------------------
# cached loaders

_inventory_cache: dict[str, dict[str, list[dict]]] = {}
_demo_cache: dict[tuple[str, str], list[dict]] | None = None
_demo_fallback_cache: dict[tuple[str, str], list[dict]] | None = None


def _load_inventory(granularity: str) -> dict[str, list[dict]]:
    """lemma -> [ {sid, label, def_pl, def_la}, ... ] in inventory-file order."""
    if granularity not in _inventory_cache:
        by_lemma: dict[str, list[dict]] = {}
        with open(INVENTORY_CSV[granularity], encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_lemma.setdefault(row["lemma"], []).append({
                    "sid": row["sid"],
                    "label": row["label"],
                    "def_pl": (row.get("def_pl") or "").strip(),
                    "def_la": (row.get("def_la") or "").strip(),
                })
        _inventory_cache[granularity] = by_lemma
    return _inventory_cache[granularity]


def _build_demo_caches() -> None:
    """Populate, in one pass over the split CSV, both demo caches keyed by
    (lemma, granularity): `_demo_cache` (split=='demo') and
    `_demo_fallback_cache` (split!='demo')."""
    global _demo_cache, _demo_fallback_cache
    demo: dict[tuple[str, str], list[dict]] = {}
    fallback: dict[tuple[str, str], list[dict]] = {}
    with open(SPLIT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = demo if row.get("split") == "demo" else fallback
            for gran in ("l1", "l1l2"):
                gold = row.get(f"gold_{gran}")
                if not gold:                     # out of scope at this granularity
                    continue
                target.setdefault((row["lemma"], gran), []).append({
                    "quotation_cleaned": row["quotation_cleaned"],
                    "gold_sid": gold,
                })
    _demo_cache, _demo_fallback_cache = demo, fallback


def _load_demos() -> dict[tuple[str, str], list[dict]]:
    """(lemma, granularity) -> [ {quotation_cleaned, gold_sid}, ... ] — the
    demo-pool quotes (split=='demo') in scope at that granularity."""
    if _demo_cache is None:
        _build_demo_caches()
    assert _demo_cache is not None
    return _demo_cache


def _load_demo_fallback() -> dict[tuple[str, str], list[dict]]:
    """(lemma, granularity) -> the lemma's NON-demo SLS quotes.

    Used by `_pick_demos` ONLY for a lemma with zero demo-tagged quotes
    (e.g. PRAEFECTUS: 4 SLS quotes, none tagged `split=='demo'`). Safe by
    construction: in Exp 1 every evaluated lemma has demo quotes, so this never
    fires; in Exp 2 the eval items are corpus lines — disjoint from all SLS
    dictionary quotes — so any SLS quote is a leak-free demo. See
    analysis/exp2_analysis_plan.md §11-D."""
    if _demo_fallback_cache is None:
        _build_demo_caches()
    assert _demo_fallback_cache is not None
    return _demo_fallback_cache


# ----------------------------------------------------------------------------
# sense presentation

def _definition(sense: dict, def_language: str) -> str:
    """Definition in the requested language, falling back to the other."""
    primary, other = ("def_pl", "def_la") if def_language == "pl" else ("def_la", "def_pl")
    return sense[primary] or sense[other] or ""


def numbered_senses(
    lemma: str,
    granularity: str,
    def_language: str,
    sense_order_seed: str | int | None = None,
) -> list[tuple[int, str, str]]:
    """Ordered [(number, sid, definition), ...] for the lemma.

    Number is the 1-based position among the definition-bearing senses. Senses
    with no definition in either language are skipped — they get no number and
    cannot be chosen.

    If ``sense_order_seed`` is provided, the (sid, definition) pairs are
    shuffled by a Random seeded on the string ``f"sense_order:{seed}"`` before
    being numbered, so the same seed always produces the same order (and the
    same seed across configurations isolates positional bias as item-level
    rather than configuration-level noise). If None, falls back to inventory-
    file natural-sid order (used by the self-test and ad-hoc diagnostics).
    """
    pool: list[tuple[str, str]] = []  # (sid, definition)
    for s in _load_inventory(granularity).get(lemma, []):
        d = _definition(s, def_language)
        if not d:
            continue
        pool.append((s["sid"], d))

    if sense_order_seed is not None and pool:
        rng = random.Random(f"sense_order:{sense_order_seed}")
        rng.shuffle(pool)

    return [(i + 1, sid, d) for i, (sid, d) in enumerate(pool)]


def format_senses(numbered: list[tuple[int, str, str]]) -> str:
    """The candidate-sense block: one 'N. definition' line per sense."""
    return "\n".join(f"{n}. {d}" for n, _, d in numbered)


def gold_number(numbered: list[tuple[int, str, str]], gold_sid: str) -> int | None:
    """The presented number of the gold sense, or None if it is not in the list."""
    for n, sid, _ in numbered:
        if sid == gold_sid:
            return n
    return None


# ----------------------------------------------------------------------------
# quotation marking

_BRACKET_TOKEN = re.compile(r"\S*\[[^\]]+\]\S*")


def mark_wsd(quotation_cleaned: str, lemma: str) -> str:
    """Wrap the target headword form in <WSD>...</WSD>.

    The headword-expansion bracket (e.g. ``a[uctoritat]is``) pinpoints the
    target in ~99% of quotes; we wrap that token and strip its brackets. For the
    rare bracket-less quote, fall back to a lemma-stem regex; if that also
    fails, return the quote unmarked.
    """
    text = quotation_cleaned or ""
    if "[" in text:
        return _BRACKET_TOKEN.sub(
            lambda m: "<WSD>" + m.group(0).replace("[", "").replace("]", "") + "</WSD>",
            text,
        )
    stem = re.escape(lemma.split("-")[0][:4].lower())
    if stem:
        marked, n = re.subn(
            rf"\b(\w*{stem}\w*)\b", r"<WSD>\1</WSD>", text, count=1, flags=re.IGNORECASE,
        )
        if n:
            return marked
    return text


# ----------------------------------------------------------------------------
# few-shot demonstrations

def _pick_demos(
    lemma: str,
    granularity: str,
    def_language: str,
    sense_order_seed: str | int | None,
) -> list[tuple[str, str, int, str]]:
    """Pick the few-shot demos. Returns [(quote_wsd, sid, sense_number, definition), ...].

    Used by both build_examples (plain format) and build_examples_json (JSON
    format) — same demo SELECTION; differs only in the rendered output shape.
    Sense numbering uses the same shuffled order as the live test item
    (``sense_order_seed``) so demos teach the inventory the model sees.
    """
    demos = _load_demos().get((lemma, granularity), [])
    if not demos:
        # No demo-tagged quotes for this lemma (e.g. PRAEFECTUS) — fall back to
        # its other SLS quotes. Only fires when the demo pool is empty, so
        # demo'd lemmas are unaffected. See _load_demo_fallback.
        demos = _load_demo_fallback().get((lemma, granularity), [])
    numbered = numbered_senses(lemma, granularity, def_language, sense_order_seed)
    sid_to_num = {sid: n for n, sid, _ in numbered}
    sid_to_def = {sid: d for _, sid, d in numbered}

    by_sense: dict[str, list[dict]] = {}
    for d in demos:
        by_sense.setdefault(d["gold_sid"], []).append(d)

    # only senses that actually have a presented number can be demonstrated
    senses = [sid for sid in sorted(by_sense) if sid in sid_to_num]
    rng = random.Random(f"{DEMO_SEED}:{lemma}:{granularity}")
    rng.shuffle(senses)

    picked: list[tuple[str, str, int, str]] = []
    for sid in senses[:FEWSHOT_K]:
        d = rng.choice(by_sense[sid])
        picked.append((
            mark_wsd(d["quotation_cleaned"], lemma),
            sid,
            sid_to_num[sid],
            sid_to_def[sid],
        ))
    return picked


def build_examples(
    lemma: str,
    granularity: str,
    def_language: str,
    sense_order_seed: str | int | None = None,
) -> str:
    """Static, sense-balanced few-shot demonstrations in PLAIN format.

    Format: ``"<quote>" -> N`` per demo, one per line. Used by the
    ``fewshot_examples`` block; renders into the plain-output fewshot template.
    See _pick_demos for selection logic.
    """
    return "\n".join(
        f'"{quote}" -> {n}'
        for quote, _sid, n, _defn in _pick_demos(lemma, granularity, def_language, sense_order_seed)
    )


def build_examples_json(
    lemma: str,
    granularity: str,
    def_language: str,
    sense_order_seed: str | int | None = None,
) -> str:
    """Static, sense-balanced few-shot demonstrations in JSON format.

    Format per demo: ``"<quote>" -> {"justification": "<sense def>", "confidence": 5, "sense_number": N}``.
    The justification text is the gold sense's *definition* (information the
    model already has from the candidate list — not leaky); confidence is 5
    (gold-standard demos = max confidence by construction). Key order matches
    output_json (justification → confidence → sense_number, per
    PROMPT_RECOMMENDATIONS.md §1.2). Used by the ``fewshot_examples_json``
    block; the format matches the model's expected output schema per Min et
    al. EMNLP 2022.
    """
    lines = []
    for quote, _sid, n, defn in _pick_demos(lemma, granularity, def_language, sense_order_seed):
        # dict is insertion-ordered (Python 3.7+); json.dumps preserves the order.
        payload = {"justification": defn, "confidence": 5, "sense_number": n}
        lines.append(f'"{quote}" -> {json.dumps(payload, ensure_ascii=False)}')
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# promptfoo entry point

def get_vars(test_vars: dict, *_) -> dict:
    """Derive the template variables for one test case (see module docstring).

    promptfoo passes `(test_vars, provider, prompt)` positionally; `*_` swallows
    provider/prompt — the transform is prompt-agnostic.
    """
    lemma = test_vars["lemma"]
    granularity = test_vars.get("granularity", "l1")
    def_language = test_vars.get("def_language", "la")
    sense_order_seed = test_vars.get("sense_order_seed")  # set by exp1_tests.py
    test_vars["sense_order_seed"] = sense_order_seed       # echo for analysis

    numbered = numbered_senses(lemma, granularity, def_language, sense_order_seed)
    test_vars["senses"] = format_senses(numbered)
    test_vars["gold_number"] = gold_number(numbered, test_vars.get("gold", ""))
    # Prefer the precomputed quotation_plain_wsd column (sc notes stripped,
    # brackets resolved, headword wrapped in <WSD>...</WSD>) — written to the
    # split CSV by scripts/augment_split_with_wsd.py per Exp 1's clean-quote
    # design (PLAN.md §2.2). Fall back to the legacy runtime mark_wsd over
    # quotation_cleaned when the column is absent (older split CSVs).
    plain_wsd = test_vars.get("quotation_plain_wsd")
    if plain_wsd:
        test_vars["quotation_wsd"] = plain_wsd
    else:
        test_vars["quotation_wsd"] = mark_wsd(test_vars.get("quotation_cleaned", ""), lemma)
    test_vars["examples"] = build_examples(lemma, granularity, def_language, sense_order_seed)
    test_vars["examples_json"] = build_examples_json(lemma, granularity, def_language, sense_order_seed)
    test_vars["freeform_lang"] = _LANG_NAME.get(def_language, "Latin")

    sm = test_vars.get("show_metadata", False)
    test_vars["show_metadata"] = (
        sm if isinstance(sm, bool)
        else str(sm).strip().lower() in ("true", "1", "yes")
    )
    return test_vars


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test against the real data files
    sample = {
        "lemma": "AUCTORITAS",
        "quotation_cleaned": "qui in omnes reges habet a[uctoritat]em et imperium",
        "granularity": "l1",
        "def_language": "pl",
        "show_metadata": "true",
        "source": "KodWp III", "page": "p. 97", "date": "a. 1358",
        "gold": "auctoritas.1.1.2", "gold_label": "I.A.2",
    }
    out = get_vars(dict(sample))
    print("quotation_wsd:", repr(out["quotation_wsd"]))
    print("gold_number  :", repr(out["gold_number"]))
    print("freeform_lang:", repr(out["freeform_lang"]))
    print("show_metadata:", repr(out["show_metadata"]), type(out["show_metadata"]).__name__)
    print(f"\nsenses:\n{out['senses']}")
    print(f"\nexamples:\n{out['examples']}")
    assert out["senses"], "senses block is empty"
    assert out["examples"], "examples block is empty"
    assert isinstance(out["gold_number"], int) and out["gold_number"] >= 1, "gold_number unresolved"
    assert out["quotation_wsd"].count("<WSD>") == 1, "expected exactly one WSD marker"
    print("\nself-test OK")
