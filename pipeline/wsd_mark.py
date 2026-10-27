#!/usr/bin/env python3
"""
WSD target marking for Experiment 2 — locate the lemma occurrence in a quote and
wrap it in <WSD>…</WSD> (the same tag Experiment 1 uses), robust to restoration.

The Argilla bolding (from char offsets) is lost once a quote is restored/edited,
so we re-locate the target from the recorded surface `form`. A bare single-word
match is not enough — a form can occur several times in one sentence — so when
there is more than one candidate we disambiguate with a **multi-word window**:
the W tokens on each side of the target in the *original* sentence are matched
against the neighbourhood of each candidate in the *final* sentence, and the
best-matching occurrence wins.

`mark_target` returns (marked_sentence, status):
    unique    — exactly one occurrence of the form; marked it
    window    — several occurrences; picked the one whose ±W-token neighbourhood
                best matches the original (needs orig sentence + offsets)
    ambiguous — several occurrences and no window matched well → leave for manual
    missing   — the form does not occur (spelling changed in restoration) → manual
    noform    — no recorded form

Self-test:  python scripts/wsd_mark.py --selftest
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

TAG_OPEN, TAG_CLOSE = "<WSD>", "</WSD>"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s or "", re.UNICODE)


def _neighbourhood(text: str, s: int, e: int, w: int) -> str:
    """The w tokens before `s` + w tokens after `e` (the target itself excluded)."""
    left = _tokens(text[:s])[-w:]
    right = _tokens(text[e:])[:w]
    return " ".join(left + right).lower()


def _wrap(text: str, s: int, e: int) -> str:
    return text[:s] + TAG_OPEN + text[s:e] + TAG_CLOSE + text[e:]


def mark_target(final: str, form: str, orig: str | None = None,
                match_start=None, match_end=None, window: int = 3) -> tuple[str, str]:
    """Mark the target lemma in `final`. See module docstring for `status` values."""
    final = _norm(final)
    if not form:
        return final, "noform"

    occ = [(m.start(), m.end())
           for m in re.finditer(r"\b" + re.escape(form) + r"\b", final, re.IGNORECASE)]
    if not occ:
        return final, "missing"
    if len(occ) == 1:
        s, e = occ[0]
        return _wrap(final, s, e), "unique"

    # Several candidates → disambiguate by the original ±window neighbourhood.
    ms = me = None
    try:
        if match_start not in (None, "") and match_end not in (None, ""):
            ms, me = int(match_start), int(match_end)
    except (TypeError, ValueError):
        ms = me = None
    if orig and ms is not None and me is not None:
        target = _neighbourhood(orig, ms, me, window)
        best, chosen = -1.0, None
        for s, e in occ:
            sim = SequenceMatcher(None, target, _neighbourhood(final, s, e, window)).ratio()
            if sim > best:
                best, chosen = sim, (s, e)
        if chosen and best >= 0.5:
            return _wrap(final, *chosen), "window"
    return final, "ambiguous"


def _selftest() -> int:
    cases = [
        # (final, form, orig, ms, me, expect_status, expect_marked_contains)
        ("Civitas militum copias mittebat.", "Civitas", None, None, None,
         "unique", "<WSD>Civitas</WSD>"),
        # form twice; original neighbourhood is "inter ... inter Serenissimum"
        ("Quia vigentibus olim bellis inter Regem et bellis novis orta sunt mala.",
         "bellis", "olim bellis inter Regem", 5, 11, "window", None),
        ("Nullum bellum hic.", "bellis", None, None, None, "missing", None),
        ("text", "", None, None, None, "noform", None),
    ]
    ok = True
    for final, form, orig, ms, me, exp_status, contains in cases:
        marked, status = mark_target(final, form, orig, ms, me)
        good = status == exp_status and (contains is None or contains in marked)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] status={status} (want {exp_status}) :: {marked[:60]}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
