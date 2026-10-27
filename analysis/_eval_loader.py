"""
Shared loader/classifier/CI helpers for exp1_eval_analysis.ipynb.

Reads a promptfoo eval from the private evaluation store (the SQLite store
written during every eval) into a tidy DataFrame with derived columns:

  provider_label, lemma, quote_idx, granularity, def_language, show_metadata,
  gold_number, gold_sid, template, output, success, outcome,
  prompt_tokens, completion_tokens, reasoning_tokens, cost

Why SQLite, not JSONL: the DB is unredacted and carries the full per-row
JSON (test_case, response, grading_result, provider); JSONL is for archival.
Spec: analysis/exp1_eval_analysis_design.md §1.

The four canonical Exp 1 templates and their classifier (filename → label)
are documented in [`prompts/exp1/rendered/`](../prompts/exp1/rendered/).
Anything outside the known set is labelled `unknown_<basename>`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_ENV = os.environ.get("DIALEXPOL_DATA_DIR")
if not _DATA_ENV:
    raise RuntimeError("Set DIALEXPOL_DATA_DIR to the dialexpol-wsd-datasets checkout before importing _eval_loader")
_DATA_ROOT = Path(_DATA_ENV)
if not _DATA_ROOT.is_dir():
    raise RuntimeError(f"DIALEXPOL_DATA_DIR does not exist or is not a directory: {_DATA_ROOT}")
DEFAULT_DB = None
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DATA_ROOT
REGISTRY_YAML = None
QUOTE_FEATURES_CSV = _DATA_ROOT / "dictionary_quotations" / "quote_features.csv"
SPLIT_CSV = _DATA_ROOT / "dictionary_quotations" / "quotations.csv"
SIGLA_CSV = _DATA_ROOT / "neolatin_concordances" / "sigla.csv"

# ---------------------------------------------------------------------------
# Provider rate card — copy of run_exp1.py RATES; budget lineup, 2026-05-15.
# (input $/MTok, output $/MTok)
RATES: dict[str, tuple[float, float]] = {
    "claude-haiku-4.5":       (1.00,  5.00),
    "gemini-2.5-pro":         (1.25, 10.00),
    "llama-4-maverick":       (0.15,  0.60),
    "qwen3-235b-a22b-2507":   (0.071, 0.10),
    "gemma-4-31b-it":         (0.12,  0.37),
    "phi-4":                  (0.065, 0.14),
    "llama-3.1-8b-instruct":  (0.02,  0.05),
    "gemma-3-4b-it":          (0.04,  0.08),
    "llama-3.2-3b-instruct":  (0.051, 0.34),
    "llama-3.3-70b-instruct": (0.10,  0.32),
    "deepseek-chat-v3-0324":  (0.20,  0.77),
    "gpt-4o-mini":            (0.15,  0.60),
}

# ---------------------------------------------------------------------------
# Template classifier — basename of the rendered prompt file → canonical label.
# Plain set carries the four configs the CI-overlap rule operates on.
TEMPLATE_LABELS: dict[str, str] = {
    "exp1_zeroshot.txt":                 "zeroshot",
    "exp1_fewshot.txt":                  "fewshot",
    "exp1_justified.txt":                "justified",
    "exp1_zeroshot_decomposed.txt":      "decomposed",
    "exp1_zeroshot_json.txt":            "zeroshot_json",
    "exp1_fewshot_json.txt":             "fewshot_json",
    "exp1_justified_json.txt":           "justified_json",
    "exp1_zeroshot_decomposed_json.txt": "decomposed_json",
    "exp1_freeform.txt":                 "freeform",
    "exp1_fullcot.txt":                  "fullcot",
}

# Phase A lemma set — sourced from exp1_tests.py:PHASE_A. Duplicated here so
# the loader can infer the generator without importing scripts/ (the notebook
# wants to be self-contained when reading an eval from a sibling project clone).
PHASE_A_LEMMAS = {"LIBERTAS", "OFFICIUM", "AUCTORITAS", "DOMINUS", "IURISDICTIO",
                  "COMMUNITAS", "DOMINIUM", "COMES", "CONSILIUM", "HONOR", "CONVENTIO"}

# Axes the matrix was built over (PLAN.md §2.2). yaml_kind is a fourth axis
# (output-format effect) — template-derived rather than test_case.vars-derived,
# so it only varies when multiple evals are loaded. PLAN.md treats yaml_kind on
# the same footing as the others (one of the configuration dimensions Phase B
# promotion sweeps over).
AXES = ("granularity", "def_language", "show_metadata", "yaml_kind")
AXIS_VALUES = {
    "granularity":   ("l1", "l1l2"),
    "def_language":  ("la", "pl"),
    "show_metadata": (False, True),
    "yaml_kind":     ("plain", "json"),
}

_PROMPT_FILE_RE = re.compile(r"(exp1_[a-zA-Z0-9_]+\.txt)")


# ---------------------------------------------------------------------------
# classifiers

def classify_template(prompt_label: str | None) -> str:
    if not prompt_label:
        return "unknown_<none>"
    m = _PROMPT_FILE_RE.search(prompt_label)
    if not m:
        return f"unknown_{(prompt_label or '')[:30]}"
    fname = m.group(1)
    return TEMPLATE_LABELS.get(fname, f"unknown_{fname}")


def classify_outcome(success: bool, reason: str | None) -> str:
    """Three-way outcome per scripts/exp1_assert.py / PLAN.md §4.1.

    Note: promptfoo overwrites the per-assertion reason with the string
    "All assertions passed" when success is True, so we *must* key on
    success — the original "correct" reason is not preserved at the
    top-level grading_result.reason. For failing rows the assertion's
    reason survives, so the "format_failure" / "wrong" split works."""
    if success:
        return "correct"
    if reason and "format_failure" in reason.lower():
        return "format_failure"
    return "wrong"


def infer_stage(lemmas: set[str]) -> str:
    """Map a lemma set back to its generator (smoke / pilot / phase_a / other)."""
    if lemmas == {"AUCTORITAS"}:
        return "pilot or smoke (AUCTORITAS-only)"
    if lemmas == PHASE_A_LEMMAS:
        return "phase_a"
    if lemmas <= PHASE_A_LEMMAS:
        return f"phase_a_subset ({len(lemmas)}/{len(PHASE_A_LEMMAS)} lemmas)"
    overlap = lemmas & PHASE_A_LEMMAS
    if overlap:
        return f"mixed phase_a+other ({len(overlap)} A-lemmas + {len(lemmas - overlap)} others)"
    return "phase_b or other"


# ---------------------------------------------------------------------------
# cost

def row_cost(provider_label: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = RATES.get(provider_label)
    if rate is None:
        return float("nan")
    pin, pout = rate
    return ((prompt_tokens or 0) * pin + (completion_tokens or 0) * pout) / 1e6


# ---------------------------------------------------------------------------
# loader

def load_eval(eval_id: str, db_path: str | os.PathLike | None = None
              ) -> tuple[dict, pd.DataFrame]:
    """Load one eval into (meta_dict, dataframe). meta carries description +
    created_at; dataframe has the derived columns documented at module top."""
    p = Path(db_path) if db_path else DEFAULT_DB
    con = _private_backend_removed(str(p))
    try:
        meta_row = con.execute(
            "SELECT id, description, created_at FROM evals WHERE id=?",
            (eval_id,),
        ).fetchone()
        if meta_row is None:
            raise ValueError(f"eval_id {eval_id!r} not found in {p}")
        meta = {"id": meta_row[0], "description": meta_row[1],
                "created_at_ms": meta_row[2]}

        rows = con.execute("""
            SELECT
              json_extract(prompt,        '$.label'),
              json_extract(provider,      '$.label'),
              json_extract(test_case,     '$.vars.lemma'),
              json_extract(test_case,     '$.vars.granularity'),
              json_extract(test_case,     '$.vars.def_language'),
              json_extract(test_case,     '$.vars.show_metadata'),
              json_extract(test_case,     '$.vars.gold_number'),
              json_extract(test_case,     '$.vars.gold'),
              json_extract(test_case,     '$.vars.quotation_wsd'),
              json_extract(test_case,     '$.metadata.quote_idx'),
              json_extract(response,      '$.output'),
              success,
              json_extract(grading_result,'$.reason'),
              json_extract(response,      '$.tokenUsage.prompt'),
              json_extract(response,      '$.tokenUsage.completion'),
              json_extract(response,      '$.tokenUsage.completionDetails.reasoning')
            FROM eval_results
            WHERE eval_id=?
        """, (eval_id,)).fetchall()
    finally:
        con.close()

    cols = ["prompt_label", "provider_label", "lemma", "granularity",
            "def_language", "show_metadata", "gold_number", "gold_sid",
            "quotation_wsd", "quote_idx", "output", "success", "reason",
            "prompt_tokens", "completion_tokens", "reasoning_tokens"]
    df = pd.DataFrame(rows, columns=cols)

    df["template"] = df["prompt_label"].map(classify_template)
    df["yaml_kind"] = df["template"].map(lambda t: "json" if t.endswith("_json") else "plain")
    # task_type collapses plain/json sibling templates into the four canonical task
    # types (zeroshot / fewshot / justified / decomposed). Use this when the
    # yaml-kind effect is already isolated on its own axis.
    df["task_type"] = df["template"].map(lambda t: t[:-5] if t.endswith("_json") else t)
    df["success"] = df["success"].fillna(0).astype(int).astype(bool)
    df["outcome"] = [classify_outcome(s, r) for s, r in zip(df["success"], df["reason"])]
    # SQLite json_extract returns strings; coerce numeric columns so downstream
    # joins (e.g. quote_idx merge with the features CSV) work without silent
    # type-mismatch failures.
    df["quote_idx"] = pd.to_numeric(df["quote_idx"], errors="coerce")
    df["gold_number"] = pd.to_numeric(df["gold_number"], errors="coerce")
    df["show_metadata"] = df["show_metadata"].map(lambda v: bool(v) if v in (0, 1, True, False) else None)
    for c in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["cost"] = df.apply(lambda r: row_cost(r.provider_label,
                                             r.prompt_tokens, r.completion_tokens), axis=1)
    df["eval_id"] = eval_id
    return meta, df


def load_evals(*eval_ids: str, db_path: str | os.PathLike | None = None
               ) -> tuple[list[dict], pd.DataFrame]:
    """Load one or more evals into (list_of_metas, concatenated_df). yaml_kind
    is derived per-row from the template, so loading a plain eval + a JSON eval
    gives a DataFrame where yaml_kind has both values.

    The two evals may cover different subsets of the axis matrix (the JSON
    side typically runs the restricted matrix per the exp1_phase_a_restricted
    stage); the downstream analysis cells handle asymmetric coverage by
    intersecting axis cells where pairing is required (§4 yaml_kind axis test)
    and reporting marginals on each yaml's own coverage where it isn't."""
    metas, dfs = [], []
    for eid in eval_ids:
        m, d = load_eval(eid, db_path=db_path)
        metas.append(m); dfs.append(d)
    return metas, pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# confidence intervals

def wald_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation binomial CI (spec §2 formula). Returns (lo, hi)
    on [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def bootstrap_ci_binary(k: int, n: int, n_boot: int = 10_000,
                        alpha: float = 0.05, rng: np.random.Generator | None = None
                        ) -> tuple[float, float]:
    """Bootstrap CI for a binary outcome (spec §7 / PLAN.md §2.2: 10K resamples).

    For a Bernoulli vector with k ones, resampling-with-replacement gives
    counts ~ Binomial(n, k/n), so this is the exact non-parametric bootstrap
    in closed form — no per-sample loop needed. Vectorised; ~µs per (k, n)
    even at n_boot=10K."""
    if n == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    p_hat = k / n
    resampled = rng.binomial(n, p_hat, size=n_boot) / n
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# McNemar — paired discordance test (spec §3 / §4)

def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar (continuity-corrected) on the 2x2 discordant counts:
       b = #(paired-A correct, paired-B wrong); c = #(A wrong, B correct).
    Returns (chi2, p_value). For small b+c we use the exact binomial."""
    from scipy.stats import binomtest, chi2 as _chi2
    n_disc = b + c
    if n_disc == 0:
        return (0.0, 1.0)
    if n_disc < 25:
        # exact: under H0 of equal flips, b ~ Binomial(n_disc, 0.5)
        p = binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue
        return (float("nan"), float(p))
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    p = 1 - _chi2.cdf(chi2, df=1)
    return (float(chi2), float(p))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values in
    input order. (PLAN.md §2.2 — family-wise across template / axis tests.)"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, a)
        adj[idx] = running_max
    return adj


# ---------------------------------------------------------------------------
# Released run labels are stored in the prediction-table run column.
# Notebooks reference by key (e.g. 'phase_a_plain_restricted') instead of pasting raw IDs.

def _load_registry(path: str | os.PathLike = REGISTRY_YAML) -> dict:
    import yaml
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Loaded once on import. The source project may have a private registry, but a
# release checkout intentionally does not; public run names map to the
# prediction-table run labels in that case.
try:
    EVALS: dict = _load_registry()
except FileNotFoundError:
    EVALS = {name: {"id": name, "description": "released CSV prediction table"}
             for name in ("phase_a_plain_restricted", "phase_a_json_restricted",
                          "phase_b_plain", "exp2_core", "exp2_extended")}


def eval_id(name: str) -> str:
    """Resolve a registry name to an eval ID; raise if unset (id: null)."""
    entry = EVALS.get(name)
    if entry is None:
        raise KeyError(f"{name!r} not in {REGISTRY_YAML.name}; known: {sorted(EVALS)}")
    eid = entry.get("id")
    if not eid:
        raise ValueError(f"{name!r}.id is null in {REGISTRY_YAML.name} "
                         f"(description: {entry.get('description', '(none)')})")
    return eid


# ---------------------------------------------------------------------------
# Per-quote feature engineering — auto-detection with a reviewed-override
# convention. Features are intrinsic to the quote (not eval-specific) and
# live in dictionary_quotations/quote_features.csv, sibling to the split CSV.
# Manual overrides go in *_reviewed columns; loaders prefer reviewed when
# non-blank and fall back to *_auto.

POLISH_CHARS_RE = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
# Quoted-gloss families seen in the data: straight ASCII "...", Polish „...",
# and the occasional «...». A run inside any of these counts as a gloss.
QUOTED_GLOSS_RE = re.compile(r'"[^"]{1,80}"|„[^"]{1,80}"|«[^»]{1,80}»')
# Editorial scilicet notes — "(sc. clarification)" — added by the dictionary
# editor to disambiguate referents. Present in `quotation_cleaned`, visible
# to the LLM via `quotation_wsd`. Stripped from `quotation_plain`.
SC_NOTE_RE = re.compile(r"\(\s*sc\.", re.IGNORECASE)
BRACKET_RE = re.compile(r"\[[^\]]+\]")
DATE_YEAR_RE = re.compile(r"\b(1[0-7]\d{2})\b")   # a. 1234 → 1234

# Ibidem-pattern (Ib., ib., Ibid., Ibidem) — matched after asterisk-strip.
# Captures an optional trailing volume in Roman numerals: "Ib. II" → volume="II".
_IBIDEM_RE = re.compile(r"^[Ii]b(?:id|idem)?\.?(?:\s+([IVXLCDM]+))?$")
_TRAILING_ROMAN_RE = re.compile(r"\s+([IVXLCDM]+)$")


def clean_source(raw: str, prev_clean: str = "") -> str:
    """Normalise a source citation:
      - strip whitespace and leading asterisks (uncertainty marker in SLS)
      - resolve Ib. / ibid. against the previous row's clean source
      - for 'Ib. II' (volume substitution), keep the previous siglum but
        replace the trailing roman-numeral volume with the new one
    Returns the original (post-asterisk-strip) string if no resolution applies.
    """
    s = (raw or "").strip().lstrip("*").strip()
    m = _IBIDEM_RE.match(s)
    if not m:
        return s
    if not prev_clean:
        return s   # nothing to inherit from — leave as-is for reviewer
    new_volume = m.group(1)
    if not new_volume:
        return prev_clean   # plain Ib. → same as previous
    # Volume substitution: previous had trailing roman-numeral volume → replace it.
    if _TRAILING_ROMAN_RE.search(prev_clean):
        return _TRAILING_ROMAN_RE.sub(f" {new_volume}", prev_clean)
    return f"{prev_clean} {new_volume}"


def extract_sigla(source_clean: str) -> str:
    """Extract the bibliographic siglum from a cleaned source citation.
    The siglum is the first whitespace-delimited token after stripping
    surrounding parens and trailing punctuation."""
    s = (source_clean or "").strip().strip("()").strip()
    if not s:
        return ""
    first = s.split()[0].rstrip(".,;:")
    return first.strip("()")


_SIGLA_MAP_CACHE: dict[str, str] | None = None


def _sigla_to_genre_map(csv_path: str | os.PathLike = SIGLA_CSV) -> dict[str, str]:
    """Cached lookup: sigla → genre, sourced from data/source/sls_sigla.csv.
    Sigla with blank genre map to '' (analyst hasn't curated them yet)."""
    global _SIGLA_MAP_CACHE
    if _SIGLA_MAP_CACHE is None:
        try:
            df = pd.read_csv(csv_path)
            _SIGLA_MAP_CACHE = dict(zip(df["sigla"].astype(str),
                                        df["genre"].fillna("").astype(str)))
        except FileNotFoundError:
            _SIGLA_MAP_CACHE = {}
    return _SIGLA_MAP_CACHE


def guess_genre(source_clean: str) -> str:
    """Map a cleaned source citation to a broad genre via the sigla CSV.

    Lookup order — most-specific wins:
      1. full `source_clean`           (e.g. 'StPPP III'      → per-volume row)
      2. source_clean minus trailing roman volume
                                       (e.g. 'DŁUG. LibBen I' → 'DŁUG. LibBen')
      3. extracted bare sigla          (e.g. 'StPPP'          → base row)

    This lets the sigla CSV carry mappings at any granularity:
      - per-volume row (`StPPP III`) for sigla whose content varies by volume
      - per-work row (`DŁUG. LibBen`) for one sub-work of a multi-work author
      - base-sigla row (`DŁUG`) for uniform multi-volume works

    Returns 'unknown' when no key matches (or all matches have blank genre —
    curator hasn't filled them in yet)."""
    if not source_clean:
        return "unknown"
    smap = _sigla_to_genre_map()
    keys = [source_clean.strip().strip("()")]
    # Strip a trailing roman-numeral volume if present.
    without_vol = _TRAILING_ROMAN_RE.sub("", keys[0]).strip()
    if without_vol and without_vol != keys[0]:
        keys.append(without_vol)
    keys.append(extract_sigla(source_clean))
    for k in keys:
        if k and smap.get(k):
            return smap[k]
    return "unknown"


def _str_or_blank(v) -> str:
    """Coerce pandas-NaN / None / numeric to a clean string; pandas reads blank
    CSV cells as float('nan'), which breaks downstream regex operations."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def _century_from_date(date_str) -> int | None:
    """Extract a century (e.g. 1358 → 14) from an SLS date string like 'a. 1358'."""
    s = _str_or_blank(date_str)
    if not s:
        return None
    m = DATE_YEAR_RE.search(s)
    if not m:
        return None
    return int(m.group(1)) // 100 + 1


def _detect_features(row: dict, source_clean: str) -> dict:
    """Per-row auto-detected features. Used by build_quote_features().
    source_clean is computed in a separate pass (needs lemma-ordered
    iteration for Ib. resolution) and passed in here."""
    quote = _str_or_blank(row.get("quotation_cleaned"))
    has_polish_chars = bool(POLISH_CHARS_RE.search(quote))
    has_quoted_gloss = bool(QUOTED_GLOSS_RE.search(quote))
    has_sc_note = bool(SC_NOTE_RE.search(quote))
    n_brackets = len(BRACKET_RE.findall(quote))
    n_words_raw = row.get("n_words")
    try:
        n_words = int(n_words_raw) if n_words_raw and not (isinstance(n_words_raw, float) and np.isnan(n_words_raw)) else len(quote.split())
    except (TypeError, ValueError):
        n_words = len(quote.split())
    return {
        "source_clean": source_clean,
        "sigla": extract_sigla(source_clean),
        "has_polish_gloss_auto": has_polish_chars or has_quoted_gloss,
        "has_sc_note_auto": has_sc_note,
        "n_bracket_expansions": n_brackets,
        "n_words": n_words,
        "length_bin": _str_or_blank(row.get("length_bin")),
        "century_auto": _century_from_date(row.get("date")),
        "genre_auto": guess_genre(source_clean),
    }


_REVIEWABLE_COLS = ("has_polish_gloss", "has_sc_note", "genre", "century")


def build_quote_features(split_csv: str | os.PathLike = SPLIT_CSV,
                         out_csv: str | os.PathLike = QUOTE_FEATURES_CSV,
                         overwrite_reviewed: bool = False) -> pd.DataFrame:
    """Compute per-quote features from the split CSV and write to out_csv.

    If out_csv already exists, the *_reviewed columns are preserved by
    default (overwrite_reviewed=False) — only the *_auto columns get
    rewritten. A drift report is printed when the new auto value disagrees
    with a previously-reviewed value, so the reviewer knows which rows to
    re-check after a code change.

    Returns the written DataFrame (auto + reviewed columns merged).
    """
    all_rows = pd.read_csv(split_csv)
    # Resolve `Ib.` in the full row set (so demo rows can be the antecedent
    # for a test row's Ib.) before filtering. Iterate in CSV order within
    # each lemma group — the CSV preserves dictionary order per lemma.
    source_clean_per_row: list[str] = []
    prev_clean_per_lemma: dict[str, str] = {}
    for _, r in all_rows.iterrows():
        lemma = r.get("lemma", "")
        raw = _str_or_blank(r.get("source"))
        cleaned = clean_source(raw, prev_clean_per_lemma.get(lemma, ""))
        source_clean_per_row.append(cleaned)
        # Update the per-lemma antecedent so chains of consecutive Ib. all
        # resolve to the same source (and volume-substituted Ib. updates it
        # to the new volume).
        if cleaned:
            prev_clean_per_lemma[lemma] = cleaned
    all_rows = all_rows.copy()
    all_rows["_source_clean"] = source_clean_per_row

    rows = all_rows[all_rows.get("split").fillna("") != "demo"].reset_index(drop=True)
    feats = pd.DataFrame([_detect_features(r.to_dict(), r["_source_clean"])
                          for _, r in rows.iterrows()])
    # quote_text shown to the analyst: prefer quotation_plain (brackets resolved,
    # sc notes stripped — closer to what the LLM saw, minus the <WSD> tags) and
    # fall back to quotation_cleaned. Matches exp1_split.ipynb's convention.
    quote_text = (rows.get("quotation_plain")
                  .fillna(rows.get("quotation_cleaned"))
                  .fillna("")
                  .astype(str).str.slice(0, 80)
                  .rename("quote_text"))
    out = pd.concat([
        rows[["quote_idx", "lemma", "source", "date"]].reset_index(drop=True),
        quote_text.reset_index(drop=True),
        feats.reset_index(drop=True),
    ], axis=1)
    # Initialise blank *_reviewed + notes columns; preserve existing edits.
    for col in _REVIEWABLE_COLS:
        out[f"{col}_reviewed"] = ""
    out["notes"] = ""

    out_path = Path(out_csv)
    if out_path.exists() and not overwrite_reviewed:
        existing = pd.read_csv(out_path)
        # Keyed on (lemma, quote_idx) — quote_idx alone is not unique across
        # lemmas (the dictionary recycles indices per entry). Missing columns
        # in the existing CSV (e.g. when schema grew) are silently skipped.
        key = ["lemma", "quote_idx"]
        if all(k in existing.columns for k in key):
            existing_keyed = existing.drop_duplicates(subset=key).set_index(key)
            for col in (*[f"{c}_reviewed" for c in _REVIEWABLE_COLS], "notes"):
                if col in existing_keyed.columns:
                    out[col] = out.set_index(key).index.map(existing_keyed[col]).fillna("")
            # Drift report: where reviewed value disagrees with new auto value.
            for col in _REVIEWABLE_COLS:
                rev_col = f"{col}_reviewed"
                if rev_col not in out.columns:
                    continue
                rev = out[rev_col].astype(str).str.strip()
                auto = out[f"{col}_auto"].astype(str)
                disagreements = (rev != "") & (rev != auto)
                if disagreements.any():
                    print(f"NOTE: {disagreements.sum()} reviewed rows now disagree with new {col}_auto "
                          f"(reviewer should re-check these): "
                          f"{out.loc[disagreements, 'quote_idx'].head(5).tolist()}…")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def load_quote_features(csv_path: str | os.PathLike = QUOTE_FEATURES_CSV
                        ) -> pd.DataFrame:
    """Load the per-quote features, preferring *_reviewed values when set.
    Returned DataFrame has the resolved columns under their base names
    (e.g. 'has_polish_gloss' — reviewed if non-blank else _auto)."""
    df = pd.read_csv(csv_path)
    for col in _REVIEWABLE_COLS:
        auto_col, rev_col = f"{col}_auto", f"{col}_reviewed"
        if auto_col not in df.columns:
            continue
        if rev_col in df.columns:
            # NaN-safe: pandas reads blank CSV cells as float NaN; fillna
            # before strip so blanks become empty strings, not 'nan'.
            rev = df[rev_col].fillna("").astype(str).str.strip()
        else:
            rev = pd.Series([""] * len(df))
        df[col] = rev.where(rev != "", df[auto_col])
    return df


# ---------------------------------------------------------------------------
# Phase B helpers — MFS baseline, κ vs gold, per-quote outcome matrix,
# quote drill-down. Used by exp1_phase_b_analysis.ipynb; usable from any
# analysis notebook.

def mfs_baseline(split_csv: str | os.PathLike = SPLIT_CSV,
                 granularity: str = "l1") -> tuple[dict[str, float], float]:
    """Most-Frequent-Sense baseline per lemma. Returns (per-lemma acc, macro mean).

    Per-lemma MFS acc = max_count(gold_sense) / total_test_quotes for the
    given granularity. The macro mean averages across lemmas (each lemma
    contributes equally regardless of quote count), matching PLAN.md §2.5
    'each lemma's MFS' framing. Uses test-split rows only."""
    df = pd.read_csv(split_csv)
    df = df[df.get("split").fillna("") == "test"]
    gold_col = f"gold_{granularity}"
    per_lemma: dict[str, float] = {}
    for lemma, g in df.groupby("lemma"):
        gold = g[gold_col].dropna()
        if len(gold) == 0:
            continue
        per_lemma[lemma] = gold.value_counts().iloc[0] / len(gold)
    macro = sum(per_lemma.values()) / len(per_lemma) if per_lemma else float("nan")
    return per_lemma, macro


def kappa_vs_gold(predicted, gold) -> float:
    """Cohen's kappa between predicted and gold sense numbers, NaN-safe.
    Both inputs are iterables of integers (or strings parseable as ints).
    Rows where either side is NaN are dropped before the computation."""
    from sklearn.metrics import cohen_kappa_score
    p = pd.to_numeric(pd.Series(predicted), errors="coerce")
    g = pd.to_numeric(pd.Series(gold), errors="coerce")
    mask = p.notna() & g.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(cohen_kappa_score(g[mask].astype(int), p[mask].astype(int)))


# ---------------------------------------------------------------------------
# Per-quote outcome matrix — wide DataFrame with one row per quote and one
# column per (provider, config) cell carrying that cell's predicted_number
# and outcome. Plus consensus / difficulty derived columns. Used by
# §5a / §5b / §5c of the Phase B notebook for quote-level error analysis.

_PREDICTED_RE = re.compile(r"^\s*(\d+)\s*$")


def parse_predicted_number(output: str | None) -> int | None:
    """Extract the predicted sense number from a model output string.
    Mirrors exp1_assert.py logic in a simplified form (we just need the
    number; the assertion outcome is already in the success column)."""
    if not output:
        return None
    s = str(output).strip()
    m = _PREDICTED_RE.match(s)
    if m:
        return int(m.group(1))
    # JSON sense_number field
    m = re.search(r'"sense_number"\s*:\s*"?(\d+)"?', s)
    if m:
        return int(m.group(1))
    # "SENSE: N" line
    m = re.search(r"SENSE\s*:\s*\"?(\d+)\"?", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # First standalone integer anywhere
    m = re.search(r"\b\d+\b", s)
    if m:
        return int(m.group(0))
    return None


def per_quote_outcome_matrix(df: pd.DataFrame,
                             features_df: pd.DataFrame | None = None
                             ) -> pd.DataFrame:
    """Wide table indexed by (lemma, quote_idx). Columns per (provider × config)
    cell: predicted, outcome. Plus consensus signals (n_correct, n_wrong,
    n_ff, difficulty) and the merged quote-level features when provided.

    Used as the workhorse for §5a per-quote drill-down. Joined to the
    features DataFrame if given (typically `load_quote_features()`)."""
    work = df.copy()
    work["lemma"] = work["lemma"].astype(str).str.strip().str.upper()
    work["quote_idx"] = pd.to_numeric(work["quote_idx"], errors="coerce")
    work["predicted"] = work["output"].map(parse_predicted_number)
    # Per-quote consensus signals — across the entire df (all providers, all
    # configs). The analyst can re-compute on a filtered subset if they want
    # a per-config view.
    by_quote = work.groupby(["lemma", "quote_idx"]).agg(
        n_cells=("success", "size"),
        n_correct=("success", "sum"),
        gold_number=("gold_number", "first"),
        gold_sid=("gold_sid", "first"),
    )
    by_quote["n_ff"] = work[work.outcome == "format_failure"].groupby(
        ["lemma", "quote_idx"]).size().reindex(by_quote.index, fill_value=0)
    by_quote["n_wrong"] = by_quote["n_cells"] - by_quote["n_correct"] - by_quote["n_ff"]
    by_quote["difficulty"] = 1.0 - by_quote["n_correct"] / by_quote["n_cells"]

    if features_df is not None:
        # Merge per-quote features (Polish gloss, genre, century, ...) on
        # (lemma, quote_idx). Drop redundant identity columns first.
        feat = features_df.copy()
        feat["lemma"] = feat["lemma"].astype(str).str.strip().str.upper()
        feat["quote_idx"] = pd.to_numeric(feat["quote_idx"], errors="coerce")
        feat = feat.set_index(["lemma", "quote_idx"])
        keep = [c for c in feat.columns
                if c not in ("source", "date", "quote_text", "n_words", "length_bin")]
        # Keep the most useful for analysis even if redundant with eval-side cols
        keep = list(set(keep + ["quote_text", "n_words", "length_bin",
                                "source_clean", "sigla", "genre",
                                "has_polish_gloss", "century"]))
        by_quote = by_quote.join(feat[keep], how="left")
        if len(by_quote) and by_quote["genre"].notna().mean() < 0.95:
            raise RuntimeError("quote-feature join matched fewer than 95% of evaluation quotes; check lemma/quote_idx normalization")
    return by_quote


def inspect_quote(df: pd.DataFrame, lemma: str, quote_idx,
                  config: str | None = None,
                  features_df: pd.DataFrame | None = None) -> None:
    """Pretty-print everything about one quote across all (provider, config)
    cells. Use from a notebook for error drill-down (§5c)."""
    qkey = (df["lemma"] == lemma) & (df["quote_idx"] == quote_idx)
    sub = df[qkey].copy()
    if config:
        sub = sub[sub["config"] == config]
    if sub.empty:
        print(f"no rows for {lemma} q{quote_idx}"
              + (f" config={config!r}" if config else ""))
        return
    head = sub.iloc[0]
    print(f"=== {lemma} quote_idx={quote_idx} ===")
    print(f"  gold sense  : #{head.get('gold_number')} ({head.get('gold_sid')})")
    # quotation_wsd is what the LLM actually saw (brackets resolved, target
    # word wrapped in <WSD>...</WSD>; sc notes still present per transform.py
    # mark_wsd convention).
    if head.get("quotation_wsd"):
        print(f"  LLM input   : {head['quotation_wsd']}")
    if features_df is not None:
        feat_row = features_df[(features_df.lemma == lemma)
                               & (features_df.quote_idx == quote_idx)]
        if not feat_row.empty:
            fr = feat_row.iloc[0]
            print(f"  source      : {fr.get('source_clean')!r}  ({fr.get('genre')})")
            print(f"  length / cy : {fr.get('length_bin')} ({fr.get('n_words')} words) / "
                  f"{fr.get('century')}c.")
            print(f"  pl gloss?   : {fr.get('has_polish_gloss')}")
            print(f"  quote_text  : {(fr.get('quote_text') or '')[:160]}…")
    # Group outputs by (template, yaml_kind, axes) → provider
    sub["predicted"] = sub["output"].map(parse_predicted_number)
    print(f"\n  {'provider':<25} {'cfg':<55} {'pred':>4}  {'outcome'}")
    for _, r in sub.sort_values(["provider_label", "config" if "config" in sub.columns else "template"]).iterrows():
        cfg = r.get("config", r.get("template", "?"))
        print(f"  {r['provider_label']:<25} {str(cfg)[:55]:<55} "
              f"{str(r['predicted']):>4}  {r['outcome']}")


# ---------------------------------------------------------------------------
# §5 drill-down helpers — filter the per-quote outcome matrix by any
# combination of covariates / difficulty thresholds, then pretty-print
# the filtered cohort via inspect_quote.

def quotes_where(per_q: pd.DataFrame, lemma: str | None = None,
                 difficulty_gte: float | None = None,
                 difficulty_lte: float | None = None,
                 n_correct_eq: int | None = None,
                 **equals) -> pd.DataFrame:
    """Filter the per-quote outcome matrix by any combination of column-value
    equals and difficulty thresholds. Returns a subset with the same
    (lemma, quote_idx) MultiIndex, ready for inspect_quotes().

    Examples:
        quotes_where(per_q, has_sc_note=True, difficulty_gte=0.7)
        quotes_where(per_q, lemma='BENEFICIUM', has_polish_gloss=True)
        quotes_where(per_q, n_correct_eq=0)   # consensus failures
        quotes_where(per_q, genre='correspondence', difficulty_gte=0.5)
    """
    sub = per_q
    if lemma is not None:
        sub = sub[sub.index.get_level_values("lemma") == lemma]
    if difficulty_gte is not None:
        sub = sub[sub["difficulty"] >= difficulty_gte]
    if difficulty_lte is not None:
        sub = sub[sub["difficulty"] <= difficulty_lte]
    if n_correct_eq is not None:
        sub = sub[sub["n_correct"] == n_correct_eq]
    for col, val in equals.items():
        if col not in sub.columns:
            continue
        sub = sub[sub[col] == val]
    return sub


def inspect_quotes(per_q_subset: pd.DataFrame, df: pd.DataFrame,
                   features_df: pd.DataFrame | None = None,
                   max_n: int = 5) -> None:
    """Pretty-print up to max_n quotes from a per-quote subset via inspect_quote.
    Use after quotes_where() to walk through a filtered cohort."""
    n = len(per_q_subset)
    for i, ((lemma, qidx), _) in enumerate(per_q_subset.iterrows()):
        if i >= max_n:
            print(f"\n… and {n - max_n} more (raise max_n to see them)")
            return
        inspect_quote(df, lemma, qidx, features_df=features_df)
        print()


# ---------------------------------------------------------------------------
# Lemma-level features — semantic-complexity proxies that join per_q on lemma.
# n_candidate_senses is the number of senses the model was actually shown
# (after the transform.py def-presence filter). quotes_per_sense is the
# coverage ratio — low = polysemous but sparsely evidenced (harder).

_L1_INVENTORY_CSV = _DATA_ROOT / "senses" / "senses_l1.csv"

_INVENTORY_CACHE: dict[str, pd.DataFrame] = {}


def _load_inventory(inventory_csv: str | os.PathLike = _L1_INVENTORY_CSV
                    ) -> pd.DataFrame:
    """Cached read of the senses inventory (sid → def_la/def_pl/label)."""
    key = str(inventory_csv)
    if key not in _INVENTORY_CACHE:
        _INVENTORY_CACHE[key] = pd.read_csv(inventory_csv)
    return _INVENTORY_CACHE[key]


def compute_lemma_features(df_eval: pd.DataFrame,
                           inventory_csv: str | os.PathLike = _L1_INVENTORY_CSV
                           ) -> pd.DataFrame:
    """Per-lemma covariates derived from inventory + eval. Returns DataFrame
    indexed by lemma with columns:
      n_candidate_senses  — senses presented to the model (def_la OR def_pl)
      n_quotes_per_lemma  — distinct quote_idx in the eval
      quotes_per_sense    — n_quotes_per_lemma / n_candidate_senses
    """
    inv = pd.read_csv(inventory_csv)
    # Count only senses that would be presented — at least one def
    has_def = inv["def_la"].fillna("").str.strip().ne("") | \
              inv["def_pl"].fillna("").str.strip().ne("")
    n_senses = inv[has_def].groupby("lemma").size().rename("n_candidate_senses")
    n_quotes = df_eval.groupby("lemma")["quote_idx"].nunique().rename("n_quotes_per_lemma")
    out = pd.concat([n_senses, n_quotes], axis=1).dropna(subset=["n_quotes_per_lemma"])
    out["quotes_per_sense"] = (out["n_quotes_per_lemma"] / out["n_candidate_senses"]).round(2)
    return out


# ---------------------------------------------------------------------------
# Tabular drill-down — replaces print-based inspect_quote for browsing.
# Pandas DataFrames render as scrollable HTML tables in Jupyter; click a
# column header to sort. Use quote_details() for the per-(provider,config)
# breakdown of one quote; use quote_picker() for a dropdown-driven UI.

_TABLE_DEFAULT_COLS = [
    "quote_text", "gold_number", "difficulty", "n_correct", "n_cells",
    "has_sc_note", "has_polish_gloss", "genre", "length_bin",
    "n_words", "quotes_per_sense_bin", "n_senses_bin", "century",
]


def quotes_table(per_q: pd.DataFrame, top_n: int = 50,
                 sort_by: str = "difficulty",
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Return a clean per-quote DataFrame for browsing in Jupyter. By default
    sorts by difficulty descending and shows the top 50. Override columns
    or sort_by to focus on a different cut."""
    cols = [c for c in (columns or _TABLE_DEFAULT_COLS) if c in per_q.columns]
    sub = per_q[cols].copy()
    if sort_by and sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)
    return sub.head(top_n)


# ---------------------------------------------------------------------------
# Sense-gloss reconstruction for the Exp-1 drill-down.
# The numbered candidate list shown to the model is shuffled PER QUOTE (seed =
# quote_idx, set by exp1_tests.py and consumed by transform.numbered_senses),
# so the same predicted *number* means a different sense for each quote. To
# read a drill-down you therefore have to map number -> (sid, label, gloss) for
# that specific quote. Verified: reconstructing with seed=float(quote_idx)
# reproduces every gold_number -> gold_sid pair in phase_b_plain (445/445). The
# number->sid map is def_language-independent (la/pl share the shuffle); only
# the gloss TEXT differs, so we carry both languages.

def _clean_gloss(v) -> str:
    """A definition cell -> a clean string ('' for NaN/None)."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _truncate(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def exp1_candidate_senses(lemma: str, quote_idx, granularity: str = "l1",
                          gold_sid: str | None = None) -> pd.DataFrame:
    """Reproduce the numbered candidate-sense menu the model saw for one Exp-1
    quote. Returns a DataFrame indexed by the presented `number` with columns
    sid, label, def_la, def_pl, is_gold. Numbering uses the exact generation
    seed (float quote_idx); labels/glosses are joined from the L1 inventory.
    This is the legend that decodes the `predicted` numbers in quote_details."""
    import sys
    # ``REPO_ROOT`` is the fetched data root in the public backend; the
    # released transform module lives beside this analysis package.
    scripts = str(Path.cwd().parent / "pipeline")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from transform import numbered_senses

    numbered = numbered_senses(str(lemma).upper(), granularity, "la",
                               sense_order_seed=float(quote_idx))
    inv = _load_inventory()
    inv_by_sid = inv.set_index("sid")[["label", "def_la", "def_pl"]].to_dict("index")
    rows = []
    for n, sid, _def in numbered:
        meta = inv_by_sid.get(sid, {})
        rows.append({
            "number": n, "sid": sid,
            "label": _clean_gloss(meta.get("label")),
            "def_la": _clean_gloss(meta.get("def_la")),
            "def_pl": _clean_gloss(meta.get("def_pl")),
            "is_gold": sid == gold_sid,
        })
    out = pd.DataFrame(rows, columns=["number", "sid", "label",
                                      "def_la", "def_pl", "is_gold"])
    return out.set_index("number") if not out.empty else out


def exp1_prediction_breakdown(preds: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Tabulate one quote's per-(provider, config) predictions BY predicted
    sense: one row per distinct predicted number, carrying the sense glosses
    (def_la/def_pl), a gold flag (★), the total count and a per-provider count
    breakdown — so it is obvious which sense the gold is being confused with,
    and which models do it. `preds` is the frame returned by quote_details (cols
    provider, config, predicted); `cand` is exp1_candidate_senses(...). Sorted
    by total count."""
    valid = preds.dropna(subset=["predicted"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["predicted"] = valid["predicted"].astype(int)
    piv = (valid.pivot_table(index="predicted", columns="provider",
                             values="config", aggfunc="count", fill_value=0)
                .astype(int))

    def _info(num: int, field: str):
        if num in cand.index:
            return cand.loc[num, field]
        return None

    out = pd.DataFrame(index=piv.index)
    out["gold"] = ["★" if bool(_info(n, "is_gold")) else "" for n in piv.index]
    out["sid"] = [_info(n, "sid") or f"<#{n} not in menu>" for n in piv.index]
    out["label"] = [_clean_gloss(_info(n, "label")) for n in piv.index]
    out["def_la"] = [_truncate(_clean_gloss(_info(n, "def_la"))) for n in piv.index]
    out["def_pl"] = [_truncate(_clean_gloss(_info(n, "def_pl"))) for n in piv.index]
    out["n_total"] = piv.sum(axis=1).astype(int)
    out = pd.concat([out, piv], axis=1).sort_values("n_total", ascending=False)
    out.index.name = "predicted"
    return out


def exp1_top_confusion(preds: pd.DataFrame, cand: pd.DataFrame,
                       top: int = 3) -> pd.DataFrame:
    """The WRONG senses most often predicted for this quote, ranked by votes —
    i.e. what the gold sense is most confused with. Excludes the gold sense and
    out-of-menu predictions (e.g. 0/novelty), though the latter still count in
    the wrong-pick denominator. Returns number, sid, label, def_la, def_pl,
    n_votes, pct_of_all, pct_of_wrong (empty if every prediction hit gold).
    `preds` is from quote_details, `cand` from exp1_candidate_senses."""
    valid = preds.dropna(subset=["predicted"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["predicted"] = valid["predicted"].astype(int)
    n_all = len(valid)
    gold_nums = {n for n in cand.index if bool(cand.loc[n, "is_gold"])}
    counts = valid["predicted"].value_counts()
    n_wrong = int(counts[[k for k in counts.index if k not in gold_nums]].sum())
    rows = []
    for num, n in counts.items():
        if num in gold_nums or num not in cand.index:   # skip gold + out-of-menu
            continue
        rows.append({
            "number": int(num), "sid": cand.loc[num, "sid"],
            "label": _clean_gloss(cand.loc[num, "label"]),
            "def_la": _truncate(_clean_gloss(cand.loc[num, "def_la"])),
            "def_pl": _truncate(_clean_gloss(cand.loc[num, "def_pl"])),
            "n_votes": int(n),
            "pct_of_all": round(100 * n / n_all, 1),
            "pct_of_wrong": round(100 * n / n_wrong, 1) if n_wrong else float("nan"),
        })
    cols = ["number", "sid", "label", "def_la", "def_pl",
            "n_votes", "pct_of_all", "pct_of_wrong"]
    if not rows:                                  # every prediction hit gold
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows, columns=cols).sort_values("n_votes", ascending=False)
                                            .head(top).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Cross-quote analysis — do the best models AGREE on hard quotes, and is any
# shared wrong pick explained by PRESENTATION POSITION? Both rest on the fact
# that within a quote every model/config saw the same shuffled numbering
# (seed = quote_idx, def-independent), so `predicted` IS the presentation
# position and within-quote vote comparison is valid; across quotes the shuffle
# decouples position from sense, which identifies position bias. Cross-quote
# label comparison (κ) needs sense ids, so use predicted_sid, not the numbers.

def exp1_best_providers(df: pd.DataFrame, k: int = 4) -> list[str]:
    """The k providers with the highest overall accuracy (pooled over configs)."""
    acc = df.groupby("provider_label")["success"].mean().sort_values(ascending=False)
    return list(acc.head(k).index)


def exp1_attach_predicted_sid(df: pd.DataFrame, granularity: str = "l1"
                              ) -> pd.DataFrame:
    out = df.copy()
    if "predicted" not in out.columns:
        out["predicted"] = out["predicted_number"]
    if "predicted_sid" not in out.columns:
        raise ValueError("released predictions must contain predicted_sid")
    out["predicted_sid"] = out["predicted_sid"].replace("", pd.NA)
    return out


def exp1_modal_votes(df: pd.DataFrame, providers: list[str] | None = None,
                     value: str = "predicted") -> pd.DataFrame:
    """Collapse each provider to its modal vote per quote (over that provider's
    configs), then pivot to a wide frame: index (lemma, quote_idx), columns =
    providers, value = the modal `value` (ties → most frequent, then first seen).
    NaN where a provider produced no parseable prediction. `value` is usually
    'predicted' (numbers, for within-quote agreement/position) or 'predicted_sid'
    (for cross-quote κ)."""
    work = df.copy()
    if value == "predicted" and "predicted" not in work.columns:
        work["predicted"] = work["output"].map(parse_predicted_number)
    if providers is not None:
        work = work[work["provider_label"].isin(providers)]
    work = work.dropna(subset=[value])

    def _mode(s):
        vc = s.value_counts()
        return vc.index[0] if len(vc) else np.nan

    g = (work.groupby(["lemma", "quote_idx", "provider_label"])[value]
             .agg(_mode).reset_index())
    wide = g.pivot(index=["lemma", "quote_idx"], columns="provider_label",
                   values=value)
    if providers is not None:
        wide = wide.reindex(columns=[p for p in providers if p in wide.columns])
    wide.columns.name = None
    return wide


def exp1_agreement_metrics(votes_wide: pd.DataFrame,
                           candidates_n: dict | None = None) -> pd.DataFrame:
    """Per-quote agreement across the provider modal votes in `votes_wide`
    (columns = providers, values = sense numbers). Returns, indexed like
    votes_wide: n_voters, modal, modal_count, modal_share (= modal_count /
    n_voters), n_distinct, entropy_norm (Shannon entropy / log(#candidates) if
    `candidates_n` maps (lemma,quote_idx)→N, else / log(n_voters))."""
    recs = []
    for idx, row in votes_wide.iterrows():
        vals = row.dropna().tolist()
        n = len(vals)
        if n == 0:
            recs.append(dict(n_voters=0, modal=np.nan, modal_count=0,
                             modal_share=np.nan, n_distinct=0,
                             entropy_norm=np.nan))
            continue
        vc = pd.Series(vals).value_counts()
        p = (vc / n).to_numpy()
        ent = float(-(p * np.log(p)).sum())
        ncand = candidates_n.get(idx) if isinstance(candidates_n, dict) else None
        denom = np.log(ncand) if (ncand and ncand > 1) else (np.log(n) if n > 1 else 1.0)
        recs.append(dict(n_voters=n, modal=vc.index[0], modal_count=int(vc.iloc[0]),
                         modal_share=float(vc.iloc[0] / n), n_distinct=int(vc.size),
                         entropy_norm=float(ent / denom) if denom else 0.0))
    return pd.DataFrame(recs, index=votes_wide.index)


# ---------------------------------------------------------------------------
# Dictionary-hierarchy distance between a predicted and the gold sense — a
# graded "how wrong" measure: same major sense (I.1 vs I.2) vs a jump to a
# different major sense (I vs II). The hierarchy is read from the LABEL path
# (e.g. "II.A.2.a" → ('II','A','2','a')), NOT the sid integers: the sids are a
# renumbered internal id (e.g. bellum.2 and bellum.3 are both labelled "II", and
# the sid integers of II.A.2.a don't nest under II.A.2), whereas the dotted label
# IS the dictionary's authoritative nesting.

def exp1_label_map(inventory_csv: str | os.PathLike = _L1_INVENTORY_CSV) -> dict:
    """sid → label ('II.A.2.a', …) from the L1 inventory, for hierarchy lookups."""
    inv = _load_inventory(inventory_csv)
    return dict(zip(inv["sid"], inv["label"]))


def sid_hierarchy_relation(gold_sid, pred_sid, sid2label: dict) -> dict:
    """Hierarchy relationship between two senses by their LABEL paths. Returns
    relation (one of: same-label / parent-child / sibling / same-major-distant /
    different-major / unknown), lca_depth (shared leading label components),
    tree_dist (edges between the two label nodes), same_major (share the top —
    Roman-numeral — component). `sid2label` from exp1_label_map()."""
    def _path(sid):
        lab = sid2label.get(sid)
        return tuple(str(lab).split(".")) if lab and not pd.isna(lab) else ()
    g, p = _path(gold_sid), _path(pred_sid)
    if not g or not p:
        return dict(relation="unknown", lca_depth=float("nan"),
                    tree_dist=float("nan"), same_major=float("nan"))
    lca = 0
    for a, b in zip(g, p):
        if a == b:
            lca += 1
        else:
            break
    same_major = g[0] == p[0]
    tree_dist = (len(g) - lca) + (len(p) - lca)
    if g == p:
        rel = "same-label"                       # distinct sids, identical label
    elif p[:len(g)] == g or g[:len(p)] == p:
        rel = "parent-child"                     # coarser/finer of one lineage
    elif not same_major:
        rel = "different-major"                  # I vs II — the gross error
    elif lca == len(g) - 1 and len(g) == len(p):
        rel = "sibling"                          # same parent, e.g. I.1 vs I.2
    else:
        rel = "same-major-distant"               # share Roman numeral, not parent
    return dict(relation=rel, lca_depth=lca, tree_dist=tree_dist,
                same_major=bool(same_major))


def plot_quote_confusion(df: pd.DataFrame, lemma: str, quote_idx,
                         by: str = "provider", gold_sid: str | None = None,
                         granularity: str = "l1", ax=None):
    """Heatmap of ONE quote's votes: rows = `by` ('provider' | 'task_type'),
    columns = the numbered candidate senses the model saw, cell = vote count.
    The gold sense's column is outlined in red and its tick marked ★. (For a
    single occurrence only one gold class exists, so this is a vote-distribution
    "confusion" view — it shows which senses the gold is mistaken for and by
    whom.) Out-of-menu predictions, e.g. 0, are kept as trailing columns.
    Draws on the current/`ax` axes; returns the Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sub = df[(df["lemma"] == lemma) & (df["quote_idx"] == float(quote_idx))].copy()
    if sub.empty:
        print(f"no rows for {lemma} q{quote_idx}")
        return None
    sub["predicted"] = sub["output"].map(parse_predicted_number)
    row_col = ("provider_label" if by == "provider"
               else ("task_type" if "task_type" in sub.columns else "template"))
    cand = exp1_candidate_senses(lemma, quote_idx, granularity=granularity,
                                 gold_sid=gold_sid or sub["gold_sid"].iloc[0])
    sense_nums = list(cand.index)

    piv = (sub.dropna(subset=["predicted"])
              .assign(predicted=lambda d: d["predicted"].astype(int))
              .pivot_table(index=row_col, columns="predicted",
                           aggfunc="size", fill_value=0))
    for n in sense_nums:                       # show every candidate, even 0-vote
        if n not in piv.columns:
            piv[n] = 0
    extra = sorted(c for c in piv.columns if c not in sense_nums)  # e.g. 0
    piv = piv[sense_nums + extra]

    gold_nums = [n for n in sense_nums if bool(cand.loc[n, "is_gold"])]
    gold_n = gold_nums[0] if gold_nums else None

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.55 * len(piv.columns) + 3),
                                      max(2.5, 0.36 * len(piv) + 1)))
    data = piv.values
    vmax = data.max() if data.size else 1
    ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=max(1, vmax))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c} ★" if c == gold_n else str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(list(piv.index), fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > vmax / 2 else "black")
    if gold_n is not None:                     # outline the gold column
        j = list(piv.columns).index(gold_n)
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(piv.index),
                               fill=False, edgecolor="crimson", lw=2))
    ax.set_xlabel("predicted sense number  (★ = gold)")
    ax.set_ylabel(by)
    ax.set_title(f"{lemma} q{int(float(quote_idx))} — votes by {by} "
                 f"(n={int(data.sum())} cells)")
    plt.tight_layout()
    return ax


def quote_details(df: pd.DataFrame, lemma: str, quote_idx,
                  features_df: pd.DataFrame | None = None
                  ) -> tuple[dict | None, pd.DataFrame | None]:
    """Return (context_dict, predictions_df) for one quote. predictions_df
    has one row per (provider, config) with predicted + outcome columns."""
    qkey = (df["lemma"] == lemma) & (df["quote_idx"] == float(quote_idx))
    sub = df[qkey].copy()
    if sub.empty:
        return None, None
    head = sub.iloc[0]
    gold_sid = head.get("gold_sid")
    context = {
        "lemma": lemma,
        "quote_idx": quote_idx,
        "gold_number": head.get("gold_number"),
        "gold_sid": gold_sid,
    }
    # Look up the gold sense's label + Latin/Polish glosses from the inventory.
    if gold_sid:
        inv = _load_inventory()
        gold_row = inv[inv["sid"] == gold_sid]
        if not gold_row.empty:
            gr = gold_row.iloc[0]
            context["gold_label"] = gr.get("label")
            context["gold_def_la"] = gr.get("def_la")
            context["gold_def_pl"] = gr.get("def_pl")
    context["quotation_wsd (LLM input)"] = head.get("quotation_wsd")
    if features_df is not None:
        fr = features_df[(features_df.lemma == lemma) &
                         (features_df.quote_idx == float(quote_idx))]
        if not fr.empty:
            fr = fr.iloc[0]
            context.update({
                "source_clean": fr.get("source_clean"),
                "genre": fr.get("genre"),
                "length_bin": fr.get("length_bin"),
                "n_words": fr.get("n_words"),
                "century": fr.get("century"),
                "has_polish_gloss": fr.get("has_polish_gloss"),
                "has_sc_note": fr.get("has_sc_note"),
                "quote_text (cleaned)": fr.get("quote_text"),
            })
    sub["predicted"] = sub["output"].map(parse_predicted_number)
    cfg_col = "config" if "config" in sub.columns else "template"
    preds = sub[["provider_label", cfg_col, "predicted", "outcome"]].rename(
        columns={"provider_label": "provider", cfg_col: "config"})
    # Decode each predicted number into the sense it points to for THIS quote
    # (numbering is per-quote shuffled — see exp1_candidate_senses). pred_sid /
    # pred_label make each row self-identifying; full glosses live in the
    # candidate legend + breakdown (exp1_prediction_breakdown).
    gran = head.get("granularity") or "l1"
    cand = exp1_candidate_senses(lemma, quote_idx, granularity=gran,
                                 gold_sid=gold_sid)

    def _pred_field(num, field):
        if pd.isna(num) or int(num) not in cand.index:
            return ""
        return cand.loc[int(num), field]

    # Decode each predicted number into the sense it points to for THIS quote:
    # the stable sid + the actual Latin/Polish glosses the model chose (so the
    # confusion is legible without a separate legend lookup).
    preds["pred_sid"] = [_pred_field(n, "sid") for n in preds["predicted"]]
    preds["def_la"] = [_clean_gloss(_pred_field(n, "def_la"))
                       for n in preds["predicted"]]
    preds["def_pl"] = [_clean_gloss(_pred_field(n, "def_pl"))
                       for n in preds["predicted"]]
    preds = preds[["provider", "config", "predicted", "pred_sid",
                   "def_la", "def_pl", "outcome"]]
    return context, preds.sort_values(["provider", "config"]).reset_index(drop=True)


def _quote_detail_html(df: pd.DataFrame, lemma: str, quote_idx,
                       features_df: pd.DataFrame | None = None) -> str:
    """One quote's drill-down rendered as a single HTML string: context block,
    candidate-sense legend (★ = gold), per-(provider,config) predictions and
    the predicted-sense × provider breakdown. Used by quote_picker; usable on
    its own via display(HTML(_quote_detail_html(df, lemma, qidx)))."""
    import html as _html
    ctx, preds = quote_details(df, lemma, quote_idx, features_df=features_df)
    if ctx is None or preds is None:
        return f"<i>no rows for {_html.escape(str(lemma))} q{quote_idx}</i>"
    ctx_lines = "\n".join(f"{k:<28}: {v}" for k, v in ctx.items())
    parts = [f"<h4 style='margin:2px 0'>{_html.escape(str(lemma))} "
             f"quote_idx={quote_idx}</h4>",
             f"<pre style='white-space:pre-wrap;font-size:0.85em;margin:2px 0'>"
             f"{_html.escape(ctx_lines)}</pre>"]
    cand = exp1_candidate_senses(lemma, quote_idx, gold_sid=ctx.get("gold_sid"))
    if not cand.empty:
        legend = cand.copy()
        legend.insert(0, "gold", legend.pop("is_gold").map({True: "★", False: ""}))
        legend["def_la"] = legend["def_la"].map(_truncate)
        legend["def_pl"] = legend["def_pl"].map(_truncate)
        parts += ["<b>Candidate senses (the numbered menu the model saw; "
                  "★ = gold):</b>",
                  legend[["gold", "sid", "label", "def_la", "def_pl"]].to_html(border=0)]
    parts += ["<b>Per-(provider, config) predictions:</b>",
              preds.to_html(index=False, border=0)]
    brk = exp1_prediction_breakdown(preds, cand)
    if not brk.empty:
        parts += ["<b>Predicted-sense breakdown — count by provider "
                  "(what the gold sense is confused with):</b>",
                  brk.to_html(border=0)]
    # Each block in its own horizontally-scrollable div.
    return "".join(
        f"<div style='overflow-x:auto;margin-bottom:6px'>{p}</div>" for p in parts)


def quote_picker(per_q: pd.DataFrame, df: pd.DataFrame,
                 features_df: pd.DataFrame | None = None,
                 top_n: int = 50, sort_by: str = "difficulty"):
    """Build a dropdown of the top-N quotes (sorted by `sort_by`); pick one to
    see context + candidate-sense legend + per-(provider,config) predictions +
    predicted-sense breakdown. Requires ipywidgets; falls back to a printed
    table if not installed.

    Renders into an ipywidgets.HTML whose .value is REPLACED on each selection
    (not streamed into an Output widget), so VSCode/Jupyter cannot duplicate or
    stack the output when the widget state re-syncs."""
    try:
        import ipywidgets as widgets
    except ImportError:
        print("(ipywidgets not installed — falling back to a static table.")
        print(" install with: pip install ipywidgets   then re-import _eval_loader)")
        return quotes_table(per_q, top_n=top_n, sort_by=sort_by)

    sorted_q = per_q.sort_values(sort_by, ascending=False).head(top_n)
    options = []
    for (lemma, qidx), r in sorted_q.iterrows():
        diff = r.get("difficulty", float("nan"))
        text = (r.get("quote_text") or "")[:50]
        label = f"{lemma} q{int(qidx)} | diff={diff:.2f} | {text}"
        options.append((label, (lemma, qidx)))
    dropdown = widgets.Dropdown(options=options, description="Quote:",
                                 layout=widgets.Layout(width="92%"))
    out = widgets.HTML()

    def _render(_change=None):
        lemma, qidx = dropdown.value
        out.value = _quote_detail_html(df, lemma, qidx, features_df=features_df)

    # names='value' so the callback fires once per selection (not once per
    # trait change — picking an option mutates index/label/value).
    dropdown.observe(_render, names="value")
    _render()
    return widgets.VBox([dropdown, out])


# ===========================================================================
# EXPERIMENT 2 — loader / classifiers / baselines.
#
# Exp 2 ran annotation FIRST (Argilla → curated gold), so the eval scores
# correctness the Exp-1 way, but the promptfoo test-case shape differs:
#   - analysis covariates live in `test_case.metadata` (not vars): key,
#     record_id, siglum, gold_sense_id, gold_category, resolution, restored,
#     wsd_status, context_truncated, n_annotators, min_conf, mean_conf,
#     genuine_ambiguity (see scripts/exp2_tests.py `_COVARIATES`);
#   - the runtime axis is `context_length` (kwic_only / kwic_plus_prev), not
#     Exp 1's `granularity`; `show_metadata` is shared; there is no yaml_kind
#     (plain only — JSON dropped, methodology log 2026-05-16c);
#   - the gold sense id (`gold_sense_id`, string; "0" = novelty) is in metadata;
#     the presented number (`gold_number`, int) is in vars;
#   - the model input var is `sentence_wsd` (not `quotation_wsd`); there is no
#     `quote_idx` — records are keyed by `key` (= lemma::record_id). record_id
#     alone is NOT unique across lemmas (analysis/exp2_pipeline.md "Gotchas").
#
# The CI / bootstrap / McNemar / Holm / kappa_vs_gold / parse_predicted_number
# helpers above are reused verbatim; only load + classify + baselines differ.
# Released runs are selected by their run column (keys `exp2_core`,
# `exp2_extended`); resolve with eval_id(name) as for Exp 1.
# ===========================================================================

EXP2_EVAL_SET_CSV = _DATA_ROOT / "neolatin_concordances" / "concordance_lines.csv"
EXP2_GOLD_CSV = _DATA_ROOT / "neolatin_concordances" / "gold.csv"
EXP2_SENSES_CSV = _DATA_ROOT / "senses" / "senses_l1.csv"

# Lemma lineup (scripts/exp2_tests.py TIER_A/TIER_B). Lower-case — the eval_set,
# gold.csv and annotated/* CSVs all key on lower-case lemmas; the SLS sense
# inventory (senses_exp2_l1.csv) keys on UPPER-case (+ homonym suffix).
EXP2_TIER_A = ("civitas", "populus", "dignitas", "imperator",
               "natio", "episcopus", "comitia", "praefectus")
EXP2_TIER_B = ("officium", "lex", "princeps", "libertas", "bellum", "dux")
EXP2_LEMMAS = EXP2_TIER_A + EXP2_TIER_B

# Runtime axes (scripts/exp2_tests.py). CORE pins context_length=kwic_only and
# show_metadata=False; EXTENDED sweeps the rest.
EXP2_AXES = ("def_language", "context_length", "show_metadata")
EXP2_AXIS_VALUES = {
    "def_language":   ("la", "pl"),
    "context_length": ("kwic_only", "kwic_plus_prev"),
    "show_metadata":  (False, True),
}

EXP2_TEMPLATE_LABELS: dict[str, str] = {
    "exp2_zeroshot.txt":            "zeroshot",
    "exp2_fewshot.txt":             "fewshot",
    "exp2_justified.txt":           "justified",
    "exp2_zeroshot_decomposed.txt": "decomposed",
}
_EXP2_PROMPT_FILE_RE = re.compile(r"(exp2_[a-zA-Z0-9_]+\.txt)")

# Covariate columns carried in test_case.metadata (string-typed in SQLite).
_EXP2_META_COVARS = ["gold_category", "resolution", "restored", "wsd_status",
                     "context_truncated", "n_annotators", "min_conf",
                     "mean_conf", "genuine_ambiguity"]


def classify_exp2_template(prompt_label: str | None) -> str:
    if not prompt_label:
        return "unknown_<none>"
    m = _EXP2_PROMPT_FILE_RE.search(prompt_label)
    if not m:
        return f"unknown_{(prompt_label or '')[:30]}"
    fname = m.group(1)
    return EXP2_TEMPLATE_LABELS.get(fname, f"unknown_{fname}")


def _coerce_bool_flag(s: pd.Series) -> pd.Series:
    """Coerce a metadata/vars flag to bool. Handles SQLite json_extract's int
    0/1 (JSON booleans) and the string forms 'True'/'False'/'' the metadata
    dict stores (eval_set covariates are written as strings)."""
    def one(v):
        if v in (1, True):
            return True
        if v in (0, False, None):
            return bool(v) if v in (0, 1, True, False) else False
        return str(v).strip().lower() in ("true", "1")
    return s.map(one).astype(bool)


def load_exp2_eval(eval_id: str, db_path: str | os.PathLike | None = None,
                   join_eval_set: bool = True,
                   eval_set_csv: str | os.PathLike = EXP2_EVAL_SET_CSV
                   ) -> tuple[dict, pd.DataFrame]:
    """Load one Exp 2 eval into (meta_dict, dataframe).

    Columns: provider_label, lemma, key, record_id, siglum, def_language,
    context_length, show_metadata, template, task_type, gold_number,
    gold_sense_id, output, predicted, success, outcome, correct, reason,
    prompt_tokens, completion_tokens, reasoning_tokens, cost, eval_id, plus the
    metadata covariates (gold_category, resolution, restored[bool], wsd_status,
    context_truncated, n_annotators, min_conf, mean_conf,
    genuine_ambiguity[bool]). With join_eval_set=True (default) the restored
    sentence text + n_words + ann_senses are merged from eval_set.csv on `key`.
    """
    p = Path(db_path) if db_path else DEFAULT_DB
    con = _private_backend_removed(str(p))
    try:
        meta_row = con.execute(
            "SELECT id, description, created_at FROM evals WHERE id=?",
            (eval_id,),
        ).fetchone()
        if meta_row is None:
            raise ValueError(f"eval_id {eval_id!r} not found in {p}")
        meta = {"id": meta_row[0], "description": meta_row[1],
                "created_at_ms": meta_row[2]}

        rows = con.execute("""
            SELECT
              json_extract(prompt,        '$.label'),
              json_extract(provider,      '$.label'),
              json_extract(test_case,     '$.metadata.key'),
              json_extract(test_case,     '$.metadata.record_id'),
              json_extract(test_case,     '$.metadata.siglum'),
              json_extract(test_case,     '$.vars.def_language'),
              json_extract(test_case,     '$.vars.context_length'),
              json_extract(test_case,     '$.vars.show_metadata'),
              json_extract(test_case,     '$.vars.gold_number'),
              json_extract(test_case,     '$.metadata.gold_sense_id'),
              json_extract(test_case,     '$.metadata.gold_category'),
              json_extract(test_case,     '$.metadata.resolution'),
              json_extract(test_case,     '$.metadata.restored'),
              json_extract(test_case,     '$.metadata.wsd_status'),
              json_extract(test_case,     '$.metadata.context_truncated'),
              json_extract(test_case,     '$.metadata.n_annotators'),
              json_extract(test_case,     '$.metadata.min_conf'),
              json_extract(test_case,     '$.metadata.mean_conf'),
              json_extract(test_case,     '$.metadata.genuine_ambiguity'),
              json_extract(response,      '$.output'),
              success,
              json_extract(grading_result,'$.reason'),
              json_extract(response,      '$.tokenUsage.prompt'),
              json_extract(response,      '$.tokenUsage.completion'),
              json_extract(response,      '$.tokenUsage.completionDetails.reasoning')
            FROM eval_results
            WHERE eval_id=?
        """, (eval_id,)).fetchall()
    finally:
        con.close()

    cols = ["prompt_label", "provider_label", "key", "record_id", "siglum",
            "def_language", "context_length", "show_metadata", "gold_number",
            "gold_sense_id", "gold_category", "resolution", "restored",
            "wsd_status", "context_truncated", "n_annotators", "min_conf",
            "mean_conf", "genuine_ambiguity", "output", "success", "reason",
            "prompt_tokens", "completion_tokens", "reasoning_tokens"]
    df = pd.DataFrame(rows, columns=cols)

    # lemma: lower-case, from the composite key (= lemma::record_id). Matches
    # eval_set / gold / annotated CSVs; record_id alone is not lemma-unique.
    df["lemma"] = df["key"].astype(str).str.split("::").str[0]
    df["template"] = df["prompt_label"].map(classify_exp2_template)
    # Exp 2 has no JSON variants, so task_type == template; kept for parity
    # with the Exp 1 notebook helpers that group on task_type.
    df["task_type"] = df["template"]
    df["success"] = df["success"].fillna(0).astype(int).astype(bool)
    df["outcome"] = [classify_outcome(s, r) for s, r in zip(df["success"], df["reason"])]
    df["correct"] = df["outcome"].eq("correct")
    df["predicted"] = df["output"].map(parse_predicted_number)

    df["show_metadata"] = _coerce_bool_flag(df["show_metadata"])
    df["restored"] = _coerce_bool_flag(df["restored"])
    df["genuine_ambiguity"] = _coerce_bool_flag(df["genuine_ambiguity"])
    df["gold_number"] = pd.to_numeric(df["gold_number"], errors="coerce")
    df["min_conf"] = pd.to_numeric(df["min_conf"], errors="coerce")
    df["mean_conf"] = pd.to_numeric(df["mean_conf"], errors="coerce")
    df["n_annotators"] = pd.to_numeric(df["n_annotators"], errors="coerce")
    for c in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["cost"] = df.apply(lambda r: row_cost(r.provider_label,
                                             r.prompt_tokens, r.completion_tokens), axis=1)
    df["eval_id"] = eval_id

    if join_eval_set:
        ev = pd.read_csv(eval_set_csv, dtype=str, keep_default_na=False)
        ev = ev[["key", "sentence", "sentence_wsd", "prev_sent", "next_sent",
                 "ann_senses"]].drop_duplicates("key")
        ev["n_words"] = ev["sentence"].str.split().map(len)
        ev["sent_chars"] = ev["sentence"].str.len()
        df = df.merge(ev, on="key", how="left")
    return meta, df


def load_exp2_evals(*eval_ids: str, db_path: str | os.PathLike | None = None,
                    join_eval_set: bool = True
                    ) -> tuple[list[dict], pd.DataFrame]:
    """Load one or more Exp 2 evals into (list_of_metas, concatenated_df).
    Loading CORE + EXTENDED together yields a frame whose context_length /
    show_metadata axes carry both values (CORE supplies only kwic_only+meta-off).
    """
    metas, dfs = [], []
    for eid in eval_ids:
        m, d = load_exp2_eval(eid, db_path=db_path, join_eval_set=join_eval_set)
        metas.append(m); dfs.append(d)
    return metas, pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Exp 2 baselines / lemma features.

def exp2_mfs_baseline(eval_set_csv: str | os.PathLike = EXP2_EVAL_SET_CSV
                      ) -> tuple[dict[str, float], float]:
    """Most-Frequent-Sense baseline per lemma on the analyzable gold set.
    Returns (per-lemma acc, macro mean). The novelty class ("0") is a legitimate
    gold value and participates in the MFS count. Macro = unweighted mean across
    the 14 lemmas (each lemma contributes equally), matching PLAN.md §2.5."""
    ev = pd.read_csv(eval_set_csv, dtype=str, keep_default_na=False)
    per_lemma: dict[str, float] = {}
    for lemma, g in ev.groupby("lemma"):
        gold = g["gold_sense_id"][g["gold_sense_id"].str.strip() != ""]
        if len(gold) == 0:
            continue
        per_lemma[lemma] = gold.value_counts().iloc[0] / len(gold)
    macro = sum(per_lemma.values()) / len(per_lemma) if per_lemma else float("nan")
    return per_lemma, macro


def exp2_lemma_features(eval_set_csv: str | os.PathLike = EXP2_EVAL_SET_CSV,
                        senses_csv: str | os.PathLike = EXP2_SENSES_CSV
                        ) -> pd.DataFrame:
    """Per-lemma covariates. Indexed by (lower-case) lemma with columns:
      n_candidate_senses — senses the model was shown (def_la OR def_pl present),
                           homonym-aware (COMITIA-2/POPULUS-1 resolved from the
                           gold sid prefix as exp2_transform._resolve_inv_lemma);
      n_records          — analyzable records for the lemma in eval_set;
      records_per_sense  — coverage ratio (low = polysemous, sparsely evidenced).
    """
    ev = pd.read_csv(eval_set_csv, dtype=str, keep_default_na=False)
    inv = pd.read_csv(senses_csv)
    has_def = inv["def_la"].fillna("").str.strip().ne("") | \
              inv["def_pl"].fillna("").str.strip().ne("")
    sense_counts = inv[has_def].groupby("lemma").size()  # keyed UPPER (+ homonym)
    out_rows = []
    for lemma, g in ev.groupby("lemma"):
        lemma_uc = lemma.upper()
        keys: set[str] = set()
        for sid in g["gold_sense_id"]:
            if sid and sid != "0":
                cand = sid.split(".")[0].upper().replace("_", "-")
                if cand in sense_counts.index:
                    keys.add(cand)
        if not keys and lemma_uc in sense_counts.index:
            keys.add(lemma_uc)
        n_senses = int(sum(int(sense_counts.get(k, 0)) for k in keys)) if keys else float("nan")
        out_rows.append({"lemma": lemma, "n_candidate_senses": n_senses,
                         "n_records": len(g)})
    out = pd.DataFrame(out_rows).set_index("lemma")
    out["records_per_sense"] = (out["n_records"] / out["n_candidate_senses"]).round(2)
    return out


def exp2_attach_sense_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `predicted_sid` column by reconstructing each record's number→sid
    map. The candidate-sense numbering is **shuffled per record** (seed =
    record_id; scripts/transform.py:numbered_senses), so the same number means
    *different* senses across records — accuracy (predicted==gold within a
    record) is unaffected, but any cross-record analysis on the numbers (κ,
    confusion matrices, per-sense P/R/F1) is invalid on numbers and must use the
    reconstructed sense ids. gold_sid is already in `gold_sense_id`.

    The map depends only on (resolved inventory lemma, record_id) — the shuffle
    seed is def_language-independent (verified: la/pl give identical orderings).
    Number 0 → "0" (the novelty option). Out-of-range predictions → <NA>.
    Requires scripts/ importable (transform.py + exp2_transform.py)."""
    out = df.copy()
    if "predicted_sid" not in out.columns:
        raise ValueError("released predictions must contain predicted_sid")
    out["predicted_sid"] = out["predicted_sid"].replace("", pd.NA)
    return out


def kappa_labels(predicted, gold) -> float:
    """Cohen's κ on arbitrary (string) labels — e.g. reconstructed sense ids.
    NaN/None pairs are dropped. Use this for Exp 2 sense-id κ; use kappa_vs_gold
    only when both sides are already integers in a *shared* numbering."""
    from sklearn.metrics import cohen_kappa_score
    p = pd.Series(list(predicted), dtype="object")
    g = pd.Series(list(gold), dtype="object")
    mask = p.notna() & g.notna() & (p.astype(str) != "") & (g.astype(str) != "")
    if mask.sum() == 0:
        return float("nan")
    return float(cohen_kappa_score(g[mask].astype(str), p[mask].astype(str)))


# ---------------------------------------------------------------------------
# Exp 2 record-level drill-down — the analogs of the Exp 1 quote_* helpers
# (per_quote_outcome_matrix / quote_details / quotes_where / inspect_quotes /
# quote_picker), keyed on `key` (lemma::record_id) with exp2 columns
# (sentence_wsd, gold_sense_id, predicted_sid, config + annotation covariates).
# Pass a frame with predicted_sid attached (exp2_attach_sense_ids) and, for the
# headline view, the CORE slice.

_PER_RECORD_AGG = dict(
    lemma=("lemma", "first"), gold_sense_id=("gold_sense_id", "first"),
    n_cells=("correct", "size"), n_correct=("correct", "sum"),
    n_words=("n_words", "first"), min_conf=("min_conf", "first"),
    mean_conf=("mean_conf", "first"), genuine_ambiguity=("genuine_ambiguity", "first"),
    resolution=("resolution", "first"), restored=("restored", "first"),
    context_truncated=("context_truncated", "first"), siglum=("siglum", "first"),
    sentence=("sentence", "first"), sentence_wsd=("sentence_wsd", "first"),
    ann_senses=("ann_senses", "first"),
)


def per_record_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per `key` with consensus signals across all (provider, config)
    cells: n_cells, n_correct, n_wrong, n_ff, difficulty (= fraction of cells
    wrong) + the per-record covariates. Exp 2 analog of per_quote_outcome_matrix."""
    work = df.copy()
    by = work.groupby("key").agg(**_PER_RECORD_AGG)
    by["n_ff"] = (work[work.outcome == "format_failure"].groupby("key").size()
                  .reindex(by.index, fill_value=0))
    by["n_wrong"] = by["n_cells"] - by["n_correct"] - by["n_ff"]
    by["difficulty"] = 1.0 - by["n_correct"] / by["n_cells"]
    return by


def record_details(df: pd.DataFrame, key: str,
                   senses_csv: str | os.PathLike = EXP2_SENSES_CSV
                   ) -> tuple[dict | None, pd.DataFrame | None]:
    """(context_dict, predictions_df) for one record. context carries the
    <WSD>-marked input, the gold sense's label + Latin/Polish definitions, and
    the annotation covariates; predictions_df is one row per (provider, config)
    with predicted (number), predicted_sid, and outcome."""
    sub = df[df["key"] == key].copy()
    if sub.empty:
        return None, None
    head = sub.iloc[0]
    gsid = head.get("gold_sense_id")
    ctx = {
        "key": key, "lemma": head.get("lemma"), "gold_sense_id": gsid,
        "difficulty": round(1.0 - sub["correct"].mean(), 3),
        "min_conf": head.get("min_conf"), "mean_conf": head.get("mean_conf"),
        "genuine_ambiguity": head.get("genuine_ambiguity"),
        "resolution": head.get("resolution"), "restored": head.get("restored"),
        "context_truncated": head.get("context_truncated"),
        "ann_senses (annotators)": head.get("ann_senses"),
        "sentence_wsd (LLM input)": head.get("sentence_wsd"),
    }
    inv = _load_inventory(senses_csv)
    grow = inv[inv["sid"] == gsid] if gsid else inv.iloc[0:0]
    if not grow.empty:
        gr = grow.iloc[0]
        ctx["gold_label"] = gr.get("label")
        ctx["gold_def_la"] = gr.get("def_la")
        ctx["gold_def_pl"] = gr.get("def_pl")
    elif gsid == "0":
        ctx["gold_label"] = "(novelty — no listed sense)"
    preds = (sub[["provider_label", "config", "predicted", "predicted_sid", "outcome"]]
             .rename(columns={"provider_label": "provider"})
             .sort_values(["provider", "config"]).reset_index(drop=True))
    return ctx, preds


def records_where(per_rec: pd.DataFrame, lemma: str | None = None,
                  difficulty_gte: float | None = None,
                  difficulty_lte: float | None = None,
                  n_correct_eq: int | None = None, **equals) -> pd.DataFrame:
    """Filter the per-record matrix by difficulty thresholds and column equals.
    Examples: records_where(per_rec, gold_sense_id='0'); records_where(per_rec,
    lemma='officium', difficulty_gte=0.7); records_where(per_rec, n_correct_eq=0)."""
    sub = per_rec
    if lemma is not None:
        sub = sub[sub["lemma"] == lemma]
    if difficulty_gte is not None:
        sub = sub[sub["difficulty"] >= difficulty_gte]
    if difficulty_lte is not None:
        sub = sub[sub["difficulty"] <= difficulty_lte]
    if n_correct_eq is not None:
        sub = sub[sub["n_correct"] == n_correct_eq]
    for col, val in equals.items():
        if col in sub.columns:
            sub = sub[sub[col] == val]
    return sub


def inspect_records(per_rec_subset: pd.DataFrame, df: pd.DataFrame, max_n: int = 5,
                    senses_csv: str | os.PathLike = EXP2_SENSES_CSV) -> None:
    """Pretty-print up to max_n records from a per-record subset (after
    records_where) with their context + a prediction summary."""
    n = len(per_rec_subset)
    for i, (key, _) in enumerate(per_rec_subset.iterrows()):
        if i >= max_n:
            print(f"\n… and {n - max_n} more (raise max_n to see them)")
            return
        ctx, preds = record_details(df, key, senses_csv)
        if ctx is None:
            continue
        print(f"=== {key}  (difficulty {ctx['difficulty']}) ===")
        for k, v in ctx.items():
            if k != "key":
                print(f"  {k:26s}: {str(v)[:170]}")
        print(f"  prediction spread (sid): {preds['predicted_sid'].value_counts().to_dict()}")
        print()


def record_picker(per_rec: pd.DataFrame, df: pd.DataFrame, top_n: int = 50,
                  sort_by: str = "difficulty",
                  senses_csv: str | os.PathLike = EXP2_SENSES_CSV):
    """Dropdown of the top-N records (by `sort_by`); pick one to see its context
    + per-(provider,config) predictions. Requires ipywidgets; falls back to a
    static table when absent (so it still renders headless). Exp 2 analog of
    quote_picker."""
    cols = ["lemma", "gold_sense_id", "difficulty", "n_correct", "n_cells",
            "min_conf", "genuine_ambiguity", "sentence"]
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("(ipywidgets not installed — static table fallback; pip install ipywidgets for the picker)")
        return per_rec.sort_values(sort_by, ascending=False).head(top_n)[
            [c for c in cols if c in per_rec.columns]]

    srt = per_rec.sort_values(sort_by, ascending=False).head(top_n)
    options = [(f"{r['lemma']} | diff={r['difficulty']:.2f} | {str(r['sentence'])[:48]}", key)
               for key, r in srt.iterrows()]
    dd = widgets.Dropdown(options=options, description="Record:",
                          layout=widgets.Layout(width="95%"))
    out = widgets.Output()

    def _render(_change=None):
        out.clear_output(wait=True)
        with out:
            ctx, preds = record_details(df, dd.value, senses_csv)
            if ctx is None:
                print("not found"); return
            print(f"=== {dd.value} ===")
            for k, v in ctx.items():
                if k != "key":
                    print(f"  {k:26s}: {v}")
            print()
            display(preds)

    dd.observe(_render, names="value")
    _render()
    return widgets.VBox([dd, out])


# ---------------------------------------------------------------------------
# Public release backend
#
# The source project historically loaded promptfoo's private SQLite store.
# Released notebooks use the exported CSVs instead.  These definitions are
# intentionally placed last so the analysis helper API above remains stable.

def _public_data_root() -> Path:
    return _DATA_ROOT


def _public_paths() -> dict[str, Path]:
    root = _public_data_root()
    return {
        "root": root,
        "exp1": root / "llm_predictions" / "exp1_predictions.csv",
        "exp2": root / "llm_predictions" / "exp2_predictions.csv",
        "features": root / "dictionary_quotations" / "quote_features.csv",
        "quotations": root / "dictionary_quotations" / "quotations.csv",
        "sources": root / "neolatin_concordances" / "sources.csv",
        "eval_set": root / "neolatin_concordances" / "concordance_lines.csv",
        "gold": root / "neolatin_concordances" / "gold.csv",
        "annotations": root / "neolatin_concordances" / "annotations.csv",
        "sigla": root / "neolatin_concordances" / "sigla.csv",
        "senses_l1": root / "senses" / "senses_l1.csv",
        "senses_l1l2": root / "senses" / "senses_l1l2.csv",
    }


def _public_meta(name: str) -> dict:
    descriptions = {
        "phase_a_plain_restricted": "Exp. 1 Phase A plain restricted",
        "phase_a_json_restricted": "Exp. 1 Phase A JSON restricted",
        "phase_b_plain": "Exp. 1 Phase B plain",
        "exp2_core": "Exp. 2 CORE",
        "exp2_extended": "Exp. 2 EXTENDED",
    }
    return {"id": name, "description": descriptions.get(name, name),
            "source": "released CSV", "created_at_ms": 0}


EVALS = {name: {"id": name, "description": _public_meta(name)["description"]}
         for name in ("phase_a_plain_restricted", "phase_a_json_restricted",
                      "phase_b_plain", "exp2_core", "exp2_extended")}


def eval_id(name: str) -> str:
    if name not in EVALS:
        raise KeyError(f"Unknown released run: {name!r}")
    return name


def _require_prediction_columns(df: pd.DataFrame, run: str) -> None:
    required = {"predicted_number", "predicted_sid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{run}: released predictions missing {sorted(missing)}")
    shares = {c: float(df[c].ne("").mean()) for c in required}
    if any(value < 0.85 for value in shares.values()):
        raise ValueError(f"{run}: suspicious prediction coverage {shares}")

def _require_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns {sorted(missing)}")


def _public_exp1() -> pd.DataFrame:
    p = _public_paths()["exp1"]
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    _require_prediction_columns(df, "Exp1")
    df["provider_label"] = df["provider"]
    df["quote_idx"] = pd.to_numeric(df["quote_idx"], errors="coerce")
    df["gold_number"] = pd.to_numeric(df["gold_number"], errors="coerce")
    df["show_metadata"] = df["show_metadata"].str.lower().eq("true")
    df["yaml_kind"] = df["template"].str.endswith("_json").map({True: "json", False: "plain"})
    df["task_type"] = df["template"].str.replace(r"_json$", "", regex=True)
    df["success"] = df["outcome_stored"].eq("correct")
    df["outcome"] = df["outcome_stored"]
    df["outcome_corrected"] = df["outcome_corrected"]
    df["predicted"] = pd.to_numeric(df["predicted_number"], errors="coerce")
    df["reason"] = ""
    df["output"] = ""
    df["eval_id"] = df["run"]
    for column in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[column] = 0
    df["cost"] = 0.0
    return df


def _public_exp2() -> pd.DataFrame:
    paths = _public_paths()
    df = pd.read_csv(paths["exp2"], dtype=str, keep_default_na=False)
    _require_prediction_columns(df, "Exp2")
    df["provider_label"] = df["provider"]
    df["lemma"] = df["key"].str.split("::").str[0]
    df["gold_number"] = pd.to_numeric(df["gold_number"], errors="coerce")
    df["show_metadata"] = df["show_metadata"].str.lower().eq("true")
    df["success"] = df["outcome_stored"].eq("correct")
    df["outcome"] = df["outcome_stored"]
    df["outcome_corrected"] = df["outcome_corrected"]
    df["predicted"] = pd.to_numeric(df["predicted_number"], errors="coerce")
    df["reason"] = ""
    df["output"] = ""
    df["correct"] = df["outcome"].eq("correct")
    ev = pd.read_csv(paths["eval_set"], dtype=str, keep_default_na=False)
    _require_columns(ev, {"key", "gold_category", "resolution", "restored", "min_conf", "mean_conf"}, "concordance_lines.csv")
    keep = [c for c in ("key", "sentence", "sentence_wsd", "prev_sent", "next_sent", "ann_senses",
                        "gold_category", "resolution", "restored", "context_truncated",
                        "min_conf", "mean_conf", "genuine_ambiguity")
            if c in ev.columns]
    df = df.merge(ev[keep].drop_duplicates("key"), on="key", how="left")
    for column in ("genuine_ambiguity", "restored", "context_truncated"):
        if column in df.columns and column != "context_truncated":
            df[column] = df[column].astype(str).str.lower().eq("true")
    if "sentence" in df:
        df["n_words"] = df["sentence"].str.split().map(len)
        df["sent_chars"] = df["sentence"].str.len()
    df["eval_id"] = df["run"]
    for column in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[column] = 0
    df["cost"] = 0.0
    return df


def load_eval(eval_id: str, db_path=None):
    name = eval_id
    df = _public_exp1()
    df = df[df["run"].eq(name)].copy()
    if df.empty:
        raise ValueError(f"Released Exp. 1 run not found: {name}")
    return _public_meta(name), df


def load_evals(*eval_ids: str, db_path=None):
    frames = []
    metas = []
    for name in eval_ids:
        meta, frame = load_eval(name)
        metas.append(meta); frames.append(frame)
    return metas, pd.concat(frames, ignore_index=True)


def load_exp2_eval(eval_id: str, db_path=None, join_eval_set=True, eval_set_csv=None):
    df = _public_exp2()
    df = df[df["run"].eq(eval_id)].copy()
    if df.empty:
        raise ValueError(f"Released Exp. 2 run not found: {eval_id}")
    if not join_eval_set:
        # The public backend already performs this join; retain the argument
        # for notebook compatibility.
        pass
    return _public_meta(eval_id), df


def load_exp2_evals(*eval_ids: str, db_path=None, join_eval_set=True):
    metas, frames = [], []
    for name in eval_ids:
        meta, frame = load_exp2_eval(name, join_eval_set=join_eval_set)
        metas.append(meta); frames.append(frame)
    return metas, pd.concat(frames, ignore_index=True)


def load_quote_features(csv_path=None):
    df = pd.read_csv(csv_path or _public_paths()["features"], dtype=str, keep_default_na=False)
    # The public CSV is intentionally text-oriented, but analysis code uses
    # these two fields as numeric join/binning keys.
    if "quote_idx" in df.columns:
        df["quote_idx"] = pd.to_numeric(df["quote_idx"], errors="raise")
    if "n_words" in df.columns:
        df["n_words"] = pd.to_numeric(df["n_words"], errors="raise")
    if "century" in df.columns:
        df["century"] = pd.to_numeric(df["century"], errors="coerce")
    for column in ("genre", "century"):
        if column in df.columns:
            df[column] = df[column].fillna("unknown")
    for column in ("has_polish_gloss", "has_sc_note"):
        if column in df.columns:
            df[column] = df[column].fillna(False)
    return df


def compute_lemma_features(df_eval: pd.DataFrame, inventory_csv=None) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv or _public_paths()["senses_l1"])
    has_def = inv["def_la"].fillna("").str.strip().ne("") | inv["def_pl"].fillna("").str.strip().ne("")
    n_senses = inv[has_def].groupby("lemma").size().rename("n_candidate_senses")
    n_quotes = df_eval.groupby("lemma")["quote_idx"].nunique().rename("n_quotes_per_lemma")
    out = pd.concat([n_senses, n_quotes], axis=1).dropna(subset=["n_quotes_per_lemma"])
    out["quotes_per_sense"] = (out["n_quotes_per_lemma"] / out["n_candidate_senses"]).round(2)
    return out


REPO_ROOT = _public_data_root()
EXP2_EVAL_SET_CSV = _public_paths()["eval_set"]
EXP2_GOLD_CSV = _public_paths()["gold"]
EXP2_SENSES_CSV = _public_paths()["senses_l1"]
QUOTE_FEATURES_CSV = _public_paths()["features"]
SPLIT_CSV = _public_paths()["quotations"]
SIGLA_CSV = _public_paths()["sigla"]
