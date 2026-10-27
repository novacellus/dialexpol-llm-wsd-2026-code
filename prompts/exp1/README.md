# Experiment 1 prompts

Prompts for Exp 1 (sense assignment on SLS dictionary quotations) are **not
hand-written as flat files**. They are composed from a declarative component
spec so that shared wording lives in exactly one place and every prompt is
traceable to one design decision.

## The model

```
components.yaml   ← SOURCE OF TRUTH: every reusable block of prompt text
conditions.yaml   ← which blocks compose each template + the variable axes
render.py         ← components.yaml + conditions.yaml  →  rendered/*.txt   (TODO)
rendered/         ← GENERATED flat .txt that promptfoo actually reads — do not hand-edit
```

Edit `components.yaml` / `conditions.yaml`, then re-run `render.py`. Never edit
`rendered/*.txt` directly — they are build artifacts (committed for
reproducibility, but regenerated).

## Two templating levels — keep them separate

| Level | Resolved by | Resolves | Example |
|---|---|---|---|
| **Build-time** | `render.py` | which *blocks* go into each template | "few-shot = system + inventory + examples + quotation + task + output" |
| **Run-time** | promptfoo + `transform.py` | `{{placeholders}}` per test case | `{{senses}}`, `{{quotation_wsd}}`, `{{pos}}` |

Build-time produces ~8 `.txt` files. Run-time variables — definition language
(`la`/`pl`), sense granularity (`l1`/`l1l2`), source metadata (`on`/`off`) — are
**not** separate files; they are promptfoo variables that multiply each template
into the full screening matrix. 8 templates × run-time axes = the ~96-config
matrix in PLAN §2.2.

## Why this exists

- **Reproducibility** — each prompt derives from one spec; you cite "the
  quotation block", not 8 near-identical copies.
- **The spec is the documentation** — `components.yaml` answers "how do we
  introduce the quotation? how is the sense inventory presented? what is the
  output contract?" in one readable place.
- **Meaningful diffs** — a wording change is one line, not a sweep across files.

## Data contract

`transform.py` reads two files under `data/final/` — see
[`data/final/README.md`](../../data/final/README.md) for full schema and
canonical keys. For Exp 1:

- **Test cases:** `exp1/exp1_quotations_split.csv` — already curation-gated,
  carrying `split` (demo/test), `n_words`, `length_bin`, and the reconciled
  gold (`gold_l1` / `gold_l1l2` + `_label`). Produced by
  `analysis/exp1_split.ipynb`; rationale in `analysis/exp1_design_decisions.md`.
- **Candidate inventory** (`{{senses}}`): `senses_exp2_l1.csv` /
  `senses_exp2_l1l2.csv` — the granularity (`l1` / `l1l2`) axis picks the file.
- **Gold sense:** `gold_l1` (L1 runs) or `gold_l1l2` (L1L2 runs) — an
  `effective_sid` joinable to the inventory. Rows where the chosen `gold_*` is
  empty are **out of scope** for that granularity — drop them.
- **Senses are presented numbered** (`1. <def>` … `N. <def>`); the model
  answers with a number. `transform.py` derives `gold_number` (the gold sid's
  position) and `exp1_assert.py` scores the model's number against it. The gold
  sid stays in the vars for number↔sid analysis. See
  `exp1_design_decisions.md` §"Sense presentation".
- Few-shot demos come from `split == 'demo'` rows; scored items are `split == 'test'`.
- **Never** use `def_source_sid` or `l1_ancestor_sid` as the gold — see the
  data README and `exp1_design_decisions.md` §"Gold reconciliation".

## Status

- [x] `components.yaml`, `conditions.yaml` — spec (POS dropped; see PLAN §2.2)
- [x] `render.py` + `rendered/*.txt` — 8 templates
- [x] data prep — `analysis/exp1_split.ipynb` → `data/final/exp1/exp1_quotations_split.csv`
- [x] `scripts/transform.py` — derives `senses` / `quotation_wsd` / `examples` /
      `freeform_lang` / `show_metadata`; self-test passes
- [x] `scripts/exp1_tests.py` — test-matrix generator; applies `transform.py` so it emits
      fully-formed test cases (`generate_phase_a` / `_phase_b` / `_all`)
- [x] `promptfooconfig.yaml` + `scripts/exp1_assert.py` — 6 core prompts × 5 Phase-A
      providers × the generator; pinned decoding params; 3-way
      (correct / wrong / format_failure) assertion. Phase A is 5 models, not 6 — the
      Qwen3.5 hybrid and DeepSeek-V4 hybrid were both deferred to Phase B / the full-CoT
      add-on after their OpenRouter routes refused to disable reasoning (see
      `docs/openrouter/reasoning.md`). Core matrix only — the free-form & full-CoT
      add-ons need their own config(s).
