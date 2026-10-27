# *DiaLexPoL: LLM-assisted Sense Assignment for a Diachronic Dictionary of Latin*

## Krzysztof Nowak, Iwona Krawczyk, Jagoda Marszałek

This repository contains the prompts, pipeline components and analysis
notebooks used for the paper's experiments. It is a reproducibility release,
not a complete re-annotation workflow: the promptfoo database, raw provider
responses and private maintainer tooling are not included.

## Requirements

The pinned environment is specified in [`environment.yml`](environment.yml).
The notebooks use the released data repository, fetched with
[`fetch_data.py`](fetch_data.py).

## Structure

- [`analysis/`](analysis/) — five reproducible analysis notebooks, including
  the Exp. 1 JSON corrigendum analysis;
- [`pipeline/`](pipeline/) — prompt rendering, test-case generation and
  assertion code;
- [`prompts/`](prompts/) — prompt components and rendered templates;
- [`promptfoo/`](promptfoo/) — reference configurations for the reported
  evaluations;
- [`corrigendum.md`](corrigendum.md) — correction to the Exp. 1 JSON scoring
  result;
- [`maintainer/`](maintainer/) — documented export tooling, not required for
  running the notebooks.

Run notebooks from `analysis/`. Outputs go to `analysis/outputs/` by default;
set `DIALEXPOL_OUTPUT_DIR` to use another location.

## Data and citation

The companion dataset is available at
[`novacellus/dialexpol-wsd-datasets`](https://github.com/novacellus/dialexpol-wsd-datasets)
and [Zenodo](https://doi.org/10.5281/zenodo.21625867). Cite the code release
with [its Zenodo DOI](https://doi.org/10.5281/zenodo.21626003) and cite the
paper listed in [`CITATION.cff`](CITATION.cff).

The promptfoo configurations and prompts document the reported experiments;
they are not intended for rerunning the full LLM evaluations.
