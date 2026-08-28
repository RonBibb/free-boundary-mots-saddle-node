# From a Free-Boundary MOTS Saddle-Node to a Spacelike Marginal Tube in Dynamical Five-Dimensional Gravity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22148554.svg)](https://doi.org/10.5281/zenodo.22148554)

[Frozen source snapshot: `zenodo-22148554`](https://github.com/RonBibb/free-boundary-mots-saddle-node/tree/zenodo-22148554)

This repository contains the complete Python source and test surface, frozen
protocol implementations, and publication files for the accompanying paper.
The calculation resolves a brane-terminating free-boundary marginally outer
trapped-surface saddle-node and follows its stable branch into a finite
spacelike marginal-tube segment in a dynamically evolved five-dimensional,
Goldberger--Wise-stabilized braneworld.

The repository is deliberately **code-only with respect to scientific
outputs**. Numerical candidate outputs, checkpoints, and large result arrays
are not committed to Git. The citable result records and frozen manifests are
preserved in
[Zenodo version 10.5281/zenodo.22148554](https://doi.org/10.5281/zenodo.22148554),
which links back to this public code repository.
The complete version series is available through the
[concept DOI 10.5281/zenodo.22085054](https://doi.org/10.5281/zenodo.22085054).

## Repository contents

| Path | Contents |
| --- | --- |
| `README.md` | This directory guide, verification instructions, and archival mapping. |
| `CITATION.cff` | Machine-readable citation metadata for GitHub and reference managers. |
| `MANIFEST.sha256` | SHA-256 inventory of every published file other than the manifest itself and Git metadata. It includes every Python source and test file. |
| `pyproject.toml` | Python package metadata, supported Python version, core dependencies, and pytest configuration. |
| `uv.lock` | Exact resolved Python dependency lockfile used by the project environment. |
| `src/bhps/` | The reusable numerical library: background construction, constraints, evolution, boundary operators, MOTS solvers, stability operators, continuation support, and validation helpers. |
| `tests/` | The complete project test suite for the reusable library and top-level scientific runners. |
| `*.py` at repository root | Complete experiment runners, analyses, audits, recovery drivers, and finalizers retained in their original import layout so the tests collect without path rewrites. |
| `protocols/` | Frozen Python source, tests, and protocol descriptions for the load-bearing Protocol 226, Protocols 228--232, and marginal-tube Protocols 247--250. Numerical `candidate-output` directories, sealed numerical inputs, and machine-specific authority records are intentionally absent from Git. |
| `article/main.tex` | Main article source. |
| `article/supplement.tex` | Supplemental-material source. |
| `article/references.bib` | Shared bibliography. |
| `article/figures/` | The four publication figures as committed PDF assets. |
| `article/main.pdf` | Current compiled article. |
| `article/supplement.pdf` | Current compiled Supplemental Material. |
| `article/pdfs/article.pdf` | Earlier compiled article retained for exact comparison. |
| `article/pdfs/supplement.pdf` | Earlier compiled supplement retained for exact comparison. |
| `article/pdfs/prd-manuscript.pdf` | Eight-page REVTeX draft prepared for submission to *Physical Review D*. |
| `article/pdfs/prd-supplement.pdf` | Fifteen-page supplement accompanying the PRD draft. |
| `article/pdfs/prd-cover-letter.pdf` | One-page cover letter for the PRD Research Article submission. |
| `article/cover-letter-prd.tex` | Editable source for the PRD cover letter. |
| `article/make_figures.py`, `article/make_tube_figures.py` | Figure-generation sources. They expect the separately archived result records described below. |
| `article/CITATION_AUDIT_2026-08-24.md` | Final bibliography and cross-reference audit. |

### Frozen protocol directories

| Protocol | Purpose |
| --- | --- |
| `protocol226-corrected-canonical-g11-2026-08-23` | Repaired-parent G11 field/tensor comparison and replay controls. |
| `protocol228-repaired-parent-formation-time-2026-08-23` | Direct G9/G10/G11 zero-to-two formation and surface-observable comparison. |
| `protocol229-free-boundary-mots-saddle-node-v4-2026-08-24` | Three-grid pseudo-arclength continuation, critical-mode, nondegeneracy, and square-root-scaling calculation. |
| `protocol230-protocol229-archive-finalization-2026-08-24` | Cross-grid aggregation and final saddle-node classification. |
| `protocol231-saddle-node-existing-data-audit-2026-08-24` | Fit sensitivity and existing-data audit. |
| `protocol232-g10-half-timestep-saddle-node-2026-08-24` | Independent G10 half-timestep parent evolution and continuation comparison. |
| `protocol247-g9-g11-bounded-spatial-transfer-2026-08-25` | Three-grid post-fold outer-tube geometry, stability, and causal-signature transfer. |
| `protocol248-three-grid-native-balance-transfer-2026-08-25` | Three-grid transfer of the brane-inclusive local balance. |
| `protocol249-three-grid-integrated-balance-2026-08-25` | Three-grid and three-stencil finite-segment charge--flux balance. |
| `protocol250-g10-full-half-causal-signature-2026-08-26` | Independent G10 full/half-timestep causal-signature comparison. |

## Code inventory

The current repository contains:

- 225 top-level Python runner, audit, analysis, and finalization files;
- 141 Python files under `src/`;
- 159 Python test files under `tests/`;
- 45 Python files within the frozen protocol directories; and
- two publication figure generators.

That is 572 versioned Python files. All are covered by `MANIFEST.sha256`.
Generated bytecode, virtual environments, caches, numerical outputs, and
candidate result directories are excluded by `.gitignore`.

## Environment and tests

Python 3.11 or newer is required. A conventional environment can be created
with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest matplotlib
```

Collect the complete test surface with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest --collect-only -q -p no:cacheprovider
```

Run the tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -q -p no:cacheprovider
```

Some full scientific runners require the immutable numerical inputs and
candidate records from the Zenodo archive. The code-only checkout collects 978
tests without import errors; tests that authenticate or read omitted archived
records fail closed until those records are restored. Their absence from Git
is intentional and must not be interpreted as missing source code.

## Building the paper

From `article/`, with a TeX Live installation containing `latexmk` and BibTeX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build/main main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build/supplement supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build/cover-letter cover-letter-prd.tex
```

The committed PDFs are provided for exact reading and comparison. The
committed figure PDFs allow the paper to compile without downloading numerical
results. `make_figures.py` is retained as the figure-generation source; to
regenerate the figures, restore the corresponding archived result records in
the paths declared near the top of that script.

## Integrity manifest

Verify the repository payload from its root with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

The manifest intentionally excludes only itself and `.git/`, avoiding a
self-referential digest. Regenerate it only after an intentional release
change.

## Scope

The repository supports the finite-resolution numerical claim made in the
paper. It does not establish a continuum theorem, an event horizon, a global
topology change, a nonsymmetric nonprincipal spectrum, phase selection, or
source ownership.

## Citation

Use the citation metadata in `CITATION.cff` and cite the permanent archive:

> Ronald Bibb, *From a Free-Boundary MOTS Saddle-Node to a Spacelike Marginal
> Tube in Dynamical Five-Dimensional Gravity*, Zenodo,
> [doi:10.5281/zenodo.22148554](https://doi.org/10.5281/zenodo.22148554), 2026.
