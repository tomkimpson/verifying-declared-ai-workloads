# Verifying Declared AI Workloads from GPU Supply Current

Code, results and paper for *Verifying Declared AI Workloads from GPU Supply Current*
(Tom Kimpson, 2026). The paper asks whether an external observer can tell that a GPU
runs only the work its operator declared, using a clamp-on probe on the GPU's supply
current. A one-class model learns only from recordings of the declared work.

- **Paper:** [`paper/main.pdf`](paper/main.pdf), with its LaTeX source in `paper/`.
- **Data:** [Zenodo, doi:10.5281/zenodo.23521636](https://doi.org/10.5281/zenodo.23521636)
  (41 GB, CC BY 4.0).

The probe is an AC-coupled Rogowski coil. It measures **current, in amperes**, and has
no DC response, so nothing here is power or energy.

## Contents

| Path | What |
|---|---|
| `paper/` | LaTeX source and the built PDF. Every number in the text is a macro in `numbers.tex`, generated from `numbers.toml` |
| `scripts/` | Entry points: one per result, figure and table |
| `code/` | The library the scripts import |
| `results/` | Every result the paper uses, as committed JSON |
| `figures/` | Generated figures and LaTeX tables |
| `tests/` | `pytest` checks of the library and the corpus |
| `spec/preregistration.md` | The pre-registered held-out analysis |
| `vendor/amodo/` | The feature code and three results from the corpus repository, unmodified, at commit `73ffebe` |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.13
```

The paper needs a TeX distribution with `latexmk`.

## Reproduce the paper from the committed results (no data needed)

These scripts read only the JSON in `results/`. Run `plot_f8_monitors.py` first: it
refuses to run on a tree with uncommitted changes, and the others rewrite tracked files.

```bash
.venv/bin/python scripts/plot_f8_monitors.py       # Fig. 1
.venv/bin/python scripts/plot_f10_reference.py     # Fig. 2
.venv/bin/python scripts/plot_f9_contamination.py  # Fig. 3
.venv/bin/python scripts/plot_f7_r1.py             # Fig. A1
.venv/bin/python scripts/table_t7.py               # Table 1
.venv/bin/python scripts/table_t2.py               # Table 2
.venv/bin/python scripts/table_t6.py               # Table 3
.venv/bin/python scripts/table_corpus.py           # Table A1
.venv/bin/python scripts/table_t5.py               # Table A2
.venv/bin/python scripts/paper_numbers.py          # paper/numbers.tex from paper/numbers.toml
.venv/bin/python scripts/check_numbers.py          # fails on a stale macro or a stray digit
cd paper && latexmk -pdf main.tex
```

PDFs can differ in their bytes from the committed ones when fonts or the matplotlib
version differ. The drawn values are also saved in the JSON beside each figure.

## Reproduce the results from the data

**1. Get the data.** Download every file in the Zenodo record into one directory
(`$D` below). Reassemble the two split tars, then extract everything into an empty
directory:

```bash
mkdir amodo-current && cd amodo-current
for n in panel negative; do cat "$D"/results-$n.tar.part?? > "$D"/results-$n.tar; done
for f in "$D"/*.tar; do tar xf "$f"; done
sha256sum -c dataset/SHA256SUMS                     # optional: checks every file
export AMODO_ARCHIVE="$PWD"
```

**2. Run the pipeline in order.** Every script that writes a result refuses to run on a
tree with uncommitted changes, and records the commit it ran at. Commit each output
before running the next script.

```bash
.venv/bin/python scripts/make_windows.py           # window features, results/windows/ (~30 min, not committed)
.venv/bin/python -m pytest tests                   # library and corpus checks
.venv/bin/python scripts/diagnose_5hz_line.py      # results/line5hz/
.venv/bin/python scripts/declaration_heldout.py    # results/declaration/
.venv/bin/python scripts/declaration_choose.py
.venv/bin/python scripts/declaration_killcheck.py
.venv/bin/python scripts/h1_coverage.py            # results/coverage/
.venv/bin/python scripts/h1_mechanism.py
.venv/bin/python scripts/h1_5hz_stage_a.py
.venv/bin/python scripts/h1_5hz_stage_b.py
.venv/bin/python scripts/h2_thresholds.py          # results/heldout/thresholds.json; commit before the rest
.venv/bin/python scripts/h2_compare.py
.venv/bin/python scripts/h3_false_alarms.py        # results/monitor/
.venv/bin/python scripts/h3_episodes.py
.venv/bin/python scripts/t4_robustness.py          # results/robustness/
.venv/bin/python scripts/s2_threshold.py           # results/ceiling/; commit before s2_ceiling
.venv/bin/python scripts/s2_ceiling.py
.venv/bin/python scripts/h2_operating_point.py     # results/heldout/operating_point.json
.venv/bin/python scripts/e3_r1_scores.py           # results/heldout/r1_scores.json
.venv/bin/python scripts/monitor_traces.py         # results/monitor/traces.json
.venv/bin/python scripts/e3_calibration_breakdown.py
.venv/bin/python scripts/e4_fit_contamination.py
.venv/bin/python scripts/limits_ac_rms.py
.venv/bin/python scripts/session_check.py
.venv/bin/python scripts/mechanism_cv_split.py
.venv/bin/python scripts/table_t3.py               # results/replication/t3.json, from vendor/amodo
```

**3. The cross-check against Gargiulo & Kulp's dataset** needs the network. The fetch
streams 1,610 traces (about 6 GB after decimation, about 4 h with 3 workers):

```bash
.venv/bin/python scripts/fetch_gk.py --workers 3 --out ~/data/gk-200kSs \
    --manifest ~/data/gk-200kSs/fetch_manifest.jsonl
.venv/bin/python scripts/gk_features.py --root ~/data/gk-200kSs
.venv/bin/python scripts/gk_own_check.py
.venv/bin/python scripts/t5_gk_transfer.py
```

Then rebuild the paper as in the section above.

## Notes on provenance

- **Splits** are fixed by the dataset (`dataset/splits/`); the test split is the
  session of 2026-09-24. Scripts read test-split windows only through a guard that
  checks the SHA-256 of `spec/preregistration.md`, so the analysis on that split is
  the one pre-registered there.
- **`repo_commit` fields** in the result JSONs refer to the history of the development
  repository, which is not public. This repository is a single-commit snapshot of
  that work.
- **`vendor/amodo/`** holds the feature definitions (`code/features.py`, `code/channel/`)
  and three results from the corpus's own repository at commit `73ffebe`, unmodified.
  `code/corpus.py` checks them against `vendor/amodo/SHA256SUMS` on import.
- **`results/sprint/`** is the committed result of an exploratory sprint that ran
  before the pre-registration. `pytest` reproduces it as a consistency check; the
  sprint scripts are not included.

## Citation

```bibtex
@misc{kimpson2026declared,
  author = {Kimpson, Tom},
  title  = {Verifying Declared {AI} Workloads from {GPU} Supply Current},
  year   = {2026},
  url    = {https://github.com/tomkimpson/verifying-declared-ai-workloads}
}

@misc{kimpson2026corpus,
  author       = {Kimpson, Tom},
  title        = {{NVIDIA} {H200} {NVL} current traces of {GPU} training, inference and non-{AI} workloads, with training episodes hidden in declared serving},
  year         = {2026},
  howpublished = {Zenodo dataset, version 1.0.0},
  doi          = {10.5281/zenodo.23521636}
}
```

## Licence

The code, results and paper in this repository are released under the MIT licence
(see `LICENSE`). The dataset on Zenodo is CC BY 4.0.
