# Ground-motion correlation evaluation

Research notebooks and Python utilities for evaluating observational ground-motion spatial correlations, stationary/non-stationary spatial correlation models (SCMs), regional variability, and predictive performance. Initial studies use the New Zealand Ground-Motion Database v4.3.

## Repository layout

```text
groundmotion-correlation-evaluation/
├── notebooks/
│   ├── PHD_OBJ2_GMMRES_V1_1.ipynb
│   ├── PHD_OBJ2_PYMC_V1_1.ipynb
│   ├── PHD_OBJ2_NZSCM_V1_5.ipynb
│   ├── PHD_OBJ2_ESUP_GMMSEL_V1_1.ipynb
│   └── PHD_OBJ2_ESUP_MODSEL_v1_2.ipynb
├── src/
│   ├── cbb21_cossim.py
│   ├── cbb21_gmres.py
│   └── cbb21_gmscm.py
├── data/
│   ├── residuals/
│   └── nzwide_scm/
├── results/
├── figures/
└── requirements.txt
```

## Notebooks

| Notebook | Role |
| --- | --- |
| `PHD_OBJ2_GMMRES_V1_1.ipynb` | GMM predictions and normalized within-event residual construction. |
| `PHD_OBJ2_PYMC_V1_1.ipynb` | Bayesian SCM fitting and predictive evaluation. |
| `PHD_OBJ2_NZSCM_V1_5.ipynb` | Empirical NZ correlations, SCM comparison, regional evaluations, figures. |
| `PHD_OBJ2_ESUP_GMMSEL_V1_1.ipynb` | Supplement A: GMM sensitivity analysis. |
| `PHD_OBJ2_ESUP_MODSEL_v1_2.ipynb` | Supplement C: model/feature selection and auxiliary figures. |

The original notebook filenames, cell order and model calculations are retained. Saved execution outputs were cleared to keep the repository compact; computed results must be regenerated or provided as CSV files.

## Getting started

Create a Python environment and install the general dependencies:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

The first code cell of each notebook locates this repository automatically from the current Jupyter working directory, adds `src/` to `sys.path`, and defines `RESIDUALS_DIR`, `SCM_DIR`, `RESULTS_DIR`, and `FIGURES_DIR`. This works when Jupyter starts in the repository root or in `notebooks/`. The notebooks are scripts for research reproduction, not independent tutorials: some require outputs from earlier analyses.

**Additional requirements:** `PHD_OBJ2_PYMC_V1_1.ipynb` needs compatible JAX (64-bit enabled in the notebook), NumPyro, ArviZ and xarray. `PHD_OBJ2_GMMRES_V1_1.ipynb` also requires a compatible OpenQuake hazardlib version. `PHD_OBJ2_NZSCM_V1_5.ipynb` imports the external `VarioCorreKrigE` package. Ensure these are available in the active kernel.

## Data locations and provenance

`data/residuals/` contains source ground-motion/site tables and derived GMM residuals. `data/nzwide_scm/` contains SCM fitting outputs, parameter summaries, LOO/LPPD diagnostics, and optional examples. See each directory's README for the required filenames and expected optional inputs. These data were not provided with the code and have not been included in this archive.

The Git ignore rules prevent accidental upload of observational datasets and computed outputs. If particular small derived datasets can be shared under their respective licences, add them explicitly using `git add -f` after validating rights and provenance.

## Scope of this cleanup

- Original computational functions and notebook analyses were retained.
- Removed the large inactive duplicate `cossim` and `ay92` code blocks in module comments.
- Renamed `cbb21_gmscm(2).py` to `cbb21_gmscm.py` and made its default parameter-file reads lazy, so module imports work before data are available.
- Updated Windows-specific absolute paths to the two requested `data/` locations; output plots are written to `figures/`.
- Corrected two mistaken Supplement A K20/P20 LPPDP input filenames.
- Preserved station identifiers as strings when loading the principal observational CSVs.
- Cleared historical notebook outputs and execution counts.

Scientific kernels, Bayesian priors, the MCMC setup, and existing data filters were intentionally not refactored. The repository has been **syntax-checked and smoke-tested only**, not rerun against the absent NZGMDB data.

## Referencing

When the corresponding manuscript is published, add the citation, DOI, and versioned release information here. The modules preserve the published-SCM abbreviations and function names used by the research notebooks.
