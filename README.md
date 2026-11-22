# paper-companion-pygamlab-v2

This repository provides the reproducibility, benchmark-testing, and supplementary-information companion for the paper **"PyGamLab V2: A Unified Computational and End-to-End Artificial Intelligence Platform for Nanotechnology and Materials Informatics"** by **Ali Pilehvar Meibody et al.**

The repository is designed to help readers reproduce the principal benchmark results reported for PyGamLab V2, inspect the packaged `.gam_ai` models distributed with the library, regenerate representative figures, and review supporting artifacts used in the manuscript. The workflows here focus on transparent comparison against established reference implementations, including scikit-learn, SciPy, and ASE, as well as on documenting the packaged AI models distributed through PyGamLab.

All benchmark datasets used here are stored locally inside the repository folders so that the scripts are directly runnable without editing external file paths. Generated outputs are written into the corresponding `outputs/`, `figures/`, or `models/` directories.

## Benchmarks

The repository contains three benchmark folders that compare existing functionality in the PyGamLab library against standard scientific or machine-learning reference implementations.

### `ai_workflow_benchmark`

This folder benchmarks the PyGamLab regression workflow against an equivalent scikit-learn implementation on the `chemical_congress.xlsx` dataset. The script applies the same preprocessing used in the chemical-congress workflow: two interquartile-range filters on solubility, followed by the `abs(log(solubility))` target transform. It then compares PyGamLab and scikit-learn for k-nearest neighbours, random forest, linear regression, decision tree, support-vector regression, and multilayer perceptron models.

Outputs include `results.csv`, `results.json`, and `summary.txt` inside `ai_workflow_benchmark/outputs/`. These files report predictive metrics and execution times for both implementations.

Run it with:

```bash
python ai_workflow_benchmark/compare_workflow.py
```

### `data_analysis_benchmark`

This folder benchmarks PyGamLab data-analysis functions against independent SciPy or NumPy reference calculations using four laboratory datasets: DSC, photoluminescence, NMR, and AES. The script evaluates transition temperatures, peak locations, kinetic onset values, optical band-gap calculations, NMR composition/impurity estimates, and AES composition and peak-detection outputs.

Outputs include `results.csv`, `results.json`, and `summary.txt` inside `data_analysis_benchmark/outputs/`. These summarize numerical agreement and timing differences between PyGamLab and the reference calculations.

Run it with:

```bash
python data_analysis_benchmark/compare_analysis.py
```

### `structures_benchmark`

This folder benchmarks PyGamLab structure-generation tools against ASE references for graphene, silicene, carbon nanotubes, and gold nanoparticles. The script compares atomic positions, nearest-neighbour bond lengths, atom counts, and generation times. It also writes paired `.xyz` files for the generated PyGamLab and ASE structures.

Outputs include `results.csv`, `results.json`, `summary.txt`, and `.xyz` structures in `structures_benchmark/outputs/`.

Run it with:

```bash
python structures_benchmark/compare_structures.py
```

## Supplementary Workflows

### `gam_ai_protocol`

This folder documents how the `.gam_ai` models discussed in the paper are trained and packaged. The protocol follows the original PyGamLab workflow style and uses PyGamLab AutoML components to train models and save them through the `GamAI_io` interface. For demonstration and reproducibility, this repository includes one concrete example: the copper-nanocomposite coefficient-of-thermal-expansion dataset `cte_cu_nanocomposite.xlsx`.

The script trains both:

- train/test models using `BASIC_regression`
- pipeline/grid-search models using `regression_PIPE`

The generated packaged models are written to `gam_ai_protocol/models/`.

Run it with:

```bash
python gam_ai_protocol/create_cte_gam_ai.py
```

### `gam_ai_details`

This folder provides a documentation-oriented inventory of all packaged `.gam_ai` models copied from `pygamlab/pygamlab/ai_core/gam_models`. The extraction script uses the PyGamLab `GAM_AI_MODEL` class to load each package exactly as the library does, then records its metadata and estimator details for documentation.

The folder contains:

- `models/`: local copies of the packaged `.gam_ai` models
- `outputs/model_details.json`: machine-readable catalog
- `outputs/model_details.md`: reader-friendly catalog

Run it with:

```bash
python gam_ai_details/extract_gam_ai_details.py
```

### `figure6_creation`

This folder recreates the chemical-congress KNN workflow used for figure generation. It performs exploratory analysis on the chemical-congress dataset, applies the same outlier filtering and target transformation used in the benchmark workflow, and then generates a professional true-versus-predicted plot and 3D prediction maps.

Generated figures are written to `figure6_creation/figures/`.

Run it with:

```bash
python figure6_creation/create_figure6.py
```

## Simple CLI

For convenience, this repository includes a small command-line interface, `companion_cli.py`, that lets readers run any workflow from one entry point.

List available tasks:

```bash
python companion_cli.py list
```

Run one task:

```bash
python companion_cli.py run ai-workflow
python companion_cli.py run data-analysis
python companion_cli.py run structures
python companion_cli.py run gam-ai-protocol
python companion_cli.py run gam-ai-details
python companion_cli.py run figure6
```

Run grouped tasks:

```bash
python companion_cli.py run benchmarks
python companion_cli.py run supplementary
python companion_cli.py run all
```

If a reader wants to run the workflows against a local source checkout of PyGamLab rather than an installed package, they can provide a source path:

```bash
python companion_cli.py run figure6 --pygamlab-source /path/to/pygamlab
```

## Repository Layout

```text
ai_workflow_benchmark/   scikit-learn comparison for the PyGamLab regression workflow
data_analysis_benchmark/ SciPy and NumPy comparison for laboratory analysis functions
structures_benchmark/    ASE comparison for structure-generation functions
gam_ai_protocol/         representative .gam_ai training and packaging workflow
gam_ai_details/          full metadata inventory of packaged .gam_ai models
figure6_creation/        figure-generation workflow for the chemical-congress example
companion_cli.py         single entry point for running repository workflows
```

## Notes

- The benchmark and supplementary scripts use `matplotlib` in non-interactive mode and save outputs to disk.
- Some workflows depend on the exact PyGamLab version available in the Python environment. The `--pygamlab-source` option in the CLI is useful when readers want to force execution against a local PyGamLab source tree.
- The packaged `.gam_ai` files may emit scikit-learn version warnings if they are loaded with a different scikit-learn version than the one originally used for training. This is expected when documenting archived model artifacts.
