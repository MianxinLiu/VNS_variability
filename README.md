# VNS manuscript analysis release

This directory contains analysis code and the compact data needed to reproduce the current manuscript main Figures 2–6.

## Contents

- `EEG_analysis/`: EEG preprocessing and feature-extraction utilities.
- `data/cohort1/`, `data/cohort2/`: anonymized EEG feature arrays.
- `data/model/`: compact 250-point model results for 50 selected solutions.
- `modelling/`: simulation, model fitting, equations, and metric calculations; see [modelling/README.md](modelling/README.md).
- `plotting/plot_figure5_model_results.py`: current main Figure 5.
- `plotting/plot_figure6_stability_mechanism.py`: current main Figure 6.
- `plotting/plot_figureS3_baseline_adjusted_sensitivity.py`: supplementary Figure S3.
- `plotting/plot_figures_2_3_4_S3.py`: entry point for Figures 2–4 and S3.
- `figures/`: rendered figures and numerical result tables.

## Figure mapping

- Figure 2: Cohort 1 longitudinal SE, DFA, and MSE trajectories.
- Figure 3: Cohort 2 replication and pooled analyses.
- Figure 4: variability–FMA associations and longitudinal mixed models.
- Figure 5: 250-point whole-brain model fits and motor-state changes.
- Figure 6: network-input, inhibitory-feedback, and stability mechanism.
- Figure S3: baseline EEG and six-week outcome sensitivity analysis.

The internal key `ipsi` refers to the motor region contralateral to the lesion; `contra` refers to the ipsilesional motor region.

## Reproduction

Run from `github_new` after creating the environment in `environment.yml`:

```bash
python plotting/plot_figures_2_3_4_S3.py
python plotting/plot_figure5_model_results.py
python plotting/plot_figure6_stability_mechanism.py
```

The first command also writes the baseline-adjusted sensitivity analysis as Figure S3. The current main Figure 6 is produced by `plot_figure6_stability_mechanism.py`. Outputs are written to `figures/`.
