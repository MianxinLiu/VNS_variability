# Main-figure model analysis

`core/` contains the reduced Wong–Wang neural/Balloon–Windkessel simulation, spectral-exponent calculation, and fixed-point/Jacobian equations. `pipeline/` contains fitting, neural metric calculation (SE, DFA, MSE), current decomposition, stability analysis, feedback-path analysis, and the small network-input perturbation. Figure layout and rendering are in `../plotting/`.

The main analysis uses 250 BOLD time points at TR = 0.7 s, a 1 ms neural step, a 10 ms hemodynamic step, and fixed background input I0 = 0.382 nA. The fitting run collected 51 accepted solutions; candidate 66 was excluded from the displayed set of 50. The seven small tables in `../data/model/` contain the fit-level values needed to reproduce manuscript Figures 5 and 6. Raw participant SC/FC arrays and selected fit JSON files are not included.

From `github_new/`, regenerate the released figures with:

```bash
python plotting/plot_figure5_model_results.py
python plotting/plot_figure6_stability_mechanism.py
```

A new fit requires a user-supplied NPZ with `roi_labels`, `SC`, and `FC`, plus a GPU JAX environment. The core analysis entry points are:

```bash
python modelling/pipeline/fit_250.py --input-npz /path/to/input.npz --target-count 51
VNS_MODEL_INPUT_NPZ=/path/to/input.npz python modelling/pipeline/analyze_mse.py
python modelling/pipeline/summarize_fits.py --input-npz /path/to/input.npz --count 50 --exclude-candidate 66 --analysis-name analysis_50_excl66
python modelling/pipeline/analyze_feedback_paths.py --help
python modelling/pipeline/perturb_network_input.py --help
```

The released tables and figure scripts were checked against the current main figures. The full GPU refit was not rerun as part of this release.
