"""Summarize paired operating-point and Jacobian terms from stability-path analysis."""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

VARIABLES = ("mean_I_E", "mean_r_E", "mean_S_E", "mean_Hprime_E",
    "mean_D_E", "mean_R_E", "mean_D_I", "recurrent_gain",
    "inhibitory_loop_restoring", "tonic_network_input",
    "tonic_inhibitory_current", "network_motor_row_gain",
    "network_external_row_gain", "mode_E_decay", "mode_R_E",
    "mode_recurrent", "mode_network", "mode_E_from_I", "mode_I_from_E",
    "mode_I_decay", "mode_I_gain", "motor_stability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.analysis_dir / "session_terms.csv")
    delta = frame.pivot(index="candidate_id", columns="session", values=list(VARIABLES))
    delta = pd.DataFrame({name: delta[(name, "followup6w")] - delta[(name, "baseline")]
                          for name in VARIABLES})
    delta["mode_inhibitory_system"] = (delta["mode_E_from_I"] +
        delta["mode_I_from_E"] + delta["mode_I_decay"] + delta["mode_I_gain"])
    delta["mode_excitatory_local"] = (delta["mode_E_decay"] +
        delta["mode_R_E"] + delta["mode_recurrent"])
    mode_balance = (delta["mode_inhibitory_system"] +
        delta["mode_excitatory_local"] + delta["mode_network"] +
        delta["motor_stability"])
    (args.analysis_dir / "decomposition_check.json").write_text(json.dumps({
        "max_absolute_mode_balance_error": float(mode_balance.abs().max()),
        "meaning": "inhibitory + excitatory-local + network + stability change = 0"
    }, indent=2))
    delta.to_csv(args.analysis_dir / "paired_changes.csv")
    target = delta["motor_stability"]
    summary = []
    for name in (*VARIABLES, "mode_inhibitory_system", "mode_excitatory_local"):
        values = delta[name]
        rho, rho_p = spearmanr(values, target) if name != "motor_stability" else (1., 0.)
        summary.append(dict(term=name, n=len(values), mean_delta=float(values.mean()),
            median_delta=float(values.median()), positive=int((values > 0).sum()),
            negative=int((values < 0).sum()),
            signed_rank_p_two_sided=float(wilcoxon(values).pvalue),
            rho_with_motor_stability=float(rho), rho_p_two_sided=float(rho_p)))
    table = pd.DataFrame(summary)
    table.to_csv(args.analysis_dir / "term_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
