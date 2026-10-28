"""Evaluate the fixed-point gain and feedback terms in the local stability derivation."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eig

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "core"))
import dmf_rww_ei_jax_gpu_fc_only as base
import dmf_rww_bw_stability_analysis as stability

J_NMDA = 0.15
GAMMA_E = 0.641
TAU_E_INV = 10.0
TAU_I_INV = 100.0


def leading_mode_contributions(jac, terms):
    values, left, right = eig(jac, left=True, right=True)
    k = int(np.argmax(values.real))
    u, v = left[:, k], right[:, k]
    overlap = np.vdot(u, v)
    contributions = {name: float(np.real(np.vdot(u, matrix @ v) / overlap))
                     for name, matrix in terms.items()}
    if abs(sum(contributions.values()) - values[k].real) > 1e-3:
        raise ValueError("The projected Jacobian terms do not sum to the leading eigenvalue")
    return float(values[k].real), contributions


def mechanism_row(record, session, sc, motor, state, jac, residual):
    n = sc.shape[0]
    g, w_ee, w_ei, w_ie = np.asarray(record["results"][session]["params"], float)
    summary = stability.neural_summary(state, sc, (g, w_ee, w_ei, w_ie), .382)
    s_e, s_i, i_e, r_e = (np.asarray(summary[k], float)
                          for k in ("S_E", "S_I", "I_E", "r_E"))
    a_ei = jac[:n, n:]
    a_ii = jac[n:, n:]
    d_e = -np.diag(a_ei) / w_ie
    d_i = -np.diag(a_ii) - TAU_I_INV
    r_e_term = GAMMA_E * r_e
    hprime_e = d_e / (GAMMA_E * np.maximum(1.0 - s_e, 1e-9))
    loop = d_e * w_ie * d_i * w_ei * J_NMDA / (TAU_I_INV + d_i)
    recurrent = d_e * w_ee * J_NMDA
    network_operator = (d_e * g * J_NMDA)[:, None] * sc
    tonic_network = g * J_NMDA * (sc @ s_e)
    midx = np.asarray(motor, int)
    all_idx = np.concatenate([midx, n + midx])
    motor_jac = jac[np.ix_(all_idx, all_idx)]
    m = len(midx)
    terms = {}
    for name in ("E_decay", "R_E", "recurrent", "network", "E_from_I",
                 "I_from_E", "I_decay", "I_gain"):
        terms[name] = np.zeros_like(motor_jac)
    terms["E_decay"][:m, :m] = -TAU_E_INV * np.eye(m)
    terms["R_E"][:m, :m] = -np.diag(r_e_term[midx])
    terms["recurrent"][:m, :m] = np.diag(recurrent[midx])
    terms["network"][:m, :m] = network_operator[np.ix_(midx, midx)]
    terms["E_from_I"][:m, m:] = a_ei[np.ix_(midx, midx)]
    terms["I_from_E"][m:, :m] = jac[n:, :n][np.ix_(midx, midx)]
    terms["I_decay"][m:, m:] = -TAU_I_INV * np.eye(m)
    terms["I_gain"][m:, m:] = -np.diag(d_i[midx])
    decomposition_error = float(np.max(np.abs(motor_jac - sum(terms.values()))))
    if decomposition_error > 1e-3:
        raise ValueError(f"Jacobian decomposition mismatch: {decomposition_error}")
    lead_lambda, contributions = leading_mode_contributions(motor_jac, terms)
    all_mask = np.ones(n, bool)
    all_mask[midx] = False
    row = dict(candidate_id=record["candidate_id"], selected_id=record["selected_id"],
        seed=record["seed"], session=session, fixed_point_residual=residual,
        motor_stability=-lead_lambda,
        mean_I_E=float(i_e[midx].mean()), mean_r_E=float(r_e[midx].mean()),
        mean_S_E=float(s_e[midx].mean()), mean_S_I=float(s_i[midx].mean()),
        mean_Hprime_E=float(hprime_e[midx].mean()),
        mean_D_E=float(d_e[midx].mean()), mean_R_E=float(r_e_term[midx].mean()),
        mean_D_I=float(d_i[midx].mean()),
        recurrent_gain=float(recurrent[midx].mean()),
        inhibitory_loop_restoring=float(loop[midx].mean()),
        tonic_network_input=float(tonic_network[midx].mean()),
        tonic_inhibitory_current=float((w_ie * s_i[midx]).mean()),
        network_motor_row_gain=float(network_operator[np.ix_(midx, midx)].sum(axis=1).mean()),
        network_external_row_gain=float(network_operator[np.ix_(midx, np.where(all_mask)[0])].sum(axis=1).mean()),
        full_jacobian_stability=-float(np.max(np.linalg.eigvals(jac).real)),
        jacobian_decomposition_error=decomposition_error)
    row.update({f"mode_{name}": value for name, value in contributions.items()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-candidate", type=int, action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input_npz, allow_pickle=True)
    left, right = base.default_motor_rois(data["roi_labels"])
    motor = sorted(set(left + right))
    sc = [base.prepare_sc(base.sanitize_matrix(
        base.finite_nanmean(data["SC"][:, i], axis=0))) for i in range(2)]
    selected_paths = [path for path in sorted(args.fit_dir.glob("selected_???.json"))
                      if json.loads(path.read_text())["candidate_id"]
                      not in args.exclude_candidate]
    if args.max_candidates is not None:
        selected_paths = selected_paths[:args.max_candidates]
    for path in selected_paths:
        record = json.loads(path.read_text())
        cid = int(record["candidate_id"])
        if cid in args.exclude_candidate:
            continue
        for i, session in enumerate(("baseline", "followup6w")):
            destination = args.output_dir / f"candidate_{cid:04d}_{session}.json"
            if destination.exists():
                continue
            session_record = record["results"][session]
            params = np.asarray(session_record["params"], float)
            sim_summary = np.asarray(session_record.get("summary", []), float)
            x0 = (np.concatenate([sim_summary[0], sim_summary[1]])
                  if sim_summary.ndim == 2 and sim_summary.shape[0] >= 2 else None)
            state, residual, success, _ = stability.solve_fixed_point(
                sc[i], params, .382, x0=x0, tol=1e-8, max_nfev=80)
            if not success or residual > 1e-5:
                raise ValueError(f"Fixed point failed for candidate {cid} {session}: {residual}")
            jac = stability.jacobian_np(state, sc[i], params, .382)
            row = mechanism_row(record, session, sc[i], motor, state, jac, residual)
            destination.write_text(json.dumps(row, indent=2))
        print(f"finished candidate {cid}", flush=True)
    rows = [json.loads(path.read_text()) for path in sorted(
        args.output_dir.glob("candidate_*.json"))]
    if len(rows) != 2 * len(selected_paths):
        raise ValueError("Unexpected number of session rows")
    with (args.output_dir / "session_terms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([sys.executable,
        str(Path(__file__).with_name("summarize_feedback_paths.py")),
        "--analysis-dir", str(args.output_dir)], check=True, cwd=MODEL.parent)


if __name__ == "__main__":
    main()
