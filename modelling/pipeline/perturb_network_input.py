"""Add tonic network-like input to E and re-solve each fitted fixed point."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "core"))
import dmf_rww_ei_jax_gpu_fc_only as base
import dmf_rww_bw_stability_analysis as stability

FIT_DIR = MODEL / "output" / "fit_250"
ANALYSIS = FIT_DIR / "analysis_mechanism_50_excl66"


def neural_rhs_extra(x, sc, params, extra_e):
    n = sc.shape[0]
    se, si = x[:n], x[n:]
    g, wee, wei, wie = params
    ie = .382 + wee * .15 * se + g * .15 * (sc @ se) - wie * si + extra_e
    ii = .7 * .382 + wei * .15 * se - si
    re = stability.h_transfer(ie, 310., 125., .16)
    ri = stability.h_transfer(ii, 615., 177., .087)
    return jnp.concatenate((-se / .1 + (1 - se) * .641 * re,
                            -si / .01 + ri))


def solve_extra(sc, params, extra_e, start):
    scj, pj, uj = map(jnp.asarray, (sc, params, extra_e))
    rhs = lambda x: neural_rhs_extra(x, scj, pj, uj)
    jac = jax.jacfwd(rhs)
    n = sc.shape[0]
    result = least_squares(lambda x: np.asarray(rhs(jnp.asarray(x))), start,
        jac=lambda x: np.asarray(jac(jnp.asarray(x))),
        bounds=(np.full(2*n, 1e-9), np.r_[np.full(n, 1-1e-9), np.full(n, 5-1e-9)]),
        xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=100)
    residual = float(np.linalg.norm(np.asarray(rhs(jnp.asarray(result.x)))) / np.sqrt(2*n))
    if not result.success or residual > 1e-5:
        raise RuntimeError(f"Fixed-point solve failed: {result.message}; residual={residual}")
    return result.x, np.asarray(jac(jnp.asarray(result.x))), residual


def metrics(state, jac, sc, params, extra_e, motor):
    n = sc.shape[0]
    se, si = state[:n], state[n:]
    g, _, wei, wie = params
    de = -np.diag(jac[:n, n:]) / wie
    di = -np.diag(jac[n:, n:]) - 100.
    ii = .7 * .382 + wei * .15 * se - si
    r_i = np.asarray(stability.h_transfer(jnp.asarray(ii), 615., 177., .087))
    net = g * .15 * (sc @ se) + extra_e
    ix = np.r_[motor, n + motor]
    margin = -float(np.max(np.linalg.eigvals(jac[np.ix_(ix, ix)]).real))
    eie = de * wie * di * wei * .15 / (100. + di)
    return dict(network_input=float(net[motor].mean()), S_E=float(se[motor].mean()),
                S_I=float(si[motor].mean()), I_I=float(ii[motor].mean()),
                r_I=float(r_i[motor].mean()), D_E=float(de[motor].mean()),
                D_I=float(di[motor].mean()), EIE_restoring=float(eie[motor].mean()),
                motor_stability=margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, default=FIT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--fraction", type=float, default=.001)
    parser.add_argument("--max-fits", type=int)
    args = parser.parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    left, right = base.default_motor_rois(data["roi_labels"])
    motor = np.asarray(sorted(set(left + right)), int)
    sc = base.prepare_sc(base.sanitize_matrix(base.finite_nanmean(data["SC"][:, 0], axis=0)))
    with (args.analysis_dir / "paired_changes.csv").open() as handle:
        valid = {int(row["candidate_id"]) for row in csv.DictReader(handle)}
    paths = [p for p in sorted(args.fit_dir.glob("selected_???.json"))
             if json.loads(p.read_text())["candidate_id"] in valid]
    if args.max_fits:
        paths = paths[:args.max_fits]
    rows = []
    for path in paths:
        record = json.loads(path.read_text())
        cid = int(record["candidate_id"])
        params = np.asarray(record["results"]["baseline"]["params"], float)
        summary = np.asarray(record["results"]["baseline"].get("summary", []), float)
        x0 = np.r_[summary[0], summary[1]] if summary.ndim == 2 and summary.shape[0] >= 2 else None
        baseline, residual, success, msg = stability.solve_fixed_point(
            sc, params, .382, x0=x0, tol=1e-8, max_nfev=80)
        if not success or residual > 1e-5:
            raise RuntimeError(f"Baseline solve failed for candidate {cid}: {msg}; residual={residual}")
        jac0 = stability.jacobian_np(baseline, sc, params, .382)
        network_current = params[0] * .15 * (sc @ baseline[:len(sc)])
        extra = args.fraction * network_current
        perturbed, jac1, residual1 = solve_extra(sc, params, extra, baseline)
        old = metrics(baseline, jac0, sc, params, np.zeros(len(sc)), motor)
        new = metrics(perturbed, jac1, sc, params, extra, motor)
        row = dict(candidate_id=cid, fraction=args.fraction,
                   baseline_residual=residual, perturbed_residual=residual1)
        row.update({f"baseline_{k}": v for k, v in old.items()})
        row.update({f"perturbed_{k}": v for k, v in new.items()})
        row.update({f"delta_{k}": new[k] - old[k] for k in old})
        rows.append(row)
        print(f"candidate {cid}: Δinput={row['delta_network_input']:+.5g}, "
              f"ΔD_I={row['delta_D_I']:+.5g}, Δstability={row['delta_motor_stability']:+.5g}", flush=True)
    suffix = f"{args.fraction*100:g}".replace(".", "p") + "pct"
    out = args.analysis_dir / f"network_input_DI_closed_loop_{suffix}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
