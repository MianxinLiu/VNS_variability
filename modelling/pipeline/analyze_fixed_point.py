#!/usr/bin/env python3
"""Fixed-point stability margins for the selected FC>0.15 fits."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "core"))
import dmf_rww_ei_jax_gpu_fc_only as base  # noqa: E402
import dmf_rww_bw_stability_analysis as stability  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = np.load(args.input_npz, allow_pickle=True)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left, right = base.default_motor_rois(labels)
    sc = [base.prepare_sc(base.sanitize_matrix(base.finite_nanmean(data["SC"][:, j], axis=0)))
          for j in range(2)]
    rows = []
    for fit_path in sorted(args.fit_dir.glob("selected_???.json")):
        record = json.loads(fit_path.read_text())
        if args.candidate_ids is not None and int(record["candidate_id"]) not in args.candidate_ids:
            continue
        for session_index, session in enumerate(("baseline", "followup6w")):
            session_record = record["results"][session]
            params = np.asarray(session_record["params"], float)
            summary = np.asarray(session_record.get("summary", []), float)
            x0 = (np.concatenate([summary[0], summary[1]])
                  if summary.ndim == 2 and summary.shape[0] >= 2 else None)
            root, residual, success, _ = stability.solve_fixed_point(
                sc[session_index], params, .382, x0=x0, tol=1e-8, max_nfev=80)
            jac = stability.jacobian_np(root, sc[session_index], params, .382)
            metric = stability.stability_metrics(
                jac, left, right, ["R", "R", "L", "R", "L", "L", "R"])
            rows.append(dict(selected_id=record["selected_id"],
                             candidate_id=record["candidate_id"], session=session,
                             fixed_point_success=success,
                             fixed_point_residual_rms=residual,
                             stability_margin=metric["stability_margin"],
                             max_real_lambda=metric["max_real_lambda"],
                             local_motor_block_max_real_lambda=metric["local_motor_block_max_real_lambda"],
                             local_motor_stability_margin=-metric["local_motor_block_max_real_lambda"]))
        print(f"finished selected={record['selected_id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
