"""Small helpers used by the 250-point fitting routine."""
import json
from pathlib import Path
from types import SimpleNamespace


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def model_args(args, optimization_id: int) -> SimpleNamespace:
    return SimpleNamespace(optimizer="forest", n_calls=args.n_calls,
        n_initial_points=args.n_initial_points, n_candidates=4, n_tr=args.fit_n_tr,
        dt=args.dt, tr=args.tr, burn_s=args.burn_s,
        summary_start_s=args.burn_s, seed=args.fit_seed_base + 10000*optimization_id,
        random_state=args.random_state_base + 10000*optimization_id,
        fixed_sigma=args.fixed_sigma, fixed_i0=args.fixed_i0,
        bold_input=args.bold_input, g_bounds=[0.05,5.0],
        w_ee_bounds=[0.5,2.5], w_ei_bounds=[0.5,2.5],
        w_ie_bounds=[0.2,10.0], bad_loss=3.0,
        subject_sides=args.subject_sides, quiet=True)
