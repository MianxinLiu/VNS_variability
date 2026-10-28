"""Matched-seed MSE audit of all selected fits, without refitting."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import json
import io
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("MULTIRATE_FIT_DIR",
    str(MODEL / "output" / "fit_250")))
OUT = SOURCE / "mse_audit"


def worker(source, output):
    import EntropyHub as EH
    from scipy.signal import butter, sosfiltfilt
    signals = np.load(source)["S_E"]
    sos = butter(4, [.5, 80], btype="bandpass", fs=1000, output="sos")
    estimator = EH.MSobject("DispEn")
    curves = []
    for signal in signals:
        filtered = sosfiltfilt(sos, signal.astype(float))
        # EEG epochs contain both endpoints: 2--87 s at 1000 Hz.
        length = 85001
        chunks = []
        for start in range(0, len(filtered)-length+1, length):
            with redirect_stdout(io.StringIO()):
                entropy, _ = EH.MSEn(filtered[start:start+length], estimator, Scales=20)
            chunks.append(np.asarray(entropy))
        curves.append(np.mean(chunks, axis=0))
    curves = np.asarray(curves)
    Path(output).write_text(json.dumps(dict(MSE=float(curves.mean()),
        roi_scale_curves=curves.tolist(), epochs_per_roi=len(chunks))))


def replay():
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(MODEL/"core"))
    import dmf_rww_ei_jax_gpu_fc_only as base
    from simulate_motor_neural import simulate_neural
    OUT.mkdir(exist_ok=True)
    data = np.load(os.environ["VNS_MODEL_INPUT_NPZ"], allow_pickle=True)
    left, right = base.default_motor_rois(data["roi_labels"])
    motor = jnp.asarray(left+right)
    sc = [jnp.asarray(base.prepare_sc(base.sanitize_matrix(
        base.finite_nanmean(data["SC"][:, i], axis=0))), jnp.float32) for i in range(2)]
    for path in sorted(SOURCE.glob("selected_???.json")):
        record = json.loads(path.read_text())
        cid = record["candidate_id"]
        total_steps = int(round((record["burn_s"] +
            record["fit_n_tr"] * record["tr_s"]) / record["neural_dt"]))
        burn_steps = int(round(record["burn_s"] / record["neural_dt"]))
        for i, session in enumerate(("baseline", "followup6w")):
            destination = OUT/f"candidate_{cid:04d}_{session}.json"
            if destination.exists():
                continue
            values = simulate_neural(sc[i], jnp.asarray(record["results"][session]["params"],
                jnp.float32), motor, jax.random.PRNGKey(record["seed"]),
                total_steps, burn_steps, record["neural_dt"], .001, .382)
            signal = np.asarray(jax.device_get(values[0]), np.float32)
            del values
            with tempfile.TemporaryDirectory(prefix="mse25_") as temp:
                archive = Path(temp)/"signal.npz"
                result = Path(temp)/"result.json"
                np.savez(archive, S_E=signal)
                subprocess.run([sys.executable,
                    __file__, "worker", str(archive), str(result)], check=True)
                row = json.loads(result.read_text())
            row.update(candidate_id=cid, seed=record["seed"], session=session)
            destination.write_text(json.dumps(row, indent=2))
            print(f"candidate={cid} {session} MSE={row['MSE']:.6f}", flush=True)
    subprocess.run([sys.executable, __file__, "summary"], check=True)


def summary():
    import pandas as pd
    from scipy.stats import wilcoxon
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("candidate_*.json"))]
    df = pd.DataFrame([{k: r[k] for k in ("candidate_id", "seed", "session", "MSE")}
                       for r in rows])
    pair = df.pivot(index="candidate_id", columns="session", values="MSE")
    expected_count = len(list(SOURCE.glob("selected_???.json")))
    assert len(pair) == expected_count and pair.notna().all().all()
    pair["delta"] = pair.followup6w-pair.baseline
    result = dict(n=len(pair), baseline_mean=pair.baseline.mean(),
        followup_mean=pair.followup6w.mean(), mean_delta=pair.delta.mean(),
        median_delta=pair.delta.median(), sd_delta=pair.delta.std(),
        increased=int((pair.delta>0).sum()), decreased=int((pair.delta<0).sum()),
        p_increase=float(wilcoxon(pair.delta, alternative="greater").pvalue),
        p_two_sided=float(wilcoxon(pair.delta).pvalue),
        method="S_E, native 1000 Hz, 0.5-80 Hz Butterworth order4 zero-phase; "
        "EntropyHub DispEn defaults, scales1-20; two 85001-sample epochs; "
        "average scales, epochs, bilateral motor ROIs; original fit seeds; burn30s")
    df.to_csv(OUT/"mse_sessions.csv", index=False)
    pair.to_csv(OUT/"mse_paired.csv")
    (OUT/"summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv)>1 and sys.argv[1] == "summary":
        summary()
    else:
        replay()
