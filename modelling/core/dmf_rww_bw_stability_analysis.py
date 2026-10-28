"""Deterministic fixed point, Jacobian, and stability metrics for the rWW model."""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

EPS = 1e-12

@jax.jit
def h_transfer(x, a, b, d):
    y = a * x - b
    den = 1.0 - jnp.exp(-d * y)
    out = jnp.where(jnp.abs(den) < 1e-8, 1.0 / d, y / den)
    return jnp.clip(out, 0.0, 500.0)


def no_overwrite_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return path


def neural_rhs_jax(x, sc, params, fixed_i0):
    n_roi = sc.shape[0]
    s_e = x[:n_roi]
    s_i = x[n_roi:]
    g, w_ee, w_ei, w_ie = params

    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    gamma_e, gamma_i = 0.641, 1.0
    tau_e, tau_i = 0.1, 0.01
    w_e, w_i, w_ii = 1.0, 0.7, 1.0

    coupling = sc @ s_e
    i_e = w_e * fixed_i0 + w_ee * j_nmda * s_e + g * j_nmda * coupling - w_ie * s_i
    i_i = w_i * fixed_i0 + w_ei * j_nmda * s_e - w_ii * s_i
    r_e = h_transfer(i_e, a_e, b_e, d_e)
    r_i = h_transfer(i_i, a_i, b_i, d_i)
    ds_e = -s_e / tau_e + (1.0 - s_e) * gamma_e * r_e
    ds_i = -s_i / tau_i + gamma_i * r_i
    return jnp.concatenate([ds_e, ds_i])


def neural_summary(x, sc, params, fixed_i0):
    n_roi = sc.shape[0]
    s_e = x[:n_roi]
    s_i = x[n_roi:]
    g, w_ee, w_ei, w_ie = params
    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    i_e = fixed_i0 + w_ee * j_nmda * s_e + g * j_nmda * (sc @ s_e) - w_ie * s_i
    i_i = 0.7 * fixed_i0 + w_ei * j_nmda * s_e - s_i
    r_e = np.asarray(h_transfer(jnp.asarray(i_e), a_e, b_e, d_e))
    r_i = np.asarray(h_transfer(jnp.asarray(i_i), a_i, b_i, d_i))
    return {
        "S_E": np.asarray(s_e),
        "S_I": np.asarray(s_i),
        "I_E": np.asarray(i_e),
        "I_I": np.asarray(i_i),
        "r_E": r_e,
        "r_I": r_i,
        "S_E_over_S_I": np.asarray(s_e) / (np.asarray(s_i) + EPS),
    }


def rhs_np(x, sc, params, fixed_i0):
    out = neural_rhs_jax(jnp.asarray(x), jnp.asarray(sc), jnp.asarray(params), fixed_i0)
    return np.asarray(jax.device_get(out), dtype=float)


def solve_fixed_point(sc, params, fixed_i0, x0=None, tol=1e-10, max_nfev=400):
    n_roi = sc.shape[0]
    if x0 is None:
        x0 = np.concatenate([np.full(n_roi, 0.05), np.full(n_roi, 0.05)])
    lower = np.zeros(2 * n_roi)
    upper = np.concatenate([np.ones(n_roi), np.full(n_roi, 5.0)])
    res = least_squares(
        lambda z: rhs_np(z, sc, params, fixed_i0),
        np.clip(x0, lower + 1e-9, upper - 1e-9),
        jac=lambda z: jacobian_np(z, sc, params, fixed_i0),
        bounds=(lower, upper),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=max_nfev,
    )
    residual = rhs_np(res.x, sc, params, fixed_i0)
    return res.x, float(np.linalg.norm(residual) / np.sqrt(residual.size)), bool(res.success), res.message


def jacobian_np(x, sc, params, fixed_i0):
    jac_fn = jax.jacfwd(lambda z: neural_rhs_jax(z, jnp.asarray(sc), jnp.asarray(params), fixed_i0))
    return np.asarray(jax.device_get(jac_fn(jnp.asarray(x))), dtype=float)


def side_weighted_roi_energy(roi_energy, left_motor, right_motor, subject_sides):
    left = float(np.nanmean(roi_energy[left_motor]))
    right = float(np.nanmean(roi_energy[right_motor]))
    n = len(subject_sides)
    n_right = sum(s == "R" for s in subject_sides)
    n_left = sum(s == "L" for s in subject_sides)
    contra = (n_right * left + n_left * right) / n
    ipsi = (n_left * left + n_right * right) / n
    bilateral = 0.5 * (left + right)
    return left, right, contra, ipsi, bilateral


def stability_metrics(jac, left_motor, right_motor, subject_sides):
    n_roi = jac.shape[0] // 2
    vals, vecs = np.linalg.eig(jac)
    lead_i = int(np.argmax(vals.real))
    lead_val = vals[lead_i]
    vec = vecs[:, lead_i]
    energy = np.abs(vec) ** 2
    total_energy = float(np.sum(energy))
    motor = sorted(set(left_motor + right_motor))
    motor_idx = np.asarray(motor, dtype=int)
    motor_state_idx = np.concatenate([motor_idx, n_roi + motor_idx])
    motor_loading = float(np.sum(energy[motor_state_idx]) / total_energy)
    motor_size_fraction = len(motor) / n_roi
    roi_energy = energy[:n_roi] + energy[n_roi:]
    roi_energy = roi_energy / (np.sum(roi_energy) + EPS)
    left, right, contra, ipsi, bilateral = side_weighted_roi_energy(roi_energy, left_motor, right_motor, subject_sides)

    e_total = float(np.sum(energy[:n_roi]))
    i_total = float(np.sum(energy[n_roi:]))
    motor_e_loading = float(np.sum(energy[motor_idx]) / (e_total + EPS))
    motor_i_loading = float(np.sum(energy[n_roi + motor_idx]) / (i_total + EPS))
    local_idx = motor_state_idx
    local_jac = jac[np.ix_(local_idx, local_idx)]
    local_vals = np.linalg.eigvals(local_jac)
    return {
        "max_real_lambda": float(lead_val.real),
        "leading_lambda_imag": float(lead_val.imag),
        "stability_margin": float(-lead_val.real),
        "motor_loading_fraction": motor_loading,
        "motor_loading_enrichment": float(motor_loading / motor_size_fraction),
        "motor_E_loading_fraction": motor_e_loading,
        "motor_I_loading_fraction": motor_i_loading,
        "contra_motor_loading_density": contra,
        "ipsi_motor_loading_density": ipsi,
        "bilateral_motor_loading_density": bilateral,
        "local_motor_block_max_real_lambda": float(np.max(local_vals.real)),
    }


