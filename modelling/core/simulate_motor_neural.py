"""Native 1-ms neural replay for fitted motor-region signals."""
import jax
import jax.numpy as jnp

@jax.jit(static_argnames=("total_steps", "burn_steps"))
def simulate_neural(sc, params, motor_indices, key, total_steps, burn_steps,
                    dt, fixed_sigma, fixed_i0):
    """Return post-burn motor S_E, r_E and I_E at the fitted 10-ms step."""
    G, w_EE, w_EI, w_IE = params
    n_roi = sc.shape[0]
    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    gamma_e, gamma_i = 0.641, 1.0
    tau_e, tau_i = 0.1, 0.01
    w_e, w_i, w_ii = 1.0, 0.7, 1.0

    def transfer(x, a, b, d):
        y = a * x - b
        den = 1.0 - jnp.exp(-d * y)
        out = jnp.where(jnp.abs(den) < 1e-8, 1.0 / d, y / den)
        return jnp.clip(out, 0.0, 500.0)

    key_e, key_i = jax.random.split(key)
    noise_e = jax.random.normal(key_e, (total_steps, n_roi), dtype=jnp.float32)
    noise_i = jax.random.normal(key_i, (total_steps, n_roi), dtype=jnp.float32)

    def step(carry, noise):
        s_e, s_i = carry
        ne, ni = noise
        coupling = sc @ s_e
        i_e = w_e * fixed_i0 + w_EE * j_nmda * s_e + G * j_nmda * coupling - w_IE * s_i
        i_i = w_i * fixed_i0 + w_EI * j_nmda * s_e - w_ii * s_i
        r_e = transfer(i_e, a_e, b_e, d_e)
        r_i = transfer(i_i, a_i, b_i, d_i)
        next_e = s_e + (-s_e / tau_e + (1.0 - s_e) * gamma_e * r_e) * dt + fixed_sigma * jnp.sqrt(dt) * ne
        next_i = s_i + (-s_i / tau_i + gamma_i * r_i) * dt + fixed_sigma * jnp.sqrt(dt) * ni
        next_e = jnp.clip(next_e, 0.0, 1.0)
        next_i = jnp.clip(next_i, 0.0, 1.0)
        return (next_e, next_i), (next_e[motor_indices], r_e[motor_indices], i_e[motor_indices])

    init = (jnp.full((n_roi,), 0.001, dtype=jnp.float32),
            jnp.full((n_roi,), 0.001, dtype=jnp.float32))
    _, hist = jax.lax.scan(step, init, (noise_e, noise_i), length=total_steps)
    return tuple(jnp.swapaxes(item[burn_steps:], 0, 1) for item in hist)
