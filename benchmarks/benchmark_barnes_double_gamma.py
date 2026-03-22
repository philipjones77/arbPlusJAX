from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import time

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import barnesg
from arbplusjax import double_gamma
from arbplusjax import double_interval as di


def _time_call(fn, *args, iters: int = 1) -> float:
    started = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - started) / float(iters)


def _legacy_vector(zs: jax.Array, tau: float) -> jax.Array:
    return jax.vmap(lambda zz: double_gamma.bdg_barnesdoublegamma(zz, tau, prec_bits=80))(zs)


def _ifj_vector(zs: jax.Array, tau: float) -> jax.Array:
    return double_gamma.ifj_barnesdoublegamma(zs, tau, dps=60)


def main() -> None:
    z_scalar = jnp.asarray(1.7 + 0.1j, dtype=jnp.complex128)
    tau = 0.5
    zs = jnp.asarray(
        [
            1.2 + 0.1j,
            1.5 + 0.15j,
            1.8 + 0.2j,
        ],
        dtype=jnp.complex128,
    )
    anchors = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    expected = jnp.asarray([1.0, 1.0, 1.0, 2.0], dtype=jnp.complex128)

    legacy_scalar_s = _time_call(lambda z: double_gamma.bdg_barnesdoublegamma(z, tau, prec_bits=80), z_scalar)
    ifj_scalar_s = _time_call(lambda z: double_gamma.ifj_barnesdoublegamma(z, tau, dps=60), z_scalar)
    legacy_vector_s = _time_call(_legacy_vector, zs, tau)
    ifj_vector_s = _time_call(_ifj_vector, zs, tau)

    legacy_shift = double_gamma.bdg_barnesdoublegamma(z_scalar + 1.0, tau, prec_bits=80) / double_gamma.bdg_barnesdoublegamma(z_scalar, tau, prec_bits=80)
    ifj_shift = double_gamma.ifj_barnesdoublegamma(z_scalar + 1.0, tau, dps=60) / double_gamma.ifj_barnesdoublegamma(z_scalar, tau, dps=60)
    shift_target = jnp.exp(barnesg._complex_loggamma(z_scalar / tau))
    legacy_shift_err = jnp.abs(legacy_shift - shift_target)
    ifj_shift_err = jnp.abs(ifj_shift - shift_target)

    legacy_anchor = jnp.asarray([double_gamma.bdg_barnesdoublegamma(x, 1.0, prec_bits=80) for x in anchors], dtype=jnp.complex128)
    ifj_anchor = jnp.asarray([double_gamma.ifj_barnesdoublegamma(x, 1.0, dps=60) for x in anchors], dtype=jnp.complex128)
    legacy_anchor_err = jnp.max(jnp.abs(legacy_anchor - expected))
    ifj_anchor_err = jnp.max(jnp.abs(ifj_anchor - expected))

    diagnostics = double_gamma.ifj_barnesdoublegamma_diagnostics(0.2 + 0.05j, 1.0, dps=60, max_m_cap=256)

    point = acb_core.acb_box(di.interval(1.3, 1.3), di.interval(0.2, 0.2))
    g_box = acb_core.acb_barnes_g(point)
    log_box = acb_core.acb_log_barnes_g(point)
    alias_consistency = jnp.abs(jnp.exp(acb_core.acb_midpoint(log_box)) - acb_core.acb_midpoint(g_box))

    print(f"legacy_scalar_s: {legacy_scalar_s:.6f}")
    print(f"ifj_scalar_s: {ifj_scalar_s:.6f}")
    print(f"legacy_vector_s: {legacy_vector_s:.6f}")
    print(f"ifj_vector_s: {ifj_vector_s:.6f}")
    print(f"legacy_shift_err_abs: {float(legacy_shift_err):.6e}")
    print(f"ifj_shift_err_abs: {float(ifj_shift_err):.6e}")
    print(f"legacy_tau1_anchor_max_abs_err: {float(legacy_anchor_err):.6e}")
    print(f"ifj_tau1_anchor_max_abs_err: {float(ifj_anchor_err):.6e}")
    print(f"ifj_diag_m_base: {diagnostics.m_base}")
    print(f"ifj_diag_m_used: {diagnostics.m_used}")
    print(f"ifj_diag_n_shift: {diagnostics.n_shift}")
    print(f"ifj_diag_m_capped: {int(diagnostics.m_capped)}")
    print(f"acb_barnes_g_alias_consistency_abs: {float(alias_consistency):.6e}")


if __name__ == "__main__":
    main()
