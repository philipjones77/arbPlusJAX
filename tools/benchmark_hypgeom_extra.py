from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from arbplusjax import hypgeom


def _timer(fn, *args, iters: int = 50) -> float:
    fn(*args)
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn(*args))
    end = time.perf_counter()
    return (end - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    x = jnp.array([0.2, 0.3], dtype=jnp.float64)
    s = jnp.array([1.0, 1.0], dtype=jnp.float64)
    a = jnp.array([0.5, 0.5], dtype=jnp.float64)
    b = jnp.array([1.5, 1.5], dtype=jnp.float64)

    bench = [
        ("fresnel", lambda: hypgeom.arb_hypgeom_fresnel_batch_jit(jnp.stack([x, x]))),
        ("ei", lambda: hypgeom.arb_hypgeom_ei_batch_jit(jnp.stack([x, x]))),
        ("si", lambda: hypgeom.arb_hypgeom_si_batch_jit(jnp.stack([x, x]))),
        ("ci", lambda: hypgeom.arb_hypgeom_ci_batch_jit(jnp.stack([x, x]))),
        ("shi", lambda: hypgeom.arb_hypgeom_shi_batch_jit(jnp.stack([x, x]))),
        ("chi", lambda: hypgeom.arb_hypgeom_chi_batch_jit(jnp.stack([x, x]))),
        ("li", lambda: hypgeom.arb_hypgeom_li_batch_jit(jnp.stack([x, x]), offset=1)),
        ("dilog", lambda: hypgeom.arb_hypgeom_dilog_batch_jit(jnp.stack([x, x]))),
        ("airy", lambda: hypgeom.arb_hypgeom_airy_batch_jit(jnp.stack([x, x]))),
        ("expint", lambda: hypgeom.arb_hypgeom_expint_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("gamma_lower", lambda: hypgeom.arb_hypgeom_gamma_lower_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("gamma_upper", lambda: hypgeom.arb_hypgeom_gamma_upper_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("beta_lower", lambda: hypgeom.arb_hypgeom_beta_lower_batch_jit(jnp.stack([a, a]), jnp.stack([b, b]), jnp.stack([x, x]))),
        ("cheb_t", lambda: hypgeom.arb_hypgeom_chebyshev_t_batch_jit(args.n, jnp.stack([x, x]))),
        ("cheb_u", lambda: hypgeom.arb_hypgeom_chebyshev_u_batch_jit(args.n, jnp.stack([x, x]))),
        ("laguerre", lambda: hypgeom.arb_hypgeom_laguerre_l_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("hermite", lambda: hypgeom.arb_hypgeom_hermite_h_batch_jit(args.n, jnp.stack([x, x]))),
        ("legendre_p", lambda: hypgeom.arb_hypgeom_legendre_p_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("legendre_q", lambda: hypgeom.arb_hypgeom_legendre_q_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("jacobi", lambda: hypgeom.arb_hypgeom_jacobi_p_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([b, b]), jnp.stack([x, x]))),
        ("gegenbauer", lambda: hypgeom.arb_hypgeom_gegenbauer_c_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("central_bin", lambda: hypgeom.arb_hypgeom_central_bin_ui_batch_jit(jnp.array([args.n, args.n + 1]))),
    ]

    for name, fn in bench:
        dt = _timer(fn, iters=args.iters)
        print(f"{name:14s} {dt*1e6:8.2f} us")


if __name__ == "__main__":
    main()
