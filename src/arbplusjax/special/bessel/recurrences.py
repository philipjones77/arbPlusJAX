from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ..tail_acceleration import TailEvaluationDiagnostics


def incomplete_bessel_k_recurrence(
    nu,
    z,
    lower_limit,
    *,
    n_terms: int = 4,
) -> tuple[jnp.ndarray, TailEvaluationDiagnostics]:
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    lower_v = jnp.asarray(lower_limit, dtype=jnp.float64)
    if n_terms <= 0:
        raise ValueError("n_terms must be > 0")

    phi_prime = lambda t: jnp.sinh(t)
    psi = lambda t: jnp.cosh(nu_v * t)
    safe_floor = jnp.asarray(1e-12, dtype=jnp.float64)

    def divide_by_phi_prime(fn: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
        def wrapped(t):
            denom = phi_prime(t)
            safe = jnp.where(jnp.abs(denom) > safe_floor, denom, safe_floor)
            return fn(t) / safe

        return wrapped

    eta = divide_by_phi_prime(psi)
    coeffs = [eta(lower_v)]
    for _ in range(n_terms - 1):
        prev = eta
        eta = lambda t, prev=prev: -divide_by_phi_prime(jax.grad(prev))(t)
        coeffs.append(eta(lower_v))

    coeff_arr = jnp.stack(coeffs)
    powers = z_v ** (jnp.arange(n_terms, dtype=jnp.float64) + 1.0)
    value = jnp.exp(-z_v * jnp.cosh(lower_v)) * jnp.sum(coeff_arr / powers)
    remainder = jnp.abs(jnp.exp(-z_v * jnp.cosh(lower_v)) * coeff_arr[-1] / (z_v ** (n_terms + 1.0)))
    instability_flags: list[str] = []
    if jnp.abs(phi_prime(lower_v)) < 0.25:
        instability_flags.append("small_phi_prime")
    if jnp.abs(z_v) < 1.0:
        instability_flags.append("slow_decay")
    if jnp.abs(nu_v) > 8.0 and jnp.abs(z_v) < 1.0:
        instability_flags.append("large_order_small_argument")
    if remainder > 0.25 * jnp.maximum(jnp.abs(value), jnp.asarray(1e-12, dtype=jnp.float64)):
        instability_flags.append("large_estimated_remainder")
    diagnostics = TailEvaluationDiagnostics(
        method="recurrence",
        chunk_count=0,
        panel_count=0,
        recurrence_steps=n_terms,
        estimated_tail_remainder=remainder,
        instability_flags=tuple(instability_flags),
        fallback_used=False,
        precision_warning=bool(instability_flags),
        note="Endpoint coefficient recurrence for the incomplete-K tail asymptotic expansion.",
    )
    return value, diagnostics
