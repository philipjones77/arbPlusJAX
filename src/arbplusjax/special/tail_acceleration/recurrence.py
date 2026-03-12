from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class TailRatioRecurrence:
    a0: float
    a1: float
    b0: float
    b1: float
    alpha: Callable[[int], float]
    beta: Callable[[int], float]
    gamma: Callable[[int], float]
    delta: Callable[[int], float]
    order: int = 2
    note: str = ""


def ratio_recurrence_terms(spec: TailRatioRecurrence, n_terms: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if n_terms < 2:
        raise ValueError("n_terms must be >= 2")
    a_terms = [jnp.asarray(spec.a0, dtype=jnp.float64), jnp.asarray(spec.a1, dtype=jnp.float64)]
    b_terms = [jnp.asarray(spec.b0, dtype=jnp.float64), jnp.asarray(spec.b1, dtype=jnp.float64)]
    for n in range(1, n_terms - 1):
        a_next = spec.alpha(n) * a_terms[-1] + spec.beta(n) * a_terms[-2]
        b_next = spec.gamma(n) * b_terms[-1] + spec.delta(n) * b_terms[-2]
        a_terms.append(jnp.asarray(a_next, dtype=jnp.float64))
        b_terms.append(jnp.asarray(b_next, dtype=jnp.float64))
    return jnp.stack(a_terms), jnp.stack(b_terms)


def ratio_recurrence_estimate(spec: TailRatioRecurrence, n_terms: int) -> jnp.ndarray:
    a_terms, b_terms = ratio_recurrence_terms(spec, n_terms=n_terms)
    return a_terms[-1] / b_terms[-1]
