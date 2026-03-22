from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


_METHOD_CODE = {
    "quadrature": 0,
    "aitken": 1,
    "wynn": 2,
    "high_precision_refine": 3,
    "mpfallback": 4,
}


def method_code(name: str) -> jax.Array:
    return jnp.asarray(_METHOD_CODE.get(name, -1), dtype=jnp.int32)


def regime_code(name: str, *, table: dict[str, int] | None = None) -> jax.Array:
    if table is None:
        table = {
            "default": 0,
            "structured": 1,
            "iterative": 2,
            "shifted": 3,
            "recycled": 4,
        }
    return jnp.asarray(table.get(name, -1), dtype=jnp.int32)


def incomplete_gamma_regime_code(*, near_singularity, cancellation_risk) -> jax.Array:
    near = jnp.asarray(near_singularity)
    cancel = jnp.asarray(cancellation_risk)
    return jnp.asarray(jnp.where(near, 1, jnp.where(cancel, 2, 0)), dtype=jnp.int32)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EvaluationFingerprint:
    regime_code: jax.Array
    method_code: jax.Array
    work_units: jax.Array
    scale: jax.Array
    compensated_sum: jax.Array
    adjoint_residual: jax.Array
    note: str = ""

    def tree_flatten(self):
        children = (
            self.regime_code,
            self.method_code,
            self.work_units,
            self.scale,
            self.compensated_sum,
            self.adjoint_residual,
        )
        return children, {"note": self.note}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        regime_code, method_code_arr, work_units, scale, compensated_sum, adjoint_residual = children
        return cls(
            regime_code=regime_code,
            method_code=method_code_arr,
            work_units=work_units,
            scale=scale,
            compensated_sum=compensated_sum,
            adjoint_residual=adjoint_residual,
            note=aux_data["note"],
        )


def make_fingerprint(
    *,
    regime_code_value,
    method_code_value,
    work_units=0,
    scale=1.0,
    compensated_sum=False,
    adjoint_residual=0.0,
    note: str = "",
) -> EvaluationFingerprint:
    return EvaluationFingerprint(
        regime_code=jnp.asarray(regime_code_value, dtype=jnp.int32),
        method_code=jnp.asarray(method_code_value, dtype=jnp.int32),
        work_units=jnp.asarray(work_units, dtype=jnp.int32),
        scale=jnp.asarray(scale, dtype=jnp.float64),
        compensated_sum=jnp.asarray(compensated_sum),
        adjoint_residual=jnp.asarray(adjoint_residual, dtype=jnp.float64),
        note=note,
    )


def with_adjoint_residual(fingerprint: EvaluationFingerprint, residual) -> EvaluationFingerprint:
    return EvaluationFingerprint(
        regime_code=fingerprint.regime_code,
        method_code=fingerprint.method_code,
        work_units=fingerprint.work_units,
        scale=fingerprint.scale,
        compensated_sum=fingerprint.compensated_sum,
        adjoint_residual=jnp.asarray(residual, dtype=jnp.float64),
        note=fingerprint.note,
    )
