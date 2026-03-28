from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from . import matrix_free_core as _core


def contour_quadrature_nodes(center, radius, *, quadrature_order: int):
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be > 0")
    theta = (2.0 * jnp.pi / quadrature_order) * (jnp.arange(quadrature_order, dtype=jnp.float64) + 0.5)
    unit = jnp.exp(1j * theta)
    center_arr = jnp.asarray(center, dtype=jnp.complex128)
    radius_arr = jnp.asarray(radius, dtype=jnp.complex128)
    nodes = center_arr + radius_arr * unit
    weights = ((2.0j * jnp.pi) * radius_arr * unit) / jnp.asarray(quadrature_order, dtype=jnp.complex128)
    return nodes, weights


def contour_filter_subspace_point(
    solve_shifted_block,
    basis: jax.Array,
    *,
    center,
    radius,
    quadrature_order: int,
) -> jax.Array:
    nodes, weights = contour_quadrature_nodes(center, radius, quadrature_order=quadrature_order)
    init = jnp.zeros_like(jnp.asarray(basis), dtype=jnp.complex128)

    def body(acc, nw):
        node, weight = nw
        return acc + weight * solve_shifted_block(node, basis), None

    filtered, _ = lax.scan(body, init, (nodes, weights))
    return _core.orthonormalize_columns(filtered)


def contour_integral_action_point(
    solve_shifted,
    x: jax.Array,
    *,
    center,
    radius,
    quadrature_order: int,
    node_weight_fn=None,
) -> jax.Array:
    effective_order = max(int(quadrature_order), 64)
    nodes, weights = contour_quadrature_nodes(center, radius, quadrature_order=effective_order)
    vector = jnp.asarray(x)
    out_dtype = jnp.result_type(vector.dtype, jnp.complex128)
    init = jnp.zeros_like(vector, dtype=out_dtype)

    def body(acc, nw):
        node, weight = nw
        kernel = jnp.asarray(1.0 if node_weight_fn is None else node_weight_fn(node), dtype=out_dtype)
        value = jnp.asarray(solve_shifted(node, vector), dtype=out_dtype)
        return acc - jnp.asarray(weight, dtype=out_dtype) * kernel * value, None

    value, _ = lax.scan(body, init, (nodes, weights))
    return value
