from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp

from . import matfree_adjoints


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperatorPlan:
    kind: str
    payload: object
    orientation: str
    algebra: str

    def tree_flatten(self):
        return (self.payload,), {
            "kind": self.kind,
            "orientation": self.orientation,
            "algebra": self.algebra,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload,) = children
        return cls(
            kind=aux_data["kind"],
            payload=payload,
            orientation=aux_data["orientation"],
            algebra=aux_data["algebra"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ScaledOperator:
    operator: object
    scale: object

    def tree_flatten(self):
        return (self.operator, self.scale), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        operator, scale = children
        return cls(operator=operator, scale=scale)


def _oriented_dense_matrix(mid: jax.Array, *, orientation: str):
    if orientation == "forward":
        return mid
    if orientation == "transpose":
        return jnp.swapaxes(mid, -1, -2)
    if orientation == "adjoint":
        return jnp.swapaxes(jnp.conjugate(mid), -1, -2)
    raise ValueError(f"unsupported operator orientation: {orientation}")


def dense_operator_plan(mid: jax.Array, *, orientation: str, algebra: str) -> OperatorPlan:
    return OperatorPlan(kind="dense", payload=_oriented_dense_matrix(mid, orientation=orientation), orientation=orientation, algebra=algebra)


def sparse_bcoo_operator_plan(a, *, as_sparse_bcoo, sparse_bcoo_cls, orientation: str, algebra: str, conjugate_transpose: bool = False) -> OperatorPlan:
    a = as_sparse_bcoo(a, algebra=algebra, label=f"matrix_free_core.{algebra}.operator_plan")
    if orientation == "forward":
        payload = a
    elif orientation == "transpose":
        payload = sparse_bcoo_cls(
            data=a.data,
            indices=a.indices[:, ::-1],
            rows=a.cols,
            cols=a.rows,
            algebra=algebra,
        )
    elif orientation == "adjoint":
        payload = sparse_bcoo_cls(
            data=jnp.conjugate(a.data) if conjugate_transpose else a.data,
            indices=a.indices[:, ::-1],
            rows=a.cols,
            cols=a.rows,
            algebra=algebra,
        )
    else:
        raise ValueError(f"unsupported operator orientation: {orientation}")
    return OperatorPlan(kind="sparse_bcoo", payload=payload, orientation=orientation, algebra=algebra)


def operator_plan_apply(plan: OperatorPlan, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    vv = midpoint_vector(v)
    if plan.kind == "dense":
        return jnp.asarray(jnp.einsum("...ij,...j->...i", plan.payload, vv), dtype=dtype)
    if plan.kind == "sparse_bcoo":
        return jnp.asarray(
            sparse_bcoo_matvec(plan.payload, vv, algebra=plan.algebra, label=f"matrix_free_core.{plan.algebra}.operator_plan_apply"),
            dtype=dtype,
        )
    raise ValueError(f"unsupported operator plan kind: {plan.kind}")


def operator_apply_midpoint(operator, v: jax.Array, *, midpoint_vector, sparse_bcoo_matvec, dtype):
    if isinstance(operator, ScaledOperator):
        applied = operator_apply_midpoint(
            operator.operator,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
        return jnp.asarray(operator.scale, dtype=dtype) * applied
    if isinstance(operator, OperatorPlan):
        return operator_plan_apply(
            operator,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=sparse_bcoo_matvec,
            dtype=dtype,
        )
    return jnp.asarray(operator(v), dtype=dtype)


def scaled_operator(operator, scale) -> ScaledOperator:
    return ScaledOperator(operator=operator, scale=scale)


def canonicalize_sparse_bcoo(x, *, algebra: str, sparse_common, label: str):
    if isinstance(x, sparse_common.SparseCOO):
        return sparse_common.coo_to_bcoo(x)
    if isinstance(x, sparse_common.SparseCSR):
        dense = sparse_common.sparse_to_dense(x, algebra=algebra, label=label)
        return sparse_common.dense_to_sparse_bcoo(dense, algebra=algebra)
    return sparse_common.as_sparse_bcoo(x, algebra=algebra, label=label)


def det_from_logdet(logdet):
    return jnp.exp(logdet)


def krylov_diagnostics(
    diagnostics_type,
    *,
    algorithm_code: int,
    steps: int | jax.Array,
    basis_dim: int | jax.Array,
    beta0,
    tail_norm,
    breakdown,
    used_adjoint: bool | jax.Array = False,
    gradient_supported: bool | jax.Array = True,
    probe_count: int | jax.Array = 1,
    restart_count: int | jax.Array = 0,
):
    return diagnostics_type(
        algorithm_code=jnp.asarray(algorithm_code, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(basis_dim, dtype=jnp.int32),
        restart_count=jnp.asarray(restart_count, dtype=jnp.int32),
        beta0=jnp.asarray(beta0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(breakdown),
        used_adjoint=jnp.asarray(used_adjoint),
        gradient_supported=jnp.asarray(gradient_supported),
        probe_count=jnp.asarray(probe_count, dtype=jnp.int32),
    )


def dense_funm_hermitian_eigh(scalar_fun, *, dtype, conjugate_right: bool):
    def apply(matrix: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(jnp.asarray(matrix, dtype=dtype))
        right = jnp.conjugate(evecs).T if conjugate_right else evecs.T
        return evecs @ jnp.diag(scalar_fun(evals)) @ right

    return apply


def dense_funm_general_eig(scalar_fun, *, dtype):
    def apply(matrix: jax.Array) -> jax.Array:
        vals, vecs = jnp.linalg.eig(jnp.asarray(matrix, dtype=dtype))
        inv = jnp.linalg.inv(vecs)
        return vecs @ jnp.diag(scalar_fun(vals)) @ inv

    return apply


def projected_krylov_action_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    *,
    krylov_decomp,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    coeff_dtype,
):
    basis, projected, beta0 = krylov_decomp(matvec, x, steps)
    e1 = jnp.zeros((steps,), dtype=coeff_dtype).at[0].set(jnp.asarray(1, dtype=coeff_dtype))
    y = beta0 * (basis.T @ (dense_funm(projected) @ e1))
    out = point_from_midpoint(y)
    finite = finite_mask_fn(y)
    return jnp.where(finite[..., None], out, full_like(out))


def projected_krylov_integrand_point(
    matvec,
    x: jax.Array,
    dense_funm,
    steps: int,
    *,
    krylov_decomp,
    coeff_dtype,
    scalar_dtype,
    scalar_postprocess,
):
    _, projected, beta0 = krylov_decomp(matvec, x, steps)
    e1 = jnp.zeros((steps,), dtype=coeff_dtype).at[0].set(jnp.asarray(1, dtype=coeff_dtype))
    value = (beta0**2) * jnp.vdot(e1, dense_funm(projected) @ e1)
    return jnp.asarray(scalar_postprocess(value), dtype=scalar_dtype)


def dense_operator(mid: jax.Array, *, midpoint_vector, orientation: str = "forward"):
    plan = dense_operator_plan(mid, orientation=orientation, algebra="matrix_free")

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(
            plan,
            v,
            midpoint_vector=midpoint_vector,
            sparse_bcoo_matvec=lambda *_args, **_kwargs: None,
            dtype=plan.payload.dtype,
        )

    return matvec


def dense_operator_adjoint(mid: jax.Array, *, midpoint_vector, conjugate: bool):
    return dense_operator(mid, midpoint_vector=midpoint_vector, orientation="adjoint" if conjugate else "transpose")


def dense_operator_rmatvec(mid: jax.Array, *, midpoint_vector):
    return dense_operator(mid, midpoint_vector=midpoint_vector, orientation="transpose")


def sparse_bcoo_operator(a, *, as_sparse_bcoo, sparse_bcoo_cls, sparse_bcoo_matvec, midpoint_vector, dtype, algebra: str, label: str):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="forward",
        algebra=algebra,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def sparse_bcoo_operator_adjoint(
    a,
    *,
    as_sparse_bcoo,
    sparse_bcoo_cls,
    sparse_bcoo_matvec,
    midpoint_vector,
    dtype,
    algebra: str,
    label: str,
    conjugate: bool,
):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="adjoint" if conjugate else "transpose",
        algebra=algebra,
        conjugate_transpose=conjugate,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def sparse_bcoo_operator_rmatvec(
    a,
    *,
    as_sparse_bcoo,
    sparse_bcoo_cls,
    sparse_bcoo_matvec,
    midpoint_vector,
    dtype,
    algebra: str,
    label: str,
):
    plan = sparse_bcoo_operator_plan(
        a,
        as_sparse_bcoo=as_sparse_bcoo,
        sparse_bcoo_cls=sparse_bcoo_cls,
        orientation="transpose",
        algebra=algebra,
    )

    def matvec(v: jax.Array) -> jax.Array:
        return operator_plan_apply(plan, v, midpoint_vector=midpoint_vector, sparse_bcoo_matvec=sparse_bcoo_matvec, dtype=dtype)

    return matvec


def operator_apply_point(
    operator,
    x: jax.Array,
    *,
    midpoint_apply,
    coerce_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    dtype,
) -> jax.Array:
    x = coerce_vector(x)
    y = midpoint_apply(operator, x)
    out = point_from_midpoint(y)
    finite = finite_mask_fn(y)
    return jnp.where(finite[..., None, None], out, full_like(out))


def poly_action_point(
    operator,
    x: jax.Array,
    coefficients: jax.Array,
    *,
    midpoint_apply,
    coerce_vector,
    midpoint_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    coeff_dtype,
):
    x = coerce_vector(x)
    coeffs = jnp.asarray(coefficients, dtype=coeff_dtype)
    x_mid = midpoint_vector(x)

    def step(carry, coeff):
        term, acc = carry
        next_acc = acc + coeff * term
        next_term = midpoint_apply(operator, point_from_midpoint(term))
        return (next_term, next_acc), None

    if coeffs.ndim != 1:
        raise ValueError("coefficients must be rank-1")
    init = (x_mid, jnp.zeros_like(x_mid))
    (_, acc), _ = lax.scan(step, init, coeffs)
    out = point_from_midpoint(acc)
    finite = finite_mask_fn(acc)
    return jnp.where(finite[..., None, None], out, full_like(out))


def expm_action_point(
    operator,
    x: jax.Array,
    *,
    terms: int,
    midpoint_apply,
    coerce_vector,
    midpoint_vector,
    point_from_midpoint,
    full_like,
    finite_mask_fn,
    scalar_dtype,
):
    x = coerce_vector(x)
    if terms <= 0:
        raise ValueError("terms must be > 0")
    x_mid = midpoint_vector(x)

    def step(carry, k):
        term, acc = carry
        next_term = midpoint_apply(operator, point_from_midpoint(term)) / jnp.asarray(k, dtype=scalar_dtype)
        next_acc = acc + next_term
        return (next_term, next_acc), None

    init = (x_mid, x_mid)
    (_, acc), _ = lax.scan(step, init, jnp.arange(1, terms, dtype=jnp.int32))
    out = point_from_midpoint(acc)
    finite = finite_mask_fn(acc)
    return jnp.where(finite[..., None, None], out, full_like(out))


def restarted_action_point(apply_once, x: jax.Array, *, restarts: int):
    if restarts <= 0:
        raise ValueError("restarts must be > 0")

    def body(y, _):
        next_y = apply_once(y)
        return next_y, None

    y, _ = lax.scan(body, x, xs=None, length=restarts)
    return y


def block_action_point(apply_one, xs: jax.Array) -> jax.Array:
    return jax.vmap(apply_one)(xs)


def rademacher_probes_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    mids = jax.random.rademacher(key, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(mids)


def normal_probes_real(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    mids = jax.random.normal(key, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(mids)


def rademacher_probes_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    key_re, key_im = jax.random.split(key)
    re = jax.random.rademacher(key_re, shape=(num, length), dtype=jnp.float64)
    im = jax.random.rademacher(key_im, shape=(num, length), dtype=jnp.float64)
    return jax.vmap(point_from_midpoint)(re + 1j * im)


def normal_probes_complex(point_from_midpoint, length: int, *, key: jax.Array, num: int) -> jax.Array:
    key_re, key_im = jax.random.split(key)
    re = jax.random.normal(key_re, shape=(num, length), dtype=jnp.float64)
    im = jax.random.normal(key_im, shape=(num, length), dtype=jnp.float64)
    mids = (re + 1j * im) / jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return jax.vmap(point_from_midpoint)(mids)


__all__ = [
    "OperatorPlan",
    "ScaledOperator",
    "matfree_adjoints",
    "dense_operator",
    "dense_operator_adjoint",
    "dense_operator_rmatvec",
    "dense_operator_plan",
    "sparse_bcoo_operator",
    "sparse_bcoo_operator_adjoint",
    "sparse_bcoo_operator_rmatvec",
    "sparse_bcoo_operator_plan",
    "operator_plan_apply",
    "operator_apply_midpoint",
    "scaled_operator",
    "canonicalize_sparse_bcoo",
    "det_from_logdet",
    "krylov_diagnostics",
    "dense_funm_hermitian_eigh",
    "dense_funm_general_eig",
    "projected_krylov_action_point",
    "projected_krylov_integrand_point",
    "operator_apply_point",
    "poly_action_point",
    "expm_action_point",
    "restarted_action_point",
    "block_action_point",
    "rademacher_probes_real",
    "normal_probes_real",
    "rademacher_probes_complex",
    "normal_probes_complex",
]
