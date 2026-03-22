from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import kernel_helpers as kh



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseMatvecPlan:
    matrix: jax.Array
    rows: int
    cols: int
    algebra: str

    def tree_flatten(self):
        return (self.matrix,), {"rows": self.rows, "cols": self.cols, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (matrix,) = children
        return cls(matrix=matrix, rows=aux_data["rows"], cols=aux_data["cols"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseLUSolvePlan:
    p: jax.Array
    l: jax.Array
    u: jax.Array
    rows: int
    algebra: str

    def tree_flatten(self):
        return (self.p, self.l, self.u), {"rows": self.rows, "algebra": self.algebra}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        p, l, u = children
        return cls(p=p, l=l, u=u, rows=aux_data["rows"], algebra=aux_data["algebra"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseCholeskySolvePlan:
    factor: jax.Array
    rows: int
    algebra: str
    structure: str

    def tree_flatten(self):
        return (self.factor,), {"rows": self.rows, "algebra": self.algebra, "structure": self.structure}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (factor,) = children
        return cls(
            factor=factor,
            rows=aux_data["rows"],
            algebra=aux_data["algebra"],
            structure=aux_data["structure"],
        )


def is_dense_plan_like(x) -> bool:
    if isinstance(x, (DenseMatvecPlan, DenseLUSolvePlan, DenseCholeskySolvePlan)):
        return True
    if isinstance(x, tuple) and len(x) == 3:
        return all(hasattr(part, "shape") for part in x)
    return False


def is_batch_pad_candidate(x) -> bool:
    shape = getattr(x, "shape", None)
    return shape is not None and len(shape) > 0


def as_interval_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check_tail_shape(arr, (2, 2, 2), label)
    return arr


def as_interval_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], f"{label}.square")
    return arr


def as_interval_rect_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    return arr


def as_interval_vector(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    return arr


def as_interval_rhs(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    return arr


def as_box_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check_tail_shape(arr, (2, 2, 4), label)
    return arr


def as_box_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], f"{label}.square")
    return arr


def as_box_rect_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    return arr


def as_box_vector(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    return arr


def as_box_rhs(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    return arr


def full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def interval_from_point(x: jax.Array) -> jax.Array:
    return di.interval(di._below(x), di._above(x))


def box_from_point(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def interval_is_finite(x: jax.Array) -> jax.Array:
    return jnp.isfinite(x[..., 0]) & jnp.isfinite(x[..., 1])


def box_is_finite(x: jax.Array) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    return interval_is_finite(re) & interval_is_finite(im)


def complex_is_finite(z: jax.Array) -> jax.Array:
    return jnp.isfinite(jnp.real(z)) & jnp.isfinite(jnp.imag(z))


def interval_overlaps(a: jax.Array, b: jax.Array) -> jax.Array:
    return (a[..., 0] <= b[..., 1]) & (b[..., 0] <= a[..., 1])


def box_overlaps(a: jax.Array, b: jax.Array) -> jax.Array:
    return interval_overlaps(acb_core.acb_real(a), acb_core.acb_real(b)) & interval_overlaps(acb_core.acb_imag(a), acb_core.acb_imag(b))


def interval_equal(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.all(a == b, axis=-1)


def box_equal(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.all(a == b, axis=-1)


def interval_is_exact(x: jax.Array) -> jax.Array:
    return x[..., 0] == x[..., 1]


def box_is_exact(x: jax.Array) -> jax.Array:
    return interval_is_exact(acb_core.acb_real(x)) & interval_is_exact(acb_core.acb_imag(x))


def interval_is_zero(x: jax.Array) -> jax.Array:
    return (x[..., 0] == 0) & (x[..., 1] == 0)


def interval_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    lo = jnp.sum(xs[..., 0], axis=axis)
    hi = jnp.sum(xs[..., 1], axis=axis)
    return di.interval(di._below(lo), di._above(hi))


def box_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    re = acb_core.acb_real(xs)
    im = acb_core.acb_imag(xs)
    re_out = di.interval(di._below(jnp.sum(re[..., 0], axis=axis)), di._above(jnp.sum(re[..., 1], axis=axis)))
    im_out = di.interval(di._below(jnp.sum(im[..., 0], axis=axis)), di._above(jnp.sum(im[..., 1], axis=axis)))
    return acb_core.acb_box(re_out, im_out)


def real_midpoint_symmetric_part(a: jax.Array) -> jax.Array:
    return 0.5 * (a + jnp.swapaxes(a, -2, -1))


def complex_midpoint_hermitian_part(a: jax.Array) -> jax.Array:
    return 0.5 * (a + jnp.conj(jnp.swapaxes(a, -2, -1)))


def real_midpoint_is_symmetric(a: jax.Array) -> jax.Array:
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(a), axis=(-2, -1)))
    tol = 32.0 * jnp.finfo(a.dtype).eps * scale
    err = jnp.max(jnp.abs(a - jnp.swapaxes(a, -2, -1)), axis=(-2, -1))
    return err <= tol


def complex_midpoint_is_hermitian(a: jax.Array) -> jax.Array:
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(a), axis=(-2, -1)))
    tol = 32.0 * jnp.finfo(jnp.real(a).dtype).eps * scale
    err = jnp.max(jnp.abs(a - jnp.conj(jnp.swapaxes(a, -2, -1))), axis=(-2, -1))
    return err <= tol


def midpoint_is_diagonal(a: jax.Array) -> jax.Array:
    diag = jnp.diagonal(a, axis1=-2, axis2=-1)
    rebuilt = jnp.eye(a.shape[-1], dtype=a.dtype) * diag[..., None, :]
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(a), axis=(-2, -1)))
    tol = 32.0 * jnp.finfo(jnp.real(a).dtype).eps * scale
    err = jnp.max(jnp.abs(a - rebuilt), axis=(-2, -1))
    return err <= tol


def midpoint_is_triangular(a: jax.Array, *, lower: bool) -> jax.Array:
    masked = jnp.tril(a) if lower else jnp.triu(a)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(a), axis=(-2, -1)))
    tol = 32.0 * jnp.finfo(jnp.real(a).dtype).eps * scale
    err = jnp.max(jnp.abs(a - masked), axis=(-2, -1))
    return err <= tol


def lower_cholesky_finite(factor: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(factor), axis=(-2, -1))


def lower_cholesky_solve(factor: jax.Array, rhs: jax.Array) -> jax.Array:
    vector_rhs = rhs.ndim == factor.ndim - 1
    rhs2 = rhs[..., None] if vector_rhs else rhs
    y = lax.linalg.triangular_solve(factor, rhs2, left_side=True, lower=True, transpose_a=False, conjugate_a=False)
    x = lax.linalg.triangular_solve(factor, y, left_side=True, lower=True, transpose_a=True, conjugate_a=True)
    return x[..., 0] if vector_rhs else x


def lower_cholesky_solve_transpose(factor: jax.Array, rhs: jax.Array) -> jax.Array:
    vector_rhs = rhs.ndim == factor.ndim - 1
    rhs2 = rhs[..., None] if vector_rhs else rhs
    y = lax.linalg.triangular_solve(factor, rhs2, left_side=True, lower=True, transpose_a=False, conjugate_a=True)
    x = lax.linalg.triangular_solve(factor, y, left_side=True, lower=True, transpose_a=True, conjugate_a=False)
    return x[..., 0] if vector_rhs else x


def real_symmetric_eigh(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.eigh(real_midpoint_symmetric_part(a))


def complex_hermitian_eigh(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.linalg.eigh(complex_midpoint_hermitian_part(a))


def matrix_power_ui(a: jax.Array, n: int) -> jax.Array:
    checks.check(n >= 0, "mat_common.matrix_power_ui.n_nonnegative")
    size = a.shape[-1]
    result = jnp.broadcast_to(jnp.eye(size, dtype=a.dtype), a.shape)
    base = a
    exp = int(n)
    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1
    return result


def _poly_from_roots(roots: jax.Array) -> jax.Array:
    roots = jnp.asarray(roots)
    n = int(roots.shape[-1])
    coeffs = jnp.zeros(roots.shape[:-1] + (n + 1,), dtype=roots.dtype)
    coeffs = coeffs.at[..., 0].set(jnp.asarray(1, dtype=roots.dtype))
    for i in range(n):
        r = roots[..., i]
        updates = coeffs[..., : i + 1]
        coeffs = coeffs.at[..., 1 : i + 2].add(-r[..., None] * updates)
    return coeffs


def characteristic_polynomial_from_matrix(a: jax.Array, *, hermitian: bool = False) -> jax.Array:
    roots = jnp.linalg.eigvalsh(a) if hermitian else jnp.linalg.eigvals(a)
    return _poly_from_roots(roots)


def matrix_exp(a: jax.Array, *, hermitian: bool = False) -> jax.Array:
    if hermitian:
        vals, vecs = jnp.linalg.eigh(a)
        return (vecs * jnp.exp(vals)[..., None, :]) @ jnp.conj(jnp.swapaxes(vecs, -2, -1))
    vals, vecs = jnp.linalg.eig(a)
    inv = jnp.linalg.inv(vecs)
    return (vecs * jnp.exp(vals)[..., None, :]) @ inv


def companion_matrix(coeffs: jax.Array) -> jax.Array:
    coeffs = jnp.asarray(coeffs)
    checks.check(coeffs.ndim == 1, "mat_common.companion.coeffs_rank")
    checks.check(coeffs.shape[0] >= 2, "mat_common.companion.coeffs_len")
    degree = int(coeffs.shape[0] - 1)
    lead = coeffs[0]
    last_col = -coeffs[:0:-1] / lead
    out = jnp.zeros((degree, degree), dtype=coeffs.dtype)
    if degree > 1:
        idx = jnp.arange(degree - 1)
        out = out.at[idx + 1, idx].set(jnp.asarray(1, dtype=coeffs.dtype))
    out = out.at[:, -1].set(last_col)
    return out


def hilbert_matrix(n: int, *, dtype) -> jax.Array:
    idx = jnp.arange(n, dtype=jnp.asarray(0, dtype=dtype).dtype)
    return jnp.asarray(1.0, dtype=dtype) / (idx[:, None] + idx[None, :] + jnp.asarray(1.0, dtype=dtype))


def pascal_matrix(n: int, *, dtype) -> jax.Array:
    out = jnp.zeros((n, n), dtype=dtype)
    one = jnp.asarray(1, dtype=dtype)
    for i in range(n):
        row = jnp.zeros((n,), dtype=dtype).at[0].set(one)
        for j in range(1, n):
            row = row.at[j].set(row[j - 1] * jnp.asarray(i + j, dtype=dtype) / jnp.asarray(j, dtype=dtype))
        out = out.at[i].set(row)
    return out


def stirling2_matrix(n: int, *, dtype) -> jax.Array:
    out = jnp.zeros((n, n), dtype=dtype)
    if n == 0:
        return out
    out = out.at[0, 0].set(jnp.asarray(1, dtype=dtype))
    for i in range(1, n):
        for j in range(1, i + 1):
            out = out.at[i, j].set(out[i - 1, j - 1] + jnp.asarray(j, dtype=dtype) * out[i - 1, j])
    return out


def interval_trace(a: jax.Array) -> jax.Array:
    n = a.shape[-2]
    idx = jnp.arange(n)
    return interval_sum(a[..., idx, idx, :], axis=-1)


def box_trace(a: jax.Array) -> jax.Array:
    n = a.shape[-2]
    idx = jnp.arange(n)
    return box_sum(a[..., idx, idx, :], axis=-1)


def interval_det_2x2(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    return di.fast_sub(di.fast_mul(a00, a11), di.fast_mul(a01, a10))


def box_det_2x2(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    return acb_core.acb_sub(acb_core.acb_mul(a00, a11), acb_core.acb_mul(a01, a10))


def interval_det_3x3(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a02 = a[..., 0, 2, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    a12 = a[..., 1, 2, :]
    a20 = a[..., 2, 0, :]
    a21 = a[..., 2, 1, :]
    a22 = a[..., 2, 2, :]
    pos = interval_sum(
        jnp.stack(
            [
                di.fast_mul(di.fast_mul(a00, a11), a22),
                di.fast_mul(di.fast_mul(a01, a12), a20),
                di.fast_mul(di.fast_mul(a02, a10), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    neg = interval_sum(
        jnp.stack(
            [
                di.fast_mul(di.fast_mul(a02, a11), a20),
                di.fast_mul(di.fast_mul(a01, a10), a22),
                di.fast_mul(di.fast_mul(a00, a12), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    return di.fast_sub(pos, neg)


def box_det_3x3(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a02 = a[..., 0, 2, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    a12 = a[..., 1, 2, :]
    a20 = a[..., 2, 0, :]
    a21 = a[..., 2, 1, :]
    a22 = a[..., 2, 2, :]
    pos = box_sum(
        jnp.stack(
            [
                acb_core.acb_mul(acb_core.acb_mul(a00, a11), a22),
                acb_core.acb_mul(acb_core.acb_mul(a01, a12), a20),
                acb_core.acb_mul(acb_core.acb_mul(a02, a10), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    neg = box_sum(
        jnp.stack(
            [
                acb_core.acb_mul(acb_core.acb_mul(a02, a11), a20),
                acb_core.acb_mul(acb_core.acb_mul(a01, a10), a22),
                acb_core.acb_mul(acb_core.acb_mul(a00, a12), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    return acb_core.acb_sub(pos, neg)


def as_dense_matvec_plan(plan: DenseMatvecPlan | jax.Array, *, algebra: str, label: str) -> DenseMatvecPlan:
    if isinstance(plan, DenseMatvecPlan):
        checks.check(plan.algebra == algebra, f"{label}.algebra")
        return plan
    if algebra == "arb":
        matrix = as_interval_rect_matrix(plan, label)
    elif algebra == "acb":
        matrix = as_box_rect_matrix(plan, label)
    else:
        raise ValueError("algebra must be one of: arb, acb")
    return DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-3]), cols=int(matrix.shape[-2]), algebra=algebra)


def dense_matvec_plan_from_matrix(matrix: jax.Array, *, algebra: str, label: str) -> DenseMatvecPlan:
    if algebra == "arb":
        matrix = as_interval_rect_matrix(matrix, label)
    elif algebra == "acb":
        matrix = as_box_rect_matrix(matrix, label)
    else:
        raise ValueError("algebra must be one of: arb, acb")
    return DenseMatvecPlan(matrix=matrix, rows=int(matrix.shape[-3]), cols=int(matrix.shape[-2]), algebra=algebra)


def as_dense_lu_solve_plan(plan: DenseLUSolvePlan | tuple[jax.Array, jax.Array, jax.Array], *, algebra: str, label: str) -> DenseLUSolvePlan:
    if isinstance(plan, DenseLUSolvePlan):
        checks.check(plan.algebra == algebra, f"{label}.algebra")
        return plan
    checks.check(isinstance(plan, tuple) and len(plan) == 3, f"{label}.tuple")
    p, l, u = plan
    if algebra == "arb":
        p = as_interval_matrix(p, f"{label}.p")
        l = as_interval_matrix(l, f"{label}.l")
        u = as_interval_matrix(u, f"{label}.u")
    elif algebra == "acb":
        p = as_box_matrix(p, f"{label}.p")
        l = as_box_matrix(l, f"{label}.l")
        u = as_box_matrix(u, f"{label}.u")
    else:
        raise ValueError("algebra must be one of: arb, acb")
    checks.check_equal(p.shape[-3], l.shape[-3], f"{label}.rows")
    checks.check_equal(p.shape[-3], u.shape[-3], f"{label}.rows_u")
    return DenseLUSolvePlan(p=p, l=l, u=u, rows=int(p.shape[-3]), algebra=algebra)


def dense_lu_solve_plan_from_factors(
    p: jax.Array,
    l: jax.Array,
    u: jax.Array,
    *,
    algebra: str,
    label: str,
) -> DenseLUSolvePlan:
    return as_dense_lu_solve_plan((p, l, u), algebra=algebra, label=label)


def as_dense_cholesky_solve_plan(
    plan: DenseCholeskySolvePlan | jax.Array,
    *,
    algebra: str,
    structure: str,
    label: str,
) -> DenseCholeskySolvePlan:
    if isinstance(plan, DenseCholeskySolvePlan):
        checks.check(plan.algebra == algebra, f"{label}.algebra")
        checks.check(plan.structure == structure, f"{label}.structure")
        return plan
    if algebra == "arb":
        factor = as_interval_matrix(plan, label)
    elif algebra == "acb":
        factor = as_box_matrix(plan, label)
    else:
        raise ValueError("algebra must be one of: arb, acb")
    return DenseCholeskySolvePlan(factor=factor, rows=int(factor.shape[-3]), algebra=algebra, structure=structure)


def dense_cholesky_solve_plan_from_factor(
    factor: jax.Array,
    *,
    algebra: str,
    structure: str,
    label: str,
) -> DenseCholeskySolvePlan:
    return as_dense_cholesky_solve_plan(factor, algebra=algebra, structure=structure, label=label)


def pad_batch_repeat_last(args: tuple, *, pad_to: int):
    return kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)


def estimator_mean(probes: jax.Array, coerce_probes, integrand_fn, *, probe_midpoint=None) -> jax.Array:
    coerced = coerce_probes(probes)
    vals = jax.vmap(integrand_fn)(coerced)
    estimate = jnp.mean(vals)
    if probe_midpoint is None:
        return estimate

    mids = jnp.asarray(probe_midpoint(coerced))
    if mids.ndim != 2:
        return estimate
    if mids.shape[0] != mids.shape[1]:
        return estimate

    gram = mids @ jnp.conjugate(mids).T
    diag = jnp.real(jnp.diagonal(gram))
    scale_ref = jnp.maximum(jnp.max(jnp.abs(diag)), jnp.asarray(1.0, dtype=diag.dtype))
    mean_diag = jnp.mean(diag)
    offdiag = gram - jnp.diag(jnp.diagonal(gram))
    orthogonal = (jnp.max(jnp.abs(offdiag)) <= 1e-10 * scale_ref) & (jnp.max(jnp.abs(diag - mean_diag)) <= 1e-10 * scale_ref)
    safe_diag = jnp.maximum(mean_diag, jnp.asarray(1e-30, dtype=diag.dtype))
    scale = jnp.asarray(mids.shape[-1], dtype=estimate.dtype) / jnp.asarray(safe_diag, dtype=estimate.dtype)
    orthogonal = lax.stop_gradient(orthogonal)
    scale = lax.stop_gradient(scale)
    return jnp.where(orthogonal, scale * estimate, estimate)


def action_with_diagnostics(action_fn, diagnostics_fn, x: jax.Array):
    y = action_fn(x)
    diag = diagnostics_fn(x)
    return y, diag


def estimator_with_diagnostics(
    probes: jax.Array,
    *,
    coerce_probes,
    estimator_fn,
    diagnostics_fn,
    algorithm_code: int,
    steps: int | None = None,
    basis_dim: int | None = None,
):
    coerced = coerce_probes(probes)
    value = estimator_fn(coerced)
    first = coerced[0]
    diag = diagnostics_fn(first)
    diag = diag._replace(
        algorithm_code=jnp.asarray(algorithm_code, dtype=jnp.int32),
        probe_count=jnp.asarray(coerced.shape[0], dtype=jnp.int32),
    )
    if steps is not None:
        diag = diag._replace(steps=jnp.asarray(steps, dtype=jnp.int32))
    if basis_dim is not None:
        diag = diag._replace(basis_dim=jnp.asarray(basis_dim, dtype=jnp.int32))
    return value, diag


__all__ = [
    "DenseMatvecPlan",
    "DenseLUSolvePlan",
    "DenseCholeskySolvePlan",
    "as_interval_mat_2x2",
    "as_interval_matrix",
    "as_interval_rect_matrix",
    "as_interval_vector",
    "as_interval_rhs",
    "as_box_mat_2x2",
    "as_box_matrix",
    "as_box_rect_matrix",
    "as_box_vector",
    "as_box_rhs",
    "full_interval_like",
    "full_box_like",
    "interval_from_point",
    "box_from_point",
    "interval_is_finite",
    "box_is_finite",
    "complex_is_finite",
    "interval_sum",
    "box_sum",
    "real_midpoint_symmetric_part",
    "complex_midpoint_hermitian_part",
    "real_midpoint_is_symmetric",
    "complex_midpoint_is_hermitian",
    "lower_cholesky_finite",
    "lower_cholesky_solve",
    "interval_trace",
    "box_trace",
    "as_dense_matvec_plan",
    "dense_matvec_plan_from_matrix",
    "as_dense_lu_solve_plan",
    "dense_lu_solve_plan_from_factors",
    "as_dense_cholesky_solve_plan",
    "dense_cholesky_solve_plan_from_factor",
    "pad_batch_repeat_last",
    "estimator_mean",
    "action_with_diagnostics",
    "estimator_with_diagnostics",
    "interval_det_2x2",
    "box_det_2x2",
    "interval_det_3x3",
    "box_det_3x3",
]
