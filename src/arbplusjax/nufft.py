from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

from . import checks
from . import dft
from . import elementary as el
from . import kernel_helpers as kh
from . import transform_common as tc
from .lazy_jit import lazy_jit


def _as_unit_points_matrix(x: jax.Array, ndim: int) -> jax.Array:
    return tc.as_unit_points_matrix(x, ndim, "nufft._as_unit_points_matrix")


def _as_complex_vector(x: jax.Array) -> jax.Array:
    return tc.as_complex_vector(x, "nufft._as_complex_vector")


def _as_complex_modes_grid(x: jax.Array, ndim: int | None = None) -> jax.Array:
    return tc.as_complex_modes_grid(x, "nufft._as_complex_modes_grid", ndim=ndim)


def nufft_good_size(n_modes: int, oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP) -> int:
    n = int(n_modes)
    if n <= 0:
        raise ValueError("nufft_good_size requires n_modes >= 1")
    target = max(n, int(math.ceil(float(oversamp) * n)))
    return dft.dft_good_size(target)


def nufft_good_shape(
    mode_shape: tuple[int, ...],
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
) -> tuple[int, ...]:
    shape = tuple(int(m) for m in mode_shape)
    if len(shape) not in (1, 2, 3):
        raise ValueError("mode_shape must be 1D, 2D, or 3D")
    return tuple(nufft_good_size(m, oversamp=oversamp) for m in shape)


def _resolve_method(n_points: int, mode_shape: tuple[int, ...], method: str) -> str:
    total_modes = math.prod(int(m) for m in mode_shape)
    if method == "auto":
        return "direct" if n_points * total_modes <= tc.DEFAULT_NUFFT_DIRECT_THRESHOLD else "lanczos"
    if method not in {"direct", "lanczos"}:
        raise ValueError("method must be 'auto', 'direct', or 'lanczos'")
    return method


def _pack_stencils(
    stencils: list[tuple[jax.Array, jax.Array]],
    n_points: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    empty_idx, empty_w = tc.empty_stencil_arrays(n_points)
    pairs = stencils + [(empty_idx, empty_w)] * (3 - len(stencils))
    (idx0, w0), (idx1, w1), (idx2, w2) = pairs[:3]
    return idx0, w0, idx1, w1, idx2, w2


def _plan_diagnostics(plan: tc.NufftMatvecPlan) -> dict[str, int | float | str | tuple[int, ...]]:
    diagnostics: dict[str, int | float | str | tuple[int, ...]] = {
        "method": "direct" if plan.use_direct else "lanczos",
        "n_points": int(plan.points.shape[0]),
        "mode_shape": plan.mode_shape,
        "ndim": int(plan.ndim),
    }
    if not plan.use_direct:
        diagnostics["grid_shape"] = plan.grid_shape
        diagnostics["oversamp"] = float(plan.oversamp)
        diagnostics["kernel_width"] = int(plan.kernel_width)
    return diagnostics


def nufft_type1_nd_cached_prepare(
    points: jax.Array,
    mode_shape: tuple[int, ...],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tc.NufftMatvecPlan:
    shape = tuple(int(m) for m in mode_shape)
    pts = _as_unit_points_matrix(points, len(shape))
    chosen = _resolve_method(pts.shape[0], shape, method)
    if chosen == "direct":
        empty_idx, empty_w = tc.empty_stencil_arrays(pts.shape[0])
        return tc.NufftMatvecPlan(
            points=pts,
            mode_coords=tc.mode_grid(shape),
            idx0=empty_idx,
            w0=empty_w,
            idx1=empty_idx,
            w1=empty_w,
            idx2=empty_idx,
            w2=empty_w,
            mode_shape=shape,
            grid_shape=shape,
            ndim=len(shape),
            use_direct=True,
            oversamp=float(oversamp),
            kernel_width=int(kernel_width),
        )
    grid_shape = nufft_good_shape(shape, oversamp=oversamp)
    stencils = tc.lanczos_stencils(pts, grid_shape, int(kernel_width), "nufft.nufft_type1_nd_cached_prepare")
    idx0, w0, idx1, w1, idx2, w2 = _pack_stencils(stencils, pts.shape[0])
    return tc.NufftMatvecPlan(
        points=pts,
        mode_coords=jnp.zeros((0, len(shape)), dtype=tc.TRANSFORM_REAL_DTYPE),
        idx0=idx0,
        w0=w0,
        idx1=idx1,
        w1=w1,
        idx2=idx2,
        w2=w2,
        mode_shape=shape,
        grid_shape=grid_shape,
        ndim=len(shape),
        use_direct=False,
        oversamp=float(oversamp),
        kernel_width=int(kernel_width),
    )


def nufft_type2_nd_cached_prepare(
    points: jax.Array,
    mode_shape: tuple[int, ...],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tc.NufftMatvecPlan:
    return nufft_type1_nd_cached_prepare(
        points,
        mode_shape,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


@jax.jit
def nufft_type1_cached_apply(plan: tc.NufftMatvecPlan, values: jax.Array) -> jax.Array:
    c = _as_complex_vector(values)
    checks.check_equal(plan.points.shape[0], c.shape[0], "nufft.nufft_type1_cached_apply")
    if plan.use_direct:
        phase = plan.points @ plan.mode_coords.T
        expo = jnp.exp(-2.0j * el.PI * phase)
        return (expo.T @ c).reshape(plan.mode_shape)

    if plan.ndim == 1:
        spread = jnp.zeros((plan.grid_shape[0],), dtype=tc.TRANSFORM_COMPLEX_DTYPE)
        spread = spread.at[plan.idx0].add(plan.w0 * c[:, None])
        return jnp.fft.fft(spread)[: plan.mode_shape[0]]

    if plan.ndim == 2:
        n_points, stencil_len = plan.idx0.shape
        i0 = jnp.broadcast_to(plan.idx0[:, :, None], (n_points, stencil_len, stencil_len))
        i1 = jnp.broadcast_to(plan.idx1[:, None, :], (n_points, stencil_len, stencil_len))
        weights = plan.w0[:, :, None] * plan.w1[:, None, :]
        spread = jnp.zeros(plan.grid_shape, dtype=tc.TRANSFORM_COMPLEX_DTYPE)
        spread = spread.at[i0, i1].add(weights * c[:, None, None])
        return jnp.fft.fftn(spread)[: plan.mode_shape[0], : plan.mode_shape[1]]

    n_points, stencil_len = plan.idx0.shape
    i0 = jnp.broadcast_to(plan.idx0[:, :, None, None], (n_points, stencil_len, stencil_len, stencil_len))
    i1 = jnp.broadcast_to(plan.idx1[:, None, :, None], (n_points, stencil_len, stencil_len, stencil_len))
    i2 = jnp.broadcast_to(plan.idx2[:, None, None, :], (n_points, stencil_len, stencil_len, stencil_len))
    weights = plan.w0[:, :, None, None] * plan.w1[:, None, :, None] * plan.w2[:, None, None, :]
    spread = jnp.zeros(plan.grid_shape, dtype=tc.TRANSFORM_COMPLEX_DTYPE)
    spread = spread.at[i0, i1, i2].add(weights * c[:, None, None, None])
    return jnp.fft.fftn(spread)[: plan.mode_shape[0], : plan.mode_shape[1], : plan.mode_shape[2]]


@jax.jit
def nufft_type2_cached_apply(plan: tc.NufftMatvecPlan, modes: jax.Array) -> jax.Array:
    f = _as_complex_modes_grid(modes, ndim=plan.ndim)
    checks.check(f.shape == plan.mode_shape, "nufft.nufft_type2_cached_apply.shape")
    if plan.use_direct:
        phase = plan.points @ plan.mode_coords.T
        expo = jnp.exp(2.0j * el.PI * phase)
        return expo @ f.reshape(-1)

    grid_scale = float(math.prod(plan.grid_shape))

    if plan.ndim == 1:
        padded = jnp.zeros((plan.grid_shape[0],), dtype=tc.TRANSFORM_COMPLEX_DTYPE).at[: f.shape[0]].set(f)
        grid = grid_scale * jnp.fft.ifft(padded)
        return jnp.sum(plan.w0 * grid[plan.idx0], axis=1)

    if plan.ndim == 2:
        padded = jnp.zeros(plan.grid_shape, dtype=tc.TRANSFORM_COMPLEX_DTYPE).at[: f.shape[0], : f.shape[1]].set(f)
        grid = grid_scale * jnp.fft.ifftn(padded)
        n_points, stencil_len = plan.idx0.shape
        i0 = jnp.broadcast_to(plan.idx0[:, :, None], (n_points, stencil_len, stencil_len))
        i1 = jnp.broadcast_to(plan.idx1[:, None, :], (n_points, stencil_len, stencil_len))
        weights = plan.w0[:, :, None] * plan.w1[:, None, :]
        return jnp.sum(weights * grid[i0, i1], axis=(1, 2))

    padded = jnp.zeros(plan.grid_shape, dtype=tc.TRANSFORM_COMPLEX_DTYPE).at[: f.shape[0], : f.shape[1], : f.shape[2]].set(f)
    grid = grid_scale * jnp.fft.ifftn(padded)
    n_points, stencil_len = plan.idx0.shape
    i0 = jnp.broadcast_to(plan.idx0[:, :, None, None], (n_points, stencil_len, stencil_len, stencil_len))
    i1 = jnp.broadcast_to(plan.idx1[:, None, :, None], (n_points, stencil_len, stencil_len, stencil_len))
    i2 = jnp.broadcast_to(plan.idx2[:, None, None, :], (n_points, stencil_len, stencil_len, stencil_len))
    weights = plan.w0[:, :, None, None] * plan.w1[:, None, :, None] * plan.w2[:, None, None, :]
    return jnp.sum(weights * grid[i0, i1, i2], axis=(1, 2, 3))


def nufft_type1_cached_apply_with_diagnostics(
    plan: tc.NufftMatvecPlan,
    values: jax.Array,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    return nufft_type1_cached_apply(plan, values), _plan_diagnostics(plan)


def nufft_type2_cached_apply_with_diagnostics(
    plan: tc.NufftMatvecPlan,
    modes: jax.Array,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    return nufft_type2_cached_apply(plan, modes), _plan_diagnostics(plan)


def nufft_type1_cached_apply_batch_fixed(plan: tc.NufftMatvecPlan, values: jax.Array) -> jax.Array:
    vals = tc.as_complex_array(values, "nufft.nufft_type1_cached_apply_batch_fixed")
    checks.check_ndim(vals, 2, "nufft.nufft_type1_cached_apply_batch_fixed")
    return jax.vmap(lambda row: nufft_type1_cached_apply(plan, row))(vals)


def nufft_type1_cached_apply_batch_padded(plan: tc.NufftMatvecPlan, values: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last(
        (tc.as_complex_array(values, "nufft.nufft_type1_cached_apply_batch_padded"),),
        pad_to=pad_to,
    )
    out = nufft_type1_cached_apply_batch_fixed(plan, *call_args)
    return kh.trim_batch_out(out, trim_n)


def nufft_type2_cached_apply_batch_fixed(plan: tc.NufftMatvecPlan, modes: jax.Array) -> jax.Array:
    grids = tc.as_complex_array(modes, "nufft.nufft_type2_cached_apply_batch_fixed")
    checks.check_ndim(grids, plan.ndim + 1, "nufft.nufft_type2_cached_apply_batch_fixed")
    checks.check(grids.shape[1:] == plan.mode_shape, "nufft.nufft_type2_cached_apply_batch_fixed.shape")
    return jax.vmap(lambda grid: nufft_type2_cached_apply(plan, grid))(grids)


def nufft_type2_cached_apply_batch_padded(plan: tc.NufftMatvecPlan, modes: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last(
        (tc.as_complex_array(modes, "nufft.nufft_type2_cached_apply_batch_padded"),),
        pad_to=pad_to,
    )
    out = nufft_type2_cached_apply_batch_fixed(plan, *call_args)
    return kh.trim_batch_out(out, trim_n)


@partial(jax.jit, static_argnames=("mode_shape",))
def nufft_type1_nd_direct(points: jax.Array, values: jax.Array, mode_shape: tuple[int, ...]) -> jax.Array:
    plan = nufft_type1_nd_cached_prepare(points, mode_shape, method="direct")
    return nufft_type1_cached_apply(plan, values)


@jax.jit
def nufft_type2_nd_direct(points: jax.Array, modes: jax.Array) -> jax.Array:
    f = _as_complex_modes_grid(modes)
    plan = nufft_type2_nd_cached_prepare(points, tuple(int(m) for m in f.shape), method="direct")
    return nufft_type2_cached_apply(plan, f)


def nufft_type1_nd(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, ...],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    plan = nufft_type1_nd_cached_prepare(
        points,
        mode_shape,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type1_cached_apply(plan, values)


def nufft_type2_nd(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    f = _as_complex_modes_grid(modes)
    plan = nufft_type2_nd_cached_prepare(
        points,
        tuple(int(m) for m in f.shape),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type2_cached_apply(plan, f)


def nufft_type1_nd_with_diagnostics(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, ...],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    plan = nufft_type1_nd_cached_prepare(
        points,
        mode_shape,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type1_cached_apply(plan, values), _plan_diagnostics(plan)


def nufft_type2_nd_with_diagnostics(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    f = _as_complex_modes_grid(modes)
    plan = nufft_type2_nd_cached_prepare(
        points,
        tuple(int(m) for m in f.shape),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type2_cached_apply(plan, f), _plan_diagnostics(plan)


def nufft_type1_nd_batch_fixed(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, ...],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    plan = nufft_type1_nd_cached_prepare(
        points,
        mode_shape,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type1_cached_apply_batch_fixed(plan, values)


def nufft_type1_nd_batch_padded(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, ...],
    *,
    pad_to: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    plan = nufft_type1_nd_cached_prepare(
        points,
        mode_shape,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type1_cached_apply_batch_padded(plan, values, pad_to=pad_to)


def nufft_type2_nd_batch_fixed(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    grids = tc.as_complex_array(modes, "nufft.nufft_type2_nd_batch_fixed")
    checks.check(grids.ndim >= 2, "nufft.nufft_type2_nd_batch_fixed")
    plan = nufft_type2_nd_cached_prepare(
        points,
        tuple(int(m) for m in grids.shape[1:]),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type2_cached_apply_batch_fixed(plan, grids)


def nufft_type2_nd_batch_padded(
    points: jax.Array,
    modes: jax.Array,
    *,
    pad_to: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    grids = tc.as_complex_array(modes, "nufft.nufft_type2_nd_batch_padded")
    checks.check(grids.ndim >= 2, "nufft.nufft_type2_nd_batch_padded")
    plan = nufft_type2_nd_cached_prepare(
        points,
        tuple(int(m) for m in grids.shape[1:]),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    return nufft_type2_cached_apply_batch_padded(plan, grids, pad_to=pad_to)


@partial(jax.jit, static_argnames=("n_modes",))
def nufft_type1_direct(points: jax.Array, values: jax.Array, n_modes: int) -> jax.Array:
    return nufft_type1_nd_direct(points, values, (int(n_modes),))


@jax.jit
def nufft_type2_direct(points: jax.Array, modes: jax.Array) -> jax.Array:
    return nufft_type2_nd_direct(points, modes)


def nufft_type1(
    points: jax.Array,
    values: jax.Array,
    n_modes: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type1_nd(points, values, (int(n_modes),), method=method, oversamp=oversamp, kernel_width=kernel_width)


def nufft_type2(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type2_nd(points, modes, method=method, oversamp=oversamp, kernel_width=kernel_width)


def nufft_type1_with_diagnostics(
    points: jax.Array,
    values: jax.Array,
    n_modes: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    result, diagnostics = nufft_type1_nd_with_diagnostics(
        points,
        values,
        (int(n_modes),),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    diagnostics["n_modes"] = int(n_modes)
    if "grid_shape" in diagnostics:
        diagnostics["grid_size"] = int(diagnostics["grid_shape"][0])
    return result, diagnostics


def nufft_type2_with_diagnostics(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tuple[jax.Array, dict[str, int | float | str | tuple[int, ...]]]:
    result, diagnostics = nufft_type2_nd_with_diagnostics(
        points,
        modes,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )
    diagnostics["n_modes"] = int(jnp.asarray(modes).shape[0])
    if "grid_shape" in diagnostics:
        diagnostics["grid_size"] = int(diagnostics["grid_shape"][0])
    return result, diagnostics


def nufft_type1_cached_prepare(
    points: jax.Array,
    n_modes: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tc.NufftMatvecPlan:
    return nufft_type1_nd_cached_prepare(
        points,
        (int(n_modes),),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type2_cached_prepare(
    points: jax.Array,
    n_modes: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> tc.NufftMatvecPlan:
    return nufft_type2_nd_cached_prepare(
        points,
        (int(n_modes),),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type1_batch_fixed(
    points: jax.Array,
    values: jax.Array,
    n_modes: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type1_nd_batch_fixed(
        points,
        values,
        (int(n_modes),),
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type1_batch_padded(
    points: jax.Array,
    values: jax.Array,
    n_modes: int,
    *,
    pad_to: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type1_nd_batch_padded(
        points,
        values,
        (int(n_modes),),
        pad_to=pad_to,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type2_batch_fixed(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type2_nd_batch_fixed(
        points,
        modes,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type2_batch_padded(
    points: jax.Array,
    modes: jax.Array,
    *,
    pad_to: int,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type2_nd_batch_padded(
        points,
        modes,
        pad_to=pad_to,
        method=method,
        oversamp=oversamp,
        kernel_width=kernel_width,
    )


def nufft_type1_2d_direct(points: jax.Array, values: jax.Array, mode_shape: tuple[int, int]) -> jax.Array:
    return nufft_type1_nd_direct(points, values, tuple(int(m) for m in mode_shape))


def nufft_type2_2d_direct(points: jax.Array, modes: jax.Array) -> jax.Array:
    return nufft_type2_nd_direct(points, _as_complex_modes_grid(modes, ndim=2))


def nufft_type1_2d(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, int],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type1_nd(points, values, tuple(int(m) for m in mode_shape), method=method, oversamp=oversamp, kernel_width=kernel_width)


def nufft_type2_2d(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type2_nd(points, _as_complex_modes_grid(modes, ndim=2), method=method, oversamp=oversamp, kernel_width=kernel_width)


def nufft_type1_3d_direct(points: jax.Array, values: jax.Array, mode_shape: tuple[int, int, int]) -> jax.Array:
    return nufft_type1_nd_direct(points, values, tuple(int(m) for m in mode_shape))


def nufft_type2_3d_direct(points: jax.Array, modes: jax.Array) -> jax.Array:
    return nufft_type2_nd_direct(points, _as_complex_modes_grid(modes, ndim=3))


def nufft_type1_3d(
    points: jax.Array,
    values: jax.Array,
    mode_shape: tuple[int, int, int],
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type1_nd(points, values, tuple(int(m) for m in mode_shape), method=method, oversamp=oversamp, kernel_width=kernel_width)


def nufft_type2_3d(
    points: jax.Array,
    modes: jax.Array,
    method: str = "auto",
    oversamp: float = tc.DEFAULT_NUFFT_OVERSAMP,
    kernel_width: int = tc.DEFAULT_NUFFT_KERNEL_WIDTH,
) -> jax.Array:
    return nufft_type2_nd(points, _as_complex_modes_grid(modes, ndim=3), method=method, oversamp=oversamp, kernel_width=kernel_width)


nufft_type1_jit = lazy_jit(lambda: jax.jit(nufft_type1, static_argnames=("n_modes", "method", "oversamp", "kernel_width")))
nufft_type2_jit = lazy_jit(lambda: jax.jit(nufft_type2, static_argnames=("method", "oversamp", "kernel_width")))
nufft_type1_nd_jit = lazy_jit(lambda: jax.jit(nufft_type1_nd, static_argnames=("mode_shape", "method", "oversamp", "kernel_width")))
nufft_type2_nd_jit = lazy_jit(lambda: jax.jit(nufft_type2_nd, static_argnames=("method", "oversamp", "kernel_width")))
nufft_type1_cached_apply_jit = lazy_jit(lambda: jax.jit(nufft_type1_cached_apply))
nufft_type2_cached_apply_jit = lazy_jit(lambda: jax.jit(nufft_type2_cached_apply))
nufft_type1_cached_apply_batch_fixed_jit = lazy_jit(lambda: jax.jit(nufft_type1_cached_apply_batch_fixed))
nufft_type2_cached_apply_batch_fixed_jit = lazy_jit(lambda: jax.jit(nufft_type2_cached_apply_batch_fixed))


__all__ = [
    "nufft_good_size",
    "nufft_good_shape",
    "nufft_type1_direct",
    "nufft_type2_direct",
    "nufft_type1",
    "nufft_type2",
    "nufft_type1_with_diagnostics",
    "nufft_type2_with_diagnostics",
    "nufft_type1_nd_direct",
    "nufft_type2_nd_direct",
    "nufft_type1_nd",
    "nufft_type2_nd",
    "nufft_type1_nd_with_diagnostics",
    "nufft_type2_nd_with_diagnostics",
    "nufft_type1_nd_cached_prepare",
    "nufft_type2_nd_cached_prepare",
    "nufft_type1_cached_prepare",
    "nufft_type2_cached_prepare",
    "nufft_type1_cached_apply",
    "nufft_type2_cached_apply",
    "nufft_type1_cached_apply_with_diagnostics",
    "nufft_type2_cached_apply_with_diagnostics",
    "nufft_type1_cached_apply_batch_fixed",
    "nufft_type1_cached_apply_batch_padded",
    "nufft_type2_cached_apply_batch_fixed",
    "nufft_type2_cached_apply_batch_padded",
    "nufft_type1_batch_fixed",
    "nufft_type1_batch_padded",
    "nufft_type2_batch_fixed",
    "nufft_type2_batch_padded",
    "nufft_type1_nd_batch_fixed",
    "nufft_type1_nd_batch_padded",
    "nufft_type2_nd_batch_fixed",
    "nufft_type2_nd_batch_padded",
    "nufft_type1_2d_direct",
    "nufft_type2_2d_direct",
    "nufft_type1_2d",
    "nufft_type2_2d",
    "nufft_type1_3d_direct",
    "nufft_type2_3d_direct",
    "nufft_type1_3d",
    "nufft_type2_3d",
    "nufft_type1_jit",
    "nufft_type2_jit",
    "nufft_type1_nd_jit",
    "nufft_type2_nd_jit",
    "nufft_type1_cached_apply_jit",
    "nufft_type2_cached_apply_jit",
    "nufft_type1_cached_apply_batch_fixed_jit",
    "nufft_type2_cached_apply_batch_fixed_jit",
]
