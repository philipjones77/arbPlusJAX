from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di


TRANSFORM_REAL_DTYPE = jnp.float64
TRANSFORM_COMPLEX_DTYPE = jnp.complex128
DEFAULT_NUFFT_OVERSAMP = 4.0
DEFAULT_NUFFT_KERNEL_WIDTH = 8
DEFAULT_NUFFT_DIRECT_THRESHOLD = 4096


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DftMatvecPlan:
    chirp: jax.Array
    kernel_fft: jax.Array
    length: int
    inverse: bool

    def tree_flatten(self):
        return (self.chirp, self.kernel_fft), {"length": self.length, "inverse": self.inverse}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        chirp, kernel_fft = children
        return cls(chirp=chirp, kernel_fft=kernel_fft, length=aux_data["length"], inverse=aux_data["inverse"])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NufftMatvecPlan:
    points: jax.Array
    mode_coords: jax.Array
    idx0: jax.Array
    w0: jax.Array
    idx1: jax.Array
    w1: jax.Array
    idx2: jax.Array
    w2: jax.Array
    mode_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    ndim: int
    use_direct: bool
    oversamp: float
    kernel_width: int

    def tree_flatten(self):
        children = (self.points, self.mode_coords, self.idx0, self.w0, self.idx1, self.w1, self.idx2, self.w2)
        aux = {
            "mode_shape": self.mode_shape,
            "grid_shape": self.grid_shape,
            "ndim": self.ndim,
            "use_direct": self.use_direct,
            "oversamp": self.oversamp,
            "kernel_width": self.kernel_width,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        points, mode_coords, idx0, w0, idx1, w1, idx2, w2 = children
        return cls(
            points=points,
            mode_coords=mode_coords,
            idx0=idx0,
            w0=w0,
            idx1=idx1,
            w1=w1,
            idx2=idx2,
            w2=w2,
            mode_shape=aux_data["mode_shape"],
            grid_shape=aux_data["grid_shape"],
            ndim=aux_data["ndim"],
            use_direct=aux_data["use_direct"],
            oversamp=aux_data["oversamp"],
            kernel_width=aux_data["kernel_width"],
        )


def as_complex_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=TRANSFORM_COMPLEX_DTYPE)
    checks.check_ndim(arr, 1, label)
    return arr


def as_complex_array(x: jax.Array, label: str) -> jax.Array:
    return jnp.asarray(x, dtype=TRANSFORM_COMPLEX_DTYPE)


def as_real_array(x: jax.Array, label: str) -> jax.Array:
    del label
    return jnp.asarray(x, dtype=TRANSFORM_REAL_DTYPE)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def canonical_axes(ndim: int, axes: tuple[int, ...] | None, label: str) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    out = tuple(int(ax) % ndim for ax in axes)
    if len(set(out)) != len(out):
        raise ValueError(f"{label}: axes must be unique")
    return out


def smooth_good_size(length: int) -> int:
    n = max(int(length), 1)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def as_unit_points_matrix(x: jax.Array, ndim: int, label: str) -> jax.Array:
    arr = jnp.asarray(x, dtype=TRANSFORM_REAL_DTYPE)
    if ndim == 1 and arr.ndim == 1:
        arr = arr[:, None]
    checks.check_ndim(arr, 2, label)
    checks.check_last_dim(arr, ndim, label)
    return jnp.mod(arr, 1.0)


def as_complex_modes_grid(x: jax.Array, label: str, ndim: int | None = None) -> jax.Array:
    arr = jnp.asarray(x, dtype=TRANSFORM_COMPLEX_DTYPE)
    if ndim is not None:
        checks.check_ndim(arr, ndim, label)
    return arr


def mode_grid(mode_shape: tuple[int, ...]) -> jax.Array:
    axes = [jnp.arange(int(m), dtype=TRANSFORM_REAL_DTYPE) for m in mode_shape]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack(mesh, axis=-1).reshape((-1, len(mode_shape)))


def lanczos_kernel(offsets: jax.Array, kernel_width: int) -> jax.Array:
    x = jnp.asarray(offsets, dtype=TRANSFORM_REAL_DTYPE)
    width = float(kernel_width)
    ax = jnp.abs(x)
    weights = jnp.sinc(x) * jnp.sinc(x / width)
    return jnp.where(ax < width, weights, 0.0)


def lanczos_stencil_axis(points_axis: jax.Array, grid_size: int, kernel_width: int) -> tuple[jax.Array, jax.Array]:
    t = jnp.mod(jnp.asarray(points_axis, dtype=TRANSFORM_REAL_DTYPE), 1.0) * float(grid_size)
    center = jnp.floor(t + 0.5).astype(jnp.int64)
    offsets = jnp.arange(-kernel_width, kernel_width + 1, dtype=jnp.int64)
    idx = center[:, None] + offsets[None, :]
    dist = t[:, None] - idx.astype(TRANSFORM_REAL_DTYPE)
    weights = lanczos_kernel(dist, kernel_width)
    weights_sum = jnp.sum(weights, axis=1, keepdims=True)
    weights = jnp.where(weights_sum != 0.0, weights / weights_sum, weights)
    return idx % grid_size, weights


def lanczos_stencils(points: jax.Array, grid_shape: tuple[int, ...], kernel_width: int, label: str) -> list[tuple[jax.Array, jax.Array]]:
    pts = as_unit_points_matrix(points, len(grid_shape), label)
    return [lanczos_stencil_axis(pts[:, axis], int(grid_shape[axis]), kernel_width) for axis in range(len(grid_shape))]


def empty_stencil_arrays(n_points: int) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.zeros((int(n_points), 0), dtype=jnp.int64),
        jnp.zeros((int(n_points), 0), dtype=TRANSFORM_REAL_DTYPE),
    )


def point_box(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z, dtype=TRANSFORM_COMPLEX_DTYPE)
    return acb_core.acb_box(di.interval(jnp.real(zz), jnp.real(zz)), di.interval(jnp.imag(zz), jnp.imag(zz)))


def as_box_vector(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.ndim >= 1 and arr.shape[-1] == 4:
        out = acb_core.as_acb_box(arr)
    else:
        out = point_box(arr)
    checks.check_ndim(out, 2, label)
    checks.check_last_dim(out, 4, label)
    return out


def as_box_array(x: jax.Array, label: str) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.ndim >= 1 and arr.shape[-1] == 4:
        return acb_core.as_acb_box(arr)
    return point_box(arr)


def box_linear_apply(matrix: jax.Array, x_box: jax.Array) -> jax.Array:
    a = jnp.asarray(matrix, dtype=TRANSFORM_COMPLEX_DTYPE)
    xb = as_box_vector(x_box, "transform_common.box_linear_apply")
    checks.check_equal(a.shape[-1], xb.shape[-2], "transform_common.box_linear_apply.inner")
    midpoint = a @ acb_core.acb_midpoint(xb)
    re_half_width = 0.5 * jnp.maximum(acb_core.acb_real(xb)[..., 1] - acb_core.acb_real(xb)[..., 0], 0.0)
    im_half_width = 0.5 * jnp.maximum(acb_core.acb_imag(xb)[..., 1] - acb_core.acb_imag(xb)[..., 0], 0.0)
    abs_re = jnp.abs(jnp.real(a))
    abs_im = jnp.abs(jnp.imag(a))
    out_re_half_width = abs_re @ re_half_width + abs_im @ im_half_width
    out_im_half_width = abs_im @ re_half_width + abs_re @ im_half_width
    re_mid = jnp.real(midpoint)
    im_mid = jnp.imag(midpoint)
    return acb_core.acb_box(
        di.interval(re_mid - out_re_half_width, re_mid + out_re_half_width),
        di.interval(im_mid - out_im_half_width, im_mid + out_im_half_width),
    )


__all__ = [
    "TRANSFORM_REAL_DTYPE",
    "TRANSFORM_COMPLEX_DTYPE",
    "DEFAULT_NUFFT_OVERSAMP",
    "DEFAULT_NUFFT_KERNEL_WIDTH",
    "DEFAULT_NUFFT_DIRECT_THRESHOLD",
    "DftMatvecPlan",
    "NufftMatvecPlan",
    "as_complex_vector",
    "as_complex_array",
    "as_real_array",
    "is_power_of_two",
    "canonical_axes",
    "smooth_good_size",
    "as_unit_points_matrix",
    "as_complex_modes_grid",
    "mode_grid",
    "lanczos_kernel",
    "lanczos_stencil_axis",
    "lanczos_stencils",
    "empty_stencil_arrays",
    "point_box",
    "as_box_vector",
    "as_box_array",
    "box_linear_apply",
]
