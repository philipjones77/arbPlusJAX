from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import checks

jax.config.update("jax_enable_x64", True)


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _acb_from_complex(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def _as_coeffs(coeffs: jax.Array) -> jax.Array:
    arr = acb_core.as_acb_box(coeffs)
    checks.check_tail_shape(arr, (4, 4), "acb_poly._as_coeffs")
    return arr


def acb_poly_eval_cubic(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    c = acb_core.acb_midpoint(coeffs)
    zz = acb_core.acb_midpoint(z)
    v = c[..., 3]
    v = v * zz + c[..., 2]
    v = v * zz + c[..., 1]
    v = v * zz + c[..., 0]
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(z))


def acb_poly_eval_cubic_rigorous(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    c0 = coeffs[..., 0, :]
    c1 = coeffs[..., 1, :]
    c2 = coeffs[..., 2, :]
    c3 = coeffs[..., 3, :]
    v = acb_core.acb_add(acb_core.acb_mul(c3, z), c2)
    v = acb_core.acb_add(acb_core.acb_mul(v, z), c1)
    v = acb_core.acb_add(acb_core.acb_mul(v, z), c0)
    finite = jnp.isfinite(acb_core.acb_real(v)[..., 0]) & jnp.isfinite(acb_core.acb_real(v)[..., 1])
    finite = finite & jnp.isfinite(acb_core.acb_imag(v)[..., 0]) & jnp.isfinite(acb_core.acb_imag(v)[..., 1])
    return jnp.where(finite[..., None], v, _full_box_like(z))


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_poly_eval_cubic_prec(
    coeffs: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_poly_eval_cubic(coeffs, z), prec_bits)


def acb_poly_eval_cubic_batch(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    return jax.vmap(acb_poly_eval_cubic)(coeffs, z)


def acb_poly_eval_cubic_batch_rigorous(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    coeffs = _as_coeffs(coeffs)
    z = acb_core.as_acb_box(z)
    return jax.vmap(acb_poly_eval_cubic_rigorous)(coeffs, z)


def acb_poly_eval_cubic_batch_prec(
    coeffs: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_poly_eval_cubic_batch(coeffs, z), prec_bits)


acb_poly_eval_cubic_batch_jit = jax.jit(acb_poly_eval_cubic_batch)
acb_poly_eval_cubic_batch_prec_jit = jax.jit(
    acb_poly_eval_cubic_batch_prec, static_argnames=("prec_bits",)
)


__all__ = [
    "acb_poly_eval_cubic",
    "acb_poly_eval_cubic_rigorous",
    "acb_poly_eval_cubic_prec",
    "acb_poly_eval_cubic_batch",
    "acb_poly_eval_cubic_batch_rigorous",
    "acb_poly_eval_cubic_batch_prec",
    "acb_poly_eval_cubic_batch_jit",
    "acb_poly_eval_cubic_batch_prec_jit",
]


from . import series_missing_impl as _smi
for _name in dir(_smi):
    if _name in globals():
        continue
    if any(_name.startswith(p) for p in ['acb_poly_', '_acb_poly_']):
        globals()[_name] = getattr(_smi, _name)
        if '__all__' in globals():
            __all__.append(_name)
