from __future__ import annotations

import inspect
from typing import Callable

import jax
import jax.numpy as jnp

from . import acb_core
from . import arb_core
from . import ball_wrappers
from . import double_interval as di
from . import wrappers_common as wc

jax.config.update("jax_enable_x64", True)


def _resolve_prec_bits(dps: int | None, prec_bits: int | None) -> int:
    return wc.resolve_prec_bits(dps, prec_bits)


def _inflate_interval(x: jax.Array, prec_bits: int, adaptive: bool) -> jax.Array:
    return wc.inflate_interval(x, prec_bits, adaptive)


def _inflate_acb(x: jax.Array, prec_bits: int, adaptive: bool) -> jax.Array:
    return wc.inflate_acb(x, prec_bits, adaptive)


def _acb_exp_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    er = arb_core.arb_exp(re)
    cr = arb_core.arb_cos(im)
    sr = arb_core.arb_sin(im)
    out = acb_core.acb_box(di.fast_mul(er, cr), di.fast_mul(er, sr))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_sin_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    sx = arb_core.arb_sin(re)
    cx = arb_core.arb_cos(re)
    shy = arb_core.arb_sinh(im)
    chy = arb_core.arb_cosh(im)
    out = acb_core.acb_box(di.fast_mul(sx, chy), di.fast_mul(cx, shy))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_cos_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    sx = arb_core.arb_sin(re)
    cx = arb_core.arb_cos(re)
    shy = arb_core.arb_sinh(im)
    chy = arb_core.arb_cosh(im)
    out = acb_core.acb_box(di.fast_mul(cx, chy), di.neg(di.fast_mul(sx, shy)))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_sinh_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    shx = arb_core.arb_sinh(re)
    chx = arb_core.arb_cosh(re)
    sy = arb_core.arb_sin(im)
    cy = arb_core.arb_cos(im)
    out = acb_core.acb_box(di.fast_mul(shx, cy), di.fast_mul(chx, sy))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_cosh_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    shx = arb_core.arb_sinh(re)
    chx = arb_core.arb_cosh(re)
    sy = arb_core.arb_sin(im)
    cy = arb_core.arb_cos(im)
    out = acb_core.acb_box(di.fast_mul(chx, cy), di.fast_mul(shx, sy))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_tan_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    s = _acb_sin_rigorous(x, prec_bits)
    c = _acb_cos_rigorous(x, prec_bits)
    out = acb_core.acb_div(s, c)
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_tanh_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    s = _acb_sinh_rigorous(x, prec_bits)
    c = _acb_cosh_rigorous(x, prec_bits)
    out = acb_core.acb_div(s, c)
    return acb_core.acb_box_round_prec(out, prec_bits)


def _arg_interval(re: jax.Array, im: jax.Array) -> jax.Array:
    a, b = di.lower(re), di.upper(re)
    c, d = di.lower(im), di.upper(im)
    contains_zero = (a <= 0.0) & (b >= 0.0) & (c <= 0.0) & (d >= 0.0)
    crosses_neg_real = (a <= 0.0) & (c <= 0.0) & (d >= 0.0)
    full = di.interval(-jnp.pi, jnp.pi)

    def atan2_pair(xv, yv):
        return jnp.arctan2(yv, xv)

    angs = jnp.stack(
        [
            atan2_pair(a, c),
            atan2_pair(a, d),
            atan2_pair(b, c),
            atan2_pair(b, d),
        ],
        axis=-1,
    )
    lo = jnp.min(angs, axis=-1)
    hi = jnp.max(angs, axis=-1)
    out = di.interval(di._below(lo), di._above(hi))
    return jnp.where((contains_zero | crosses_neg_real)[..., None], full, out)


def _acb_log_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    r2 = di.fast_add(di.fast_mul(re, re), di.fast_mul(im, im))
    r = arb_core.arb_sqrt(r2)
    contains_zero = (di.lower(re) <= 0.0) & (di.upper(re) >= 0.0) & (di.lower(im) <= 0.0) & (di.upper(im) >= 0.0)
    log_r = jnp.where(contains_zero[..., None], di.interval(-jnp.inf, jnp.inf), arb_core.arb_log(r))
    arg = _arg_interval(re, im)
    out = acb_core.acb_box(log_r, arg)
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_sqrt_rigorous(x: jax.Array, prec_bits: int) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    r2 = di.fast_add(di.fast_mul(re, re), di.fast_mul(im, im))
    r = arb_core.arb_sqrt(r2)
    theta = _arg_interval(re, im)
    half = di.fast_mul(theta, di.interval(0.5, 0.5))
    ch = arb_core.arb_cos(half)
    sh = arb_core.arb_sin(half)
    out = acb_core.acb_box(di.fast_mul(r, ch), di.fast_mul(r, sh))
    return acb_core.acb_box_round_prec(out, prec_bits)


def _acb_rigorous_adapter(name: str) -> Callable[..., jax.Array] | None:
    if name in ("acb_exp_prec", "acb_exp_batch_prec"):
        return _acb_exp_rigorous
    if name in ("acb_log_prec", "acb_log_batch_prec"):
        return _acb_log_rigorous
    if name in ("acb_sqrt_prec", "acb_sqrt_batch_prec"):
        return _acb_sqrt_rigorous
    if name in ("acb_sin_prec", "acb_sin_batch_prec"):
        return _acb_sin_rigorous
    if name in ("acb_cos_prec", "acb_cos_batch_prec"):
        return _acb_cos_rigorous
    if name in ("acb_tan_prec", "acb_tan_batch_prec"):
        return _acb_tan_rigorous
    if name in ("acb_sinh_prec", "acb_sinh_batch_prec"):
        return _acb_sinh_rigorous
    if name in ("acb_cosh_prec", "acb_cosh_batch_prec"):
        return _acb_cosh_rigorous
    if name in ("acb_tanh_prec", "acb_tanh_batch_prec"):
        return _acb_tanh_rigorous
    return None


def _acb_adaptive_adapter(name: str) -> Callable[..., jax.Array] | None:
    if name in ("acb_exp_prec", "acb_exp_batch_prec"):
        return ball_wrappers.acb_ball_exp_adaptive
    if name in ("acb_log_prec", "acb_log_batch_prec"):
        return ball_wrappers.acb_ball_log_adaptive
    if name in ("acb_sin_prec", "acb_sin_batch_prec"):
        return ball_wrappers.acb_ball_sin_adaptive
    return None


def _acb_generic_rigorous(name: str) -> Callable[..., jax.Array] | None:
    kernel_name = name[:-5] if name.endswith("_prec") else name
    kernel = getattr(acb_core, kernel_name, None)
    if kernel is None:
        return None

    def rig_fn(*args, prec_bits: int, **kwargs):
        return wc.rigorous_acb_kernel(kernel, args, prec_bits, **kwargs)

    return rig_fn


def _acb_generic_adaptive(name: str) -> Callable[..., jax.Array] | None:
    kernel_name = name[:-5] if name.endswith("_prec") else name
    kernel = getattr(acb_core, kernel_name, None)
    if kernel is None:
        return None

    def adapt_fn(*args, prec_bits: int, **kwargs):
        return wc.adaptive_acb_kernel(kernel, args, prec_bits, **kwargs)

    return adapt_fn


def _arb_rigorous_adapter(name: str) -> Callable[..., jax.Array] | None:
    if name.startswith("arb_"):
        return None
    return None


def _arb_adaptive_adapter(name: str) -> Callable[..., jax.Array] | None:
    if name in ("arb_exp_prec", "arb_exp_batch_prec"):
        return ball_wrappers.arb_ball_exp_adaptive
    if name in ("arb_log_prec", "arb_log_batch_prec"):
        return ball_wrappers.arb_ball_log_adaptive
    if name in ("arb_sin_prec", "arb_sin_batch_prec"):
        return ball_wrappers.arb_ball_sin_adaptive
    return None


_SPECIAL_RIG: dict[str, Callable[..., jax.Array]] = {}
_SPECIAL_ADAPT: dict[str, Callable[..., jax.Array]] = {}


def _dispatch(
    impl: str,
    base_fn: Callable[..., jax.Array],
    rig_fn: Callable[..., jax.Array] | None,
    adapt_fn: Callable[..., jax.Array] | None,
    is_acb: bool,
    prec_bits: int,
    args: tuple,
    kwargs: dict,
) -> jax.Array:
    return wc.dispatch_mode(impl, base_fn, rig_fn, adapt_fn, is_acb, prec_bits, args, kwargs)


def _make_wrapper(name: str, base_fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    rig_fn = _SPECIAL_RIG.get(name)
    adapt_fn = _SPECIAL_ADAPT.get(name)
    if rig_fn is None:
        if name.startswith("acb_"):
            rig_fn = _acb_rigorous_adapter(name) or _acb_generic_rigorous(name)
        elif name.startswith("arb_"):
            rig_fn = base_fn
    if adapt_fn is None:
        if name.startswith("acb_"):
            adapt_fn = _acb_adaptive_adapter(name) or _acb_generic_adaptive(name)
        elif name.startswith("arb_"):
            adapt_fn = _arb_adaptive_adapter(name)
    is_acb = name.startswith("acb_")

    def wrapper(*args, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = _resolve_prec_bits(dps, prec_bits)
        return _dispatch(impl, base_fn, rig_fn, adapt_fn, is_acb, pb, args, kwargs)

    wrapper.__name__ = name.replace("_prec", "_mode")
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: baseline|rigorous|adaptive."
    return wrapper


__all__: list[str] = []

for _mod in (arb_core, acb_core):
    for _name in dir(_mod):
        if _name.startswith("_"):
            continue
        if not (_name.endswith("_prec") or _name.endswith("_batch_prec")):
            continue
        _fn = getattr(_mod, _name, None)
        if not callable(_fn):
            continue
        try:
            sig = inspect.signature(_fn)
        except (TypeError, ValueError):
            continue
        if "prec_bits" not in sig.parameters:
            continue
        _wrapper = _make_wrapper(_name, _fn)
        globals()[_wrapper.__name__] = _wrapper
        __all__.append(_wrapper.__name__)
