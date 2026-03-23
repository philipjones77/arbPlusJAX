from __future__ import annotations

"""Boost-lineage hypergeometric alternatives.

These functions provide provenance-prefixed alternatives to canonical
hypergeometric families.

Provenance:
- classification: alternative
- module lineage: Boost.Math-inspired hypergeometric implementation family
- naming policy: see docs/standards/function_naming_standard.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers
from . import kernel_helpers as kh
from . import point_wrappers
from . import precision
from . import wrappers_common as wc


PROVENANCE = {
    "classification": "alternative",
    "module_lineage": "Boost.Math-inspired hypergeometric implementation family",
    "preferred_prefix": "boost",
    "naming_policy": "docs/standards/function_naming_standard.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}

_MODES = ("point", "basic", "rigorous", "adaptive")


def _prec_bits(dps: int | None, prec_bits: int | None) -> int:
    if prec_bits is not None:
        return int(prec_bits)
    if dps is not None:
        return precision.dps_to_bits(int(dps))
    return precision.get_prec_bits()


def _as_interval(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if xx.ndim >= 1 and xx.shape[-1] == 2:
        return di.as_interval(xx)
    xr = jnp.asarray(xx, dtype=jnp.float64)
    return di.interval(xr, xr)


def _as_acb_box(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z)
    if zz.ndim >= 1 and zz.shape[-1] == 4:
        return hypgeom.as_acb_box(zz)
    zc = jnp.asarray(zz, dtype=jnp.complex128)
    re = di.interval(jnp.real(zc), jnp.real(zc))
    im = di.interval(jnp.imag(zc), jnp.imag(zc))
    return acb_core.acb_box(re, im)


def _is_complex_like(*xs: jax.Array) -> bool:
    for x in xs:
        arr = jnp.asarray(x)
        if jnp.iscomplexobj(arr):
            return True
        if arr.ndim >= 1 and arr.shape[-1] == 4:
            return True
    return False


@jax.jit
def _boost_1f0_point(a: jax.Array, z: jax.Array) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.float64)
    zz = jnp.asarray(z, dtype=jnp.float64)
    return jnp.power(1.0 - zz, -aa)


@jax.jit
def _boost_2f0_point(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.float64)
    bb = jnp.asarray(b, dtype=jnp.float64)
    zz = jnp.asarray(z, dtype=jnp.float64)
    n_terms = 40
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, acc = state
        kf = jnp.float64(k)
        step = ((aa + kf) * (bb + kf) / (kf + 1.0)) * zz
        term = term * step
        return term, acc + term

    _, out = jax.lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    return out


def _broadcast_flatten(*args: jax.Array):
    arrays = jnp.broadcast_arrays(*[jnp.asarray(arg) for arg in args])
    shape = arrays[0].shape
    flats = [jnp.reshape(arr, (-1,)) for arr in arrays]
    return shape, flats


def _vectorize_real_scalar(fn, *args: jax.Array) -> jax.Array:
    shape, flats = _broadcast_flatten(*[jnp.asarray(arg, dtype=jnp.float64) for arg in args])
    out = jax.vmap(fn)(*flats)
    return jnp.reshape(out, shape)


def _vectorize_complex_scalar(fn, *args: jax.Array) -> jax.Array:
    shape, flats = _broadcast_flatten(*[jnp.asarray(arg, dtype=jnp.complex128) for arg in args])
    out = jax.vmap(fn)(*flats)
    return jnp.reshape(out, shape)


def _real_boost_pfq_point_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.float64)
    bb = jnp.asarray(b, dtype=jnp.float64)
    zz = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, acc = state
        kf = jnp.float64(k)
        numer = jnp.prod(aa + kf) if aa.size else jnp.float64(1.0)
        denom = jnp.prod(bb + kf) if bb.size else jnp.float64(1.0)
        step = numer / ((kf + 1.0) * denom)
        term = term * step * zz
        return term, acc + term

    _, out = jax.lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    return jnp.reciprocal(out) if reciprocal else out


def _complex_boost_pfq_point_scalar(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.complex128)
    bb = jnp.asarray(b, dtype=jnp.complex128)
    zz = jnp.asarray(z, dtype=jnp.complex128)
    term0 = jnp.complex128(1.0 + 0.0j)
    sum0 = term0

    def body(k, state):
        term, acc = state
        kf = jnp.float64(k)
        numer = jnp.prod(aa + kf) if aa.size else jnp.complex128(1.0 + 0.0j)
        denom = jnp.prod(bb + kf) if bb.size else jnp.complex128(1.0 + 0.0j)
        step = numer / ((kf + 1.0) * denom)
        term = term * step * zz
        return term, acc + term

    _, out = jax.lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    return jnp.reciprocal(out) if reciprocal else out


def _boost_0f1_point_real(b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(
        lambda bb, zz: _real_boost_pfq_point_scalar(
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.asarray([bb], dtype=jnp.float64),
            zz,
            n_terms=40,
        ),
        b,
        z,
    )
    if regularized:
        out = out * point_wrappers.arb_rgamma_point(b)
    return out


def _boost_0f1_point_complex(b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(
        lambda bb, zz: _complex_boost_pfq_point_scalar(
            jnp.zeros((0,), dtype=jnp.complex128),
            jnp.asarray([bb], dtype=jnp.complex128),
            zz,
            n_terms=40,
        ),
        b,
        z,
    )
    if regularized:
        out = out * point_wrappers.acb_rgamma_point(b)
    return out


def _boost_1f1_point_real(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    out = _vectorize_real_scalar(
        lambda aa, bb, zz: _real_boost_pfq_point_scalar(
            jnp.asarray([aa], dtype=jnp.float64),
            jnp.asarray([bb], dtype=jnp.float64),
            zz,
            n_terms=40,
        ),
        a,
        b,
        z,
    )
    if regularized:
        out = out * point_wrappers.arb_rgamma_point(b)
    return out


def _boost_1f1_point_complex(a: jax.Array, b: jax.Array, z: jax.Array, *, regularized: bool = False) -> jax.Array:
    out = _vectorize_complex_scalar(
        lambda aa, bb, zz: _complex_boost_pfq_point_scalar(
            jnp.asarray([aa], dtype=jnp.complex128),
            jnp.asarray([bb], dtype=jnp.complex128),
            zz,
            n_terms=40,
        ),
        a,
        b,
        z,
    )
    if regularized:
        out = out * point_wrappers.acb_rgamma_point(b)
    return out


def _boost_2f1_point_real(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_real_scalar(
        lambda aa, bb, cc, zz: _real_boost_pfq_point_scalar(
            jnp.asarray([aa, bb], dtype=jnp.float64),
            jnp.asarray([cc], dtype=jnp.float64),
            zz,
            n_terms=40,
        ),
        a,
        b,
        c,
        z,
    )


def _boost_2f1_point_complex(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array) -> jax.Array:
    return _vectorize_complex_scalar(
        lambda aa, bb, cc, zz: _complex_boost_pfq_point_scalar(
            jnp.asarray([aa, bb], dtype=jnp.complex128),
            jnp.asarray([cc], dtype=jnp.complex128),
            zz,
            n_terms=40,
        ),
        a,
        b,
        c,
        z,
    )


def _boost_pfq_point_real(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.float64)
    bb = jnp.asarray(b, dtype=jnp.float64)
    zz = jnp.asarray(z, dtype=jnp.float64)
    return jax.vmap(
        lambda ai, bi, zi: _real_boost_pfq_point_scalar(ai, bi, zi, reciprocal=reciprocal, n_terms=n_terms)
    )(aa, bb, zz)


def _boost_pfq_point_complex(a: jax.Array, b: jax.Array, z: jax.Array, *, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    aa = jnp.asarray(a, dtype=jnp.complex128)
    bb = jnp.asarray(b, dtype=jnp.complex128)
    zz = jnp.asarray(z, dtype=jnp.complex128)
    return jax.vmap(
        lambda ai, bi, zi: _complex_boost_pfq_point_scalar(ai, bi, zi, reciprocal=reciprocal, n_terms=n_terms)
    )(aa, bb, zz)


def _boost_interval_point_to_box(y: jax.Array, prec_bits: int, adaptive: bool = False) -> jax.Array:
    y0 = jnp.asarray(y, dtype=jnp.float64)
    out = di.round_interval_outward(di.interval(y0, y0), prec_bits)
    if not adaptive:
        return out
    eps = jnp.exp2(-jnp.float64(prec_bits)) * (1.0 + jnp.abs(y0))
    return di.interval(out[..., 0] - eps, out[..., 1] + eps)


def _dispatch_arb_mode(mode: str, base_fn, rig_fn, adapt_fn, args: tuple, pb: int):
    if mode == "point":
        out = base_fn(*args)
        arr = jnp.asarray(out)
        if arr.ndim >= 1 and arr.shape[-1] == 2:
            return di.midpoint(out)
        return out
    if mode == "basic":
        out = base_fn(*args)
        return di.round_interval_outward(di.as_interval(out), pb)
    if mode == "rigorous":
        return rig_fn(*args, prec_bits=pb)
    return adapt_fn(*args, prec_bits=pb)


def _dispatch_acb_mode(mode: str, base_fn, mode_fn, args: tuple, pb: int):
    if mode == "point":
        out = base_fn(*args)
        arr = jnp.asarray(out)
        if arr.ndim >= 1 and arr.shape[-1] == 4:
            return acb_core.acb_midpoint(out)
        return out
    return mode_fn(*args, impl=mode, prec_bits=pb)


def boost_hypergeometric_1f0(a: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, z):
        aa = jnp.asarray(a, dtype=jnp.complex128)
        zz = jnp.asarray(z, dtype=jnp.complex128)
        if mode == "point":
            return jnp.power(1.0 - zz, -aa)
        box = _as_acb_box(jnp.power(1.0 - zz, -aa))
        if mode == "basic":
            return box
        return wc.inflate_acb(box, pb, adaptive=(mode == "adaptive"))
    a_iv = _as_interval(a)
    z_iv = _as_interval(z)
    if mode == "point":
        return _boost_1f0_point(di.midpoint(a_iv), di.midpoint(z_iv))
    basic = _boost_interval_point_to_box(_boost_1f0_point(di.midpoint(a_iv), di.midpoint(z_iv)), pb, adaptive=False)
    if mode == "basic":
        return basic
    if mode == "rigorous":
        return wc.rigorous_interval_kernel(
            lambda ai, zi: _boost_interval_point_to_box(_boost_1f0_point(di.midpoint(ai), di.midpoint(zi)), pb, adaptive=False),
            (a_iv, z_iv),
            pb,
        )
    return wc.adaptive_interval_kernel(
        lambda ai, zi: _boost_interval_point_to_box(_boost_1f0_point(di.midpoint(ai), di.midpoint(zi)), pb, adaptive=False),
        (a_iv, z_iv),
        pb,
    )


def boost_hypergeometric_0f1(
    b: jax.Array,
    z: jax.Array,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(b, z):
        if mode == "point":
            return _boost_0f1_point_complex(jnp.asarray(b, dtype=jnp.complex128), jnp.asarray(z, dtype=jnp.complex128), regularized=regularized)
        return _dispatch_acb_mode(
            mode,
            lambda bb, zz: hypgeom.acb_hypgeom_0f1(_as_acb_box(bb), _as_acb_box(zz), regularized=regularized),
            partial(hypgeom_wrappers.acb_hypgeom_0f1_mode, regularized=regularized),
            (_as_acb_box(b), _as_acb_box(z)),
            pb,
        )
    if mode == "point":
        return _boost_0f1_point_real(di.midpoint(_as_interval(b)), di.midpoint(_as_interval(z)), regularized=regularized)
    return _dispatch_arb_mode(
        mode,
        lambda bb, zz: hypgeom.arb_hypgeom_0f1(_as_interval(bb), _as_interval(zz), regularized=regularized),
        partial(hypgeom.arb_hypgeom_0f1_rigorous, regularized=regularized),
        partial(hypgeom_wrappers.arb_hypgeom_0f1_mode, impl="adaptive", regularized=regularized),
        (_as_interval(b), _as_interval(z)),
        pb,
    )


def boost_hypergeometric_2f0(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, b, z):
        aa = jnp.asarray(a, dtype=jnp.complex128)
        bb = jnp.asarray(b, dtype=jnp.complex128)
        zz = jnp.asarray(z, dtype=jnp.complex128)
        if mode == "point":
            return hypgeom._complex_hypergeom_pfq(jnp.asarray([aa, bb], dtype=jnp.complex128), jnp.asarray([], dtype=jnp.complex128), zz)
        box = _as_acb_box(hypgeom._complex_hypergeom_pfq(jnp.asarray([aa, bb], dtype=jnp.complex128), jnp.asarray([], dtype=jnp.complex128), zz))
        if mode == "basic":
            return box
        return wc.inflate_acb(box, pb, adaptive=(mode == "adaptive"))
    a_iv = _as_interval(a)
    b_iv = _as_interval(b)
    z_iv = _as_interval(z)
    if mode == "point":
        return _boost_2f0_point(di.midpoint(a_iv), di.midpoint(b_iv), di.midpoint(z_iv))
    basic = _boost_interval_point_to_box(_boost_2f0_point(di.midpoint(a_iv), di.midpoint(b_iv), di.midpoint(z_iv)), pb, adaptive=False)
    if mode == "basic":
        return basic
    if mode == "rigorous":
        return wc.rigorous_interval_kernel(
            lambda ai, bi, zi: _boost_interval_point_to_box(_boost_2f0_point(di.midpoint(ai), di.midpoint(bi), di.midpoint(zi)), pb, adaptive=False),
            (a_iv, b_iv, z_iv),
            pb,
        )
    return wc.adaptive_interval_kernel(
        lambda ai, bi, zi: _boost_interval_point_to_box(_boost_2f0_point(di.midpoint(ai), di.midpoint(bi), di.midpoint(zi)), pb, adaptive=False),
        (a_iv, b_iv, z_iv),
        pb,
    )


def boost_hypergeometric_1f1(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, b, z):
        if mode == "point":
            return _boost_1f1_point_complex(
                jnp.asarray(a, dtype=jnp.complex128),
                jnp.asarray(b, dtype=jnp.complex128),
                jnp.asarray(z, dtype=jnp.complex128),
                regularized=regularized,
            )
        return _dispatch_acb_mode(
            mode,
            lambda aa, bb, zz: hypgeom.acb_hypgeom_1f1(_as_acb_box(aa), _as_acb_box(bb), _as_acb_box(zz), regularized=regularized),
            partial(hypgeom_wrappers.acb_hypgeom_1f1_mode, regularized=regularized),
            (_as_acb_box(a), _as_acb_box(b), _as_acb_box(z)),
            pb,
        )
    if mode == "point":
        return _boost_1f1_point_real(
            di.midpoint(_as_interval(a)),
            di.midpoint(_as_interval(b)),
            di.midpoint(_as_interval(z)),
            regularized=regularized,
        )
    return _dispatch_arb_mode(
        mode,
        lambda aa, bb, zz: hypgeom.arb_hypgeom_1f1(_as_interval(aa), _as_interval(bb), _as_interval(zz), regularized=regularized),
        partial(hypgeom.arb_hypgeom_1f1_rigorous, regularized=regularized),
        partial(hypgeom_wrappers.arb_hypgeom_1f1_mode, impl="adaptive", regularized=regularized),
        (_as_interval(a), _as_interval(b), _as_interval(z)),
        pb,
    )


def boost_hypergeometric_pfq(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, b, z):
        if mode == "point":
            return _boost_pfq_point_complex(a, b, z, reciprocal=reciprocal, n_terms=n_terms)
        return hypgeom_wrappers.acb_hypgeom_pfq_mode(
            jnp.asarray(a, dtype=jnp.complex128),
            jnp.asarray(b, dtype=jnp.complex128),
            _as_acb_box(z),
            impl=mode,
            prec_bits=pb,
            reciprocal=reciprocal,
            n_terms=n_terms,
        )

    a_f = jnp.asarray(a, dtype=jnp.float64)
    b_f = jnp.asarray(b, dtype=jnp.float64)
    z_iv = _as_interval(z)
    if mode == "point":
        return _boost_pfq_point_real(a_f, b_f, di.midpoint(z_iv), reciprocal=reciprocal, n_terms=n_terms)
    out = hypgeom.arb_hypgeom_pfq(a_f, b_f, z_iv, reciprocal=reciprocal, n_terms=n_terms)
    out = di.round_interval_outward(out, pb)
    if mode == "basic":
        return out
    return wc.inflate_interval(out, pb, adaptive=(mode == "adaptive"))


def boost_hypergeometric_pfq_precision(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    return boost_hypergeometric_pfq(a, b, z, mode="basic", prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)


def boost_hypergeometric_0f1_batch_fixed_point(
    b: jax.Array,
    z: jax.Array,
    *,
    regularized: bool = False,
):
    return boost_hypergeometric_0f1(b, z, mode="point", regularized=regularized)


def boost_hypergeometric_0f1_batch_padded_point(
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    regularized: bool = False,
):
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((b, z), pad_to=pad_to)
    return boost_hypergeometric_0f1(*call_args, mode="point", regularized=regularized)


def boost_hypergeometric_1f1_batch_fixed_point(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    regularized: bool = False,
):
    return boost_hypergeometric_1f1(a, b, z, mode="point", regularized=regularized)


def boost_hypergeometric_1f1_batch_padded_point(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    regularized: bool = False,
):
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b, z), pad_to=pad_to)
    return boost_hypergeometric_1f1(*call_args, mode="point", regularized=regularized)


def boost_hyp2f1_series_batch_fixed_point(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
):
    return boost_hyp2f1_series(a, b, c, z, mode="point")


def boost_hyp2f1_series_batch_padded_point(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
):
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b, c, z), pad_to=pad_to)
    return boost_hyp2f1_series(*call_args, mode="point")


def boost_hypergeometric_pfq_batch_fixed_point(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    return boost_hypergeometric_pfq(a, b, z, mode="point", reciprocal=reciprocal, n_terms=n_terms)


def boost_hypergeometric_pfq_batch_padded_point(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    call_args, _ = kh.pad_mixed_batch_args_repeat_last((a, b, z), pad_to=pad_to)
    return boost_hypergeometric_pfq(*call_args, mode="point", reciprocal=reciprocal, n_terms=n_terms)


def boost_hypergeometric_0f1_batch_padded_prec(
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(b, z):
        return hypgeom.acb_hypgeom_0f1_batch_padded_prec(b, z, pad_to=pad_to, prec_bits=prec_bits, regularized=regularized)
    return hypgeom.arb_hypgeom_0f1_batch_padded_prec(b, z, pad_to=pad_to, prec_bits=prec_bits, regularized=regularized)


def boost_hypergeometric_0f1_batch_fixed_prec(
    b: jax.Array,
    z: jax.Array,
    *,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(b, z):
        return hypgeom.acb_hypgeom_0f1_batch_fixed_prec(b, z, prec_bits=prec_bits, regularized=regularized)
    return hypgeom.arb_hypgeom_0f1_batch_fixed_prec(b, z, prec_bits=prec_bits, regularized=regularized)


def boost_hypergeometric_0f1_batch_mode_padded(
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    impl: str,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(b, z):
        return hypgeom_wrappers.acb_hypgeom_0f1_batch_mode_padded(
            b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, regularized=regularized
        )
    return hypgeom_wrappers.arb_hypgeom_0f1_batch_mode_padded(
        b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, regularized=regularized
    )


def boost_hypergeometric_0f1_batch_mode_fixed(
    b: jax.Array,
    z: jax.Array,
    *,
    impl: str,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(b, z):
        return hypgeom_wrappers.acb_hypgeom_0f1_batch_mode_fixed(
            b, z, impl=impl, prec_bits=prec_bits, regularized=regularized
        )
    return hypgeom_wrappers.arb_hypgeom_0f1_batch_mode_fixed(
        b, z, impl=impl, prec_bits=prec_bits, regularized=regularized
    )


def boost_hypergeometric_1f1_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(a, b, z):
        return hypgeom.acb_hypgeom_1f1_batch_padded_prec(a, b, z, pad_to=pad_to, prec_bits=prec_bits, regularized=regularized)
    return hypgeom.arb_hypgeom_1f1_batch_padded_prec(a, b, z, pad_to=pad_to, prec_bits=prec_bits, regularized=regularized)


def boost_hypergeometric_1f1_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(a, b, z):
        return hypgeom.acb_hypgeom_1f1_batch_fixed_prec(a, b, z, prec_bits=prec_bits, regularized=regularized)
    return hypgeom.arb_hypgeom_1f1_batch_fixed_prec(a, b, z, prec_bits=prec_bits, regularized=regularized)


def boost_hypergeometric_1f1_batch_mode_padded(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    impl: str,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(a, b, z):
        return hypgeom_wrappers.acb_hypgeom_1f1_batch_mode_padded(
            a, b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, regularized=regularized
        )
    return hypgeom_wrappers.arb_hypgeom_1f1_batch_mode_padded(
        a, b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, regularized=regularized
    )


def boost_hypergeometric_1f1_batch_mode_fixed(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    impl: str,
    prec_bits: int = 53,
    regularized: bool = False,
):
    if _is_complex_like(a, b, z):
        return hypgeom_wrappers.acb_hypgeom_1f1_batch_mode_fixed(
            a, b, z, impl=impl, prec_bits=prec_bits, regularized=regularized
        )
    return hypgeom_wrappers.arb_hypgeom_1f1_batch_mode_fixed(
        a, b, z, impl=impl, prec_bits=prec_bits, regularized=regularized
    )


def boost_hyp2f1_series_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = 53,
):
    if _is_complex_like(a, b, c, z):
        return hypgeom.acb_hypgeom_2f1_batch_padded_prec(a, b, c, z, pad_to=pad_to, prec_bits=prec_bits)
    return hypgeom.arb_hypgeom_2f1_batch_padded_prec(a, b, c, z, pad_to=pad_to, prec_bits=prec_bits)


def boost_hyp2f1_series_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
    *,
    prec_bits: int = 53,
):
    if _is_complex_like(a, b, c, z):
        return hypgeom.acb_hypgeom_2f1_batch_fixed_prec(a, b, c, z, prec_bits=prec_bits)
    return hypgeom.arb_hypgeom_2f1_batch_fixed_prec(a, b, c, z, prec_bits=prec_bits)


def boost_hyp2f1_series_batch_mode_padded(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    impl: str,
    prec_bits: int = 53,
):
    if _is_complex_like(a, b, c, z):
        return hypgeom_wrappers.acb_hypgeom_2f1_batch_mode_padded(a, b, c, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits)
    return hypgeom_wrappers.arb_hypgeom_2f1_batch_mode_padded(a, b, c, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits)


def boost_hyp2f1_series_batch_mode_fixed(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    z: jax.Array,
    *,
    impl: str,
    prec_bits: int = 53,
):
    if _is_complex_like(a, b, c, z):
        return hypgeom_wrappers.acb_hypgeom_2f1_batch_mode_fixed(a, b, c, z, impl=impl, prec_bits=prec_bits)
    return hypgeom_wrappers.arb_hypgeom_2f1_batch_mode_fixed(a, b, c, z, impl=impl, prec_bits=prec_bits)


def boost_hypergeometric_pfq_batch_padded_prec(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    prec_bits: int = 53,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    if _is_complex_like(a, b, z):
        return hypgeom.acb_hypgeom_pfq_batch_padded_prec(
            a, b, z, pad_to=pad_to, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
        )
    return hypgeom.arb_hypgeom_pfq_batch_padded_prec(
        a, b, z, pad_to=pad_to, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
    )


def boost_hypergeometric_pfq_batch_fixed_prec(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    prec_bits: int = 53,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    if _is_complex_like(a, b, z):
        return hypgeom.acb_hypgeom_pfq_batch_fixed_prec(a, b, z, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)
    return hypgeom.arb_hypgeom_pfq_batch_fixed_prec(a, b, z, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)


def boost_hypergeometric_pfq_batch_mode_padded(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    pad_to: int,
    impl: str,
    prec_bits: int = 53,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    if _is_complex_like(a, b, z):
        return hypgeom_wrappers.acb_hypgeom_pfq_batch_mode_padded(
            a, b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
        )
    return hypgeom_wrappers.arb_hypgeom_pfq_batch_mode_padded(
        a, b, z, pad_to=pad_to, impl=impl, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
    )


def boost_hypergeometric_pfq_batch_mode_fixed(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    *,
    impl: str,
    prec_bits: int = 53,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    if _is_complex_like(a, b, z):
        return hypgeom_wrappers.acb_hypgeom_pfq_batch_mode_fixed(
            a, b, z, impl=impl, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
        )
    return hypgeom_wrappers.arb_hypgeom_pfq_batch_mode_fixed(
        a, b, z, impl=impl, prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms
    )


def boost_hyp1f1_series(a: jax.Array, b: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hypergeometric_1f1(a, b, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp1f1_asym(a: jax.Array, b: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hypergeometric_1f1(a, b, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp2f1_series(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, b, c, z):
        if mode == "point":
            return _boost_2f1_point_complex(
                jnp.asarray(a, dtype=jnp.complex128),
                jnp.asarray(b, dtype=jnp.complex128),
                jnp.asarray(c, dtype=jnp.complex128),
                jnp.asarray(z, dtype=jnp.complex128),
            )
        return _dispatch_acb_mode(
            mode,
            lambda aa, bb, cc, zz: hypgeom.acb_hypgeom_2f1(_as_acb_box(aa), _as_acb_box(bb), _as_acb_box(cc), _as_acb_box(zz)),
            hypgeom_wrappers.acb_hypgeom_2f1_mode,
            (_as_acb_box(a), _as_acb_box(b), _as_acb_box(c), _as_acb_box(z)),
            pb,
        )
    if mode == "point":
        return _boost_2f1_point_real(
            di.midpoint(_as_interval(a)),
            di.midpoint(_as_interval(b)),
            di.midpoint(_as_interval(c)),
            di.midpoint(_as_interval(z)),
        )
    return _dispatch_arb_mode(
        mode,
        lambda aa, bb, cc, zz: hypgeom.arb_hypgeom_2f1(_as_interval(aa), _as_interval(bb), _as_interval(cc), _as_interval(zz)),
        hypgeom.arb_hypgeom_2f1_rigorous,
        partial(hypgeom_wrappers.arb_hypgeom_2f1_mode, impl="adaptive"),
        (_as_interval(a), _as_interval(b), _as_interval(c), _as_interval(z)),
        pb,
    )


def boost_hyp2f1_cf(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hyp2f1_series(a, b, c, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp2f1_pade(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hyp2f1_series(a, b, c, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp2f1_rational(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hyp2f1_series(a, b, c, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp1f2_series(a: jax.Array, b1: jax.Array, b2: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    aa = jnp.asarray([a], dtype=jnp.float64)
    bb = jnp.asarray([b1, b2], dtype=jnp.float64)
    return boost_hypergeometric_pfq(aa, bb, z, mode=mode, prec_bits=prec_bits, dps=dps)


__all__ = [
    "boost_hypergeometric_1f0",
    "boost_hypergeometric_0f1",
    "boost_hypergeometric_0f1_batch_fixed_prec",
    "boost_hypergeometric_0f1_batch_padded_prec",
    "boost_hypergeometric_0f1_batch_mode_fixed",
    "boost_hypergeometric_0f1_batch_mode_padded",
    "boost_hypergeometric_2f0",
    "boost_hypergeometric_1f1",
    "boost_hypergeometric_1f1_batch_fixed_prec",
    "boost_hypergeometric_1f1_batch_padded_prec",
    "boost_hypergeometric_1f1_batch_mode_fixed",
    "boost_hypergeometric_1f1_batch_mode_padded",
    "boost_hypergeometric_pfq",
    "boost_hypergeometric_pfq_batch_fixed_prec",
    "boost_hypergeometric_pfq_batch_padded_prec",
    "boost_hypergeometric_pfq_batch_mode_fixed",
    "boost_hypergeometric_pfq_batch_mode_padded",
    "boost_hypergeometric_pfq_precision",
    "boost_hyp1f1_series",
    "boost_hyp1f1_asym",
    "boost_hyp2f1_series",
    "boost_hyp2f1_series_batch_fixed_prec",
    "boost_hyp2f1_series_batch_padded_prec",
    "boost_hyp2f1_series_batch_mode_fixed",
    "boost_hyp2f1_series_batch_mode_padded",
    "boost_hyp2f1_cf",
    "boost_hyp2f1_pade",
    "boost_hyp2f1_rational",
    "boost_hyp1f2_series",
]
