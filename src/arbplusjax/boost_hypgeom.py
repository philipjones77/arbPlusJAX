from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers
from . import precision
from . import wrappers_common as wc

jax.config.update("jax_enable_x64", True)

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


def boost_hypergeometric_1F0(a: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
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


def boost_hypergeometric_0F1(
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
        return _dispatch_acb_mode(
            mode,
            lambda bb, zz: hypgeom.acb_hypgeom_0f1(_as_acb_box(bb), _as_acb_box(zz), regularized=regularized),
            partial(hypgeom_wrappers.acb_hypgeom_0f1_mode, regularized=regularized),
            (_as_acb_box(b), _as_acb_box(z)),
            pb,
        )
    return _dispatch_arb_mode(
        mode,
        lambda bb, zz: hypgeom.arb_hypgeom_0f1(_as_interval(bb), _as_interval(zz), regularized=regularized),
        partial(hypgeom.arb_hypgeom_0f1_rigorous, regularized=regularized),
        partial(hypgeom_wrappers.arb_hypgeom_0f1_mode, impl="adaptive", regularized=regularized),
        (_as_interval(b), _as_interval(z)),
        pb,
    )


def boost_hypergeometric_2F0(
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


def boost_hypergeometric_1F1(
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
        return _dispatch_acb_mode(
            mode,
            lambda aa, bb, zz: hypgeom.acb_hypgeom_1f1(_as_acb_box(aa), _as_acb_box(bb), _as_acb_box(zz), regularized=regularized),
            partial(hypgeom_wrappers.acb_hypgeom_1f1_mode, regularized=regularized),
            (_as_acb_box(a), _as_acb_box(b), _as_acb_box(z)),
            pb,
        )
    return _dispatch_arb_mode(
        mode,
        lambda aa, bb, zz: hypgeom.arb_hypgeom_1f1(_as_interval(aa), _as_interval(bb), _as_interval(zz), regularized=regularized),
        partial(hypgeom.arb_hypgeom_1f1_rigorous, regularized=regularized),
        partial(hypgeom_wrappers.arb_hypgeom_1f1_mode, impl="adaptive", regularized=regularized),
        (_as_interval(a), _as_interval(b), _as_interval(z)),
        pb,
    )


def boost_hypergeometric_pFq(
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
            out = hypgeom.acb_hypgeom_pfq(
                jnp.asarray(a, dtype=jnp.complex128),
                jnp.asarray(b, dtype=jnp.complex128),
                _as_acb_box(z),
                reciprocal=reciprocal,
                n_terms=n_terms,
            )
            return acb_core.acb_midpoint(out)
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
        return di.midpoint(hypgeom.arb_hypgeom_pfq(a_f, b_f, z_iv, reciprocal=reciprocal, n_terms=n_terms))
    out = hypgeom.arb_hypgeom_pfq(a_f, b_f, z_iv, reciprocal=reciprocal, n_terms=n_terms)
    out = di.round_interval_outward(out, pb)
    if mode == "basic":
        return out
    return wc.inflate_interval(out, pb, adaptive=(mode == "adaptive"))


def boost_hypergeometric_pFq_precision(
    a: jax.Array,
    b: jax.Array,
    z: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    reciprocal: bool = False,
    n_terms: int = 32,
):
    return boost_hypergeometric_pFq(a, b, z, mode="basic", prec_bits=prec_bits, reciprocal=reciprocal, n_terms=n_terms)


def boost_hyp1f1_series(a: jax.Array, b: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hypergeometric_1F1(a, b, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp1f1_asym(a: jax.Array, b: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    return boost_hypergeometric_1F1(a, b, z, mode=mode, prec_bits=prec_bits, dps=dps)


def boost_hyp2f1_series(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, mode: str = "point", prec_bits: int | None = None, dps: int | None = None):
    checks.check_in_set(mode, _MODES, "boost_hypgeom.mode")
    pb = _prec_bits(dps, prec_bits)
    if _is_complex_like(a, b, c, z):
        return _dispatch_acb_mode(
            mode,
            lambda aa, bb, cc, zz: hypgeom.acb_hypgeom_2f1(_as_acb_box(aa), _as_acb_box(bb), _as_acb_box(cc), _as_acb_box(zz)),
            hypgeom_wrappers.acb_hypgeom_2f1_mode,
            (_as_acb_box(a), _as_acb_box(b), _as_acb_box(c), _as_acb_box(z)),
            pb,
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
    return boost_hypergeometric_pFq(aa, bb, z, mode=mode, prec_bits=prec_bits, dps=dps)


__all__ = [
    "boost_hypergeometric_1F0",
    "boost_hypergeometric_0F1",
    "boost_hypergeometric_2F0",
    "boost_hypergeometric_1F1",
    "boost_hypergeometric_pFq",
    "boost_hypergeometric_pFq_precision",
    "boost_hyp1f1_series",
    "boost_hyp1f1_asym",
    "boost_hyp2f1_series",
    "boost_hyp2f1_cf",
    "boost_hyp2f1_pade",
    "boost_hyp2f1_rational",
    "boost_hyp1f2_series",
]
