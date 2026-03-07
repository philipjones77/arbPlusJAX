import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import boost_hypgeom
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _contains_all(rig: jax.Array, basic: jax.Array) -> bool:
    return bool(jnp.all(di.contains(rig, basic)))


def test_boost_public_hypergeom_four_modes_real():
    a = di.interval(jnp.float64(1.1), jnp.float64(1.2))
    b = di.interval(jnp.float64(2.2), jnp.float64(2.3))
    c = di.interval(jnp.float64(3.1), jnp.float64(3.2))
    z = di.interval(jnp.float64(0.15), jnp.float64(0.2))

    f1_basic = boost_hypgeom.boost_hypergeometric_1f0(a, z, mode="basic", prec_bits=80)
    f1_rig = boost_hypgeom.boost_hypergeometric_1f0(a, z, mode="rigorous", prec_bits=80)
    f1_adp = boost_hypgeom.boost_hypergeometric_1f0(a, z, mode="adaptive", prec_bits=80)
    _check(_contains_all(f1_rig, f1_basic))
    _check(_contains_all(f1_adp, f1_basic))

    f0f1_basic = boost_hypgeom.boost_hypergeometric_0f1(b, z, mode="basic", prec_bits=80)
    f0f1_rig = boost_hypgeom.boost_hypergeometric_0f1(b, z, mode="rigorous", prec_bits=80)
    f0f1_adp = boost_hypgeom.boost_hypergeometric_0f1(b, z, mode="adaptive", prec_bits=80)
    _check(f0f1_basic.shape == (2,))
    _check(f0f1_rig.shape == (2,))
    _check(f0f1_adp.shape[-1] == 2)

    f2f0_basic = boost_hypgeom.boost_hypergeometric_2f0(a, b, z, mode="basic", prec_bits=80)
    f2f0_rig = boost_hypgeom.boost_hypergeometric_2f0(a, b, z, mode="rigorous", prec_bits=80)
    f2f0_adp = boost_hypgeom.boost_hypergeometric_2f0(a, b, z, mode="adaptive", prec_bits=80)
    _check(_contains_all(f2f0_rig, f2f0_basic))
    _check(_contains_all(f2f0_adp, f2f0_basic))

    f1f1_basic = boost_hypgeom.boost_hypergeometric_1f1(a, b, z, mode="basic", prec_bits=80)
    f1f1_rig = boost_hypgeom.boost_hypergeometric_1f1(a, b, z, mode="rigorous", prec_bits=80)
    f1f1_adp = boost_hypgeom.boost_hypergeometric_1f1(a, b, z, mode="adaptive", prec_bits=80)
    _check(f1f1_basic.shape[-1] == 2)
    _check(f1f1_rig.shape[-1] == 2)
    _check(f1f1_adp.shape[-1] == 2)

    pa = jnp.asarray([1.0, 1.5], dtype=jnp.float64)
    pb = jnp.asarray([2.0], dtype=jnp.float64)
    pfq_basic = boost_hypgeom.boost_hypergeometric_pfq(pa, pb, z, mode="basic", prec_bits=80, n_terms=40)
    pfq_rig = boost_hypgeom.boost_hypergeometric_pfq(pa, pb, z, mode="rigorous", prec_bits=80, n_terms=40)
    pfq_adp = boost_hypgeom.boost_hypergeometric_pfq(pa, pb, z, mode="adaptive", prec_bits=80, n_terms=40)
    _check(_contains_all(pfq_rig, pfq_basic))
    _check(_contains_all(pfq_adp, pfq_basic))

    pprec = boost_hypgeom.boost_hypergeometric_pfq_precision(pa, pb, z, prec_bits=80, n_terms=40)
    _check(pprec.shape == (2,))


def test_boost_helpers_four_modes():
    a = di.interval(jnp.float64(1.1), jnp.float64(1.2))
    b = di.interval(jnp.float64(2.2), jnp.float64(2.3))
    c = di.interval(jnp.float64(3.2), jnp.float64(3.3))
    z = di.interval(jnp.float64(0.1), jnp.float64(0.2))
    for mode in ("point", "basic", "rigorous", "adaptive"):
        _check(boost_hypgeom.boost_hyp1f1_series(a, b, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp1f1_asym(a, b, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp2f1_series(a, b, c, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp2f1_cf(a, b, c, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp2f1_pade(a, b, c, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp2f1_rational(a, b, c, z, mode=mode, prec_bits=80) is not None)
        _check(boost_hypgeom.boost_hyp1f2_series(a, b, c, z, mode=mode, prec_bits=80) is not None)


def test_boost_api_registry_and_complex_point():
    out = api.eval_point("boost_hypgeom.boost_hypergeometric_1f0", jnp.float64(1.2), jnp.float64(0.2))
    _check(bool(jnp.isfinite(out)))

    zc = jnp.asarray(0.2 + 0.1j, dtype=jnp.complex128)
    cpt = boost_hypgeom.boost_hypergeometric_1f1(1.5 + 0.0j, 2.5 + 0.0j, zc, mode="point")
    _check(bool(jnp.all(jnp.isfinite(jnp.real(cpt)) & jnp.isfinite(jnp.imag(cpt)))))
