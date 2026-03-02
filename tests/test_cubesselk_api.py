import jax.numpy as jnp
import numpy as np

from arbplusjax import api
from arbplusjax import cubesselk
from arbplusjax import double_interval as di
from arbplusjax import hypgeom

from tests._test_checks import _check


def test_cubesselk_point_fallback_matches_hypgeom():
    nu = jnp.asarray([0.5, 1.2, 2.1], dtype=jnp.float64)
    z = jnp.asarray([0.3, 1.7, 4.4], dtype=jnp.float64)
    out = cubesselk.cubesselk_point(nu, z)
    ref = hypgeom._real_bessel_eval_k(nu, z)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_cubesselk_modes_shapes_and_containment():
    nu = di.interval(jnp.float64(0.9), jnp.float64(1.1))
    z = di.interval(jnp.float64(0.9), jnp.float64(1.1))
    basic = cubesselk.arb_cubesselk_mp(nu, z, mode="basic", prec_bits=80)
    rig = cubesselk.arb_cubesselk_mp(nu, z, mode="rigorous", prec_bits=80)
    ad = cubesselk.arb_cubesselk_mp(nu, z, mode="adaptive", prec_bits=80)
    _check(basic.shape == (2,))
    _check(rig.shape == (2,))
    _check(ad.shape == (2,))
    _check(bool(di.contains(rig, basic)))
    _check(bool(di.contains(ad, basic)))


def test_api_dispatch_for_cubesselk_point_and_interval():
    nu_pt = jnp.asarray([0.5, 0.75], dtype=jnp.float64)
    z_pt = jnp.asarray([1.0, 1.5], dtype=jnp.float64)
    y_pt = api.eval_point("CubesselK", nu_pt, z_pt)
    _check(y_pt.shape == (2,))

    nu_iv = jnp.stack([nu_pt - 1e-6, nu_pt + 1e-6], axis=-1)
    z_iv = jnp.stack([z_pt - 1e-6, z_pt + 1e-6], axis=-1)
    y_iv = api.eval_interval("CubesselK", nu_iv, z_iv, mode="basic", prec_bits=80)
    _check(y_iv.shape == (2, 2))

    y_b = api.eval_point_batch("CubesselK", nu_pt, z_pt)
    _check(y_b.shape == (2,))
