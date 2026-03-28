from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import stable_kernels as sk


def test_stable_kernel_registry_lists_curated_surface():
    names = set(sk.list_supported_kernels())
    assert {
        "gamma",
        "loggamma",
        "incomplete_gamma_lower",
        "incomplete_gamma_upper",
        "incomplete_bessel_i",
        "incomplete_bessel_k",
    } <= names


def test_stable_gamma_and_loggamma_match_api():
    x = jnp.asarray(1.75, dtype=jnp.float64)
    assert jnp.allclose(sk.gamma(x), api.eval_point("gamma", x))
    assert jnp.allclose(sk.loggamma(x), api.eval_point("arb_lgamma", x))


def test_stable_gamma_basic_mode_preserves_interval_contract():
    x = jnp.asarray(1.75, dtype=jnp.float64)
    basic = sk.gamma(x, mode="basic")
    point = sk.gamma(x)
    assert basic.shape == (2,)
    assert bool((basic[0] <= point) & (point <= basic[1]))


def test_stable_incomplete_kernel_wrappers_match_api():
    s = jnp.asarray(2.5, dtype=jnp.float64)
    z = jnp.asarray(1.75, dtype=jnp.float64)
    nu = jnp.asarray(0.5, dtype=jnp.float64)
    upper = jnp.asarray(1.1, dtype=jnp.float64)
    lower = jnp.asarray(0.2, dtype=jnp.float64)

    assert jnp.allclose(
        sk.incomplete_gamma_upper(s, z, regularized=True),
        api.incomplete_gamma_upper(s, z, regularized=True),
    )
    assert jnp.allclose(
        sk.incomplete_gamma_lower(s, z, regularized=True),
        api.incomplete_gamma_lower(s, z, regularized=True),
    )
    assert jnp.allclose(sk.incomplete_bessel_i(nu, z, upper), api.incomplete_bessel_i(nu, z, upper))
    assert jnp.allclose(sk.incomplete_bessel_k(nu, z, lower), api.incomplete_bessel_k(nu, z, lower))


def test_stable_special_batch_wrappers_support_pad_to() -> None:
    s = jnp.asarray([1.5, 2.0, 2.5], dtype=jnp.float64)
    z = jnp.asarray([0.2, 0.4, 0.6], dtype=jnp.float64)
    nu = jnp.asarray([0.5, 0.75, 1.0], dtype=jnp.float64)
    lower = jnp.asarray([0.2, 0.3, 0.4], dtype=jnp.float64)

    gamma_batch = sk.incomplete_gamma_upper_batch(s, z, mode="point", regularized=True, method="quadrature", pad_to=8)
    bessel_batch = sk.incomplete_bessel_k_batch(nu, z, lower, mode="point", method="quadrature", pad_to=8)

    assert gamma_batch.shape[0] == 3
    assert bessel_batch.shape[0] == 3
    assert jnp.allclose(gamma_batch[:3], api.incomplete_gamma_upper_batch(s, z, mode="point", regularized=True, method="quadrature"))
    assert jnp.allclose(bessel_batch[:3], api.incomplete_bessel_k_batch(nu, z, lower, mode="point", method="quadrature"))
