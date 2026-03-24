import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_interval as di


def test_special_service_binders_cover_point_and_interval_paths():
    s = jnp.asarray([1.5, 2.5, 3.5], dtype=jnp.float64)
    z = jnp.asarray([0.75, 1.25, 1.75], dtype=jnp.float64)
    nu = jnp.asarray([0.25, 0.5, 0.75], dtype=jnp.float64)
    lower = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)

    gamma_point = api.bind_point_batch(
        "incomplete_gamma_upper",
        dtype="float64",
        pad_to=8,
        method="quadrature",
        regularized=True,
    )(s, z)
    gamma_basic = api.bind_interval_batch(
        "incomplete_gamma_upper",
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
        method="quadrature",
        regularized=True,
    )(s, z)
    bessel_point = api.bind_point_batch(
        "incomplete_bessel_k",
        dtype="float64",
        pad_to=8,
        method="quadrature",
    )(nu, z, lower)
    bessel_basic = api.bind_interval_batch(
        "incomplete_bessel_k",
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
        method="quadrature",
    )(nu, z, lower)

    assert gamma_point.shape == s.shape
    assert gamma_basic.shape == s.shape + (2,)
    assert bessel_point.shape == nu.shape
    assert bessel_basic.shape == nu.shape + (2,)
    assert jnp.all(jnp.isfinite(gamma_point))
    assert jnp.all(jnp.isfinite(bessel_point))
    assert jnp.all(gamma_basic[..., 0] <= gamma_basic[..., 1])
    assert jnp.all(bessel_basic[..., 0] <= bessel_basic[..., 1])


def test_special_service_chunked_binders_match_nonchunked_api_results():
    s = jnp.asarray([1.25, 1.75, 2.25, 2.75, 3.25], dtype=jnp.float64)
    z = jnp.asarray([0.5, 0.8, 1.1, 1.4, 1.7], dtype=jnp.float64)
    nu = jnp.asarray([0.2, 0.35, 0.5, 0.65, 0.8], dtype=jnp.float64)
    lower = jnp.asarray([0.05, 0.1, 0.15, 0.2, 0.25], dtype=jnp.float64)

    gamma_bound = api.bind_interval_batch(
        "incomplete_gamma_upper",
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
        chunk_size=2,
        method="quadrature",
        regularized=True,
    )
    bessel_bound = api.bind_point_batch(
        "incomplete_bessel_k",
        dtype="float64",
        pad_to=8,
        chunk_size=2,
        method="quadrature",
    )

    assert jnp.allclose(
        gamma_bound(s, z),
        api.eval_interval_batch(
            "incomplete_gamma_upper",
            s,
            z,
            mode="basic",
            dtype="float64",
            pad_to=8,
            prec_bits=53,
            method="quadrature",
            regularized=True,
        ),
    )
    assert jnp.allclose(
        bessel_bound(nu, z, lower),
        api.eval_point_batch(
            "incomplete_bessel_k",
            nu,
            z,
            lower,
            dtype="float64",
            pad_to=8,
            method="quadrature",
        ),
    )


def test_special_service_binders_are_safe_for_repeated_calls():
    s = jnp.linspace(1.25, 3.25, 7, dtype=jnp.float64)
    z = jnp.linspace(0.5, 2.0, 7, dtype=jnp.float64)
    bound = api.bind_point_batch(
        "incomplete_gamma_upper",
        dtype="float64",
        pad_to=16,
        method="quadrature",
        regularized=True,
    )
    expected = api.eval_point_batch(
        "incomplete_gamma_upper",
        s,
        z,
        dtype="float64",
        pad_to=16,
        method="quadrature",
        regularized=True,
    )

    for _ in range(5):
        out = bound(s, z)
        assert out.shape == s.shape
        assert jnp.allclose(out, expected)
