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


def test_special_service_diagnostics_binders_cover_gamma_hypgeom_and_barnes() -> None:
    s = jnp.asarray([1.25, 1.75, 2.25], dtype=jnp.float64)
    z = jnp.asarray([0.5, 0.8, 1.1], dtype=jnp.float64)
    a = jnp.asarray([1.0, 1.25, 1.5], dtype=jnp.float64)
    b = jnp.asarray([2.0, 2.25, 2.5], dtype=jnp.float64)
    z_hyp = jnp.asarray([0.2, 0.3, 0.4], dtype=jnp.float64)
    z_barnes = jnp.asarray([1.1 + 0.05j, 1.35 + 0.08j, 1.6 + 0.1j], dtype=jnp.complex128)
    tau = jnp.asarray([0.8, 0.9, 1.0], dtype=jnp.float64)

    gamma_bound = api.bind_point_batch_jit_with_diagnostics(
        "incomplete_gamma_upper",
        dtype="float64",
        shape_bucket_multiple=8,
        method="quadrature",
        regularized=True,
        backend="auto",
    )
    gamma_value, gamma_diag = gamma_bound(s, z)

    hypgeom_bound = api.bind_point_batch_jit_with_diagnostics(
        "hypgeom.arb_hypgeom_1f1",
        dtype="float64",
        shape_bucket_multiple=8,
        backend="auto",
    )
    hypgeom_value, hypgeom_diag = hypgeom_bound(a, b, z_hyp)

    barnes_bound = api.bind_point_batch_jit_with_diagnostics(
        "ifj_barnesdoublegamma",
        shape_bucket_multiple=8,
        dps=50,
        max_m_cap=128,
        backend="auto",
    )
    barnes_value, barnes_diag = barnes_bound(z_barnes, tau)

    assert gamma_value.shape == s.shape
    assert hypgeom_value.shape == a.shape
    assert barnes_value.shape == z_barnes.shape
    assert gamma_diag.chosen_backend in {"cpu", "gpu"}
    assert hypgeom_diag.chosen_backend in {"cpu", "gpu"}
    assert barnes_diag.chosen_backend in {"cpu", "gpu"}
    assert gamma_diag.effective_pad_to == 8
    assert hypgeom_diag.effective_pad_to == 8
    assert barnes_diag.effective_pad_to == 8
    assert gamma_diag.jit_enabled is True
    assert hypgeom_diag.jit_enabled is True
    assert barnes_diag.jit_enabled is True


def test_special_scalar_diagnostics_cover_gamma_bessel_and_barnes_provider_fields() -> None:
    gamma_value, gamma_diag = api.incomplete_gamma_upper(
        jnp.float64(2.5),
        jnp.float64(0.75),
        mode="point",
        method="auto",
        regularized=True,
        return_diagnostics=True,
    )
    bessel_value, bessel_diag = api.incomplete_bessel_k(
        jnp.float64(0.6),
        jnp.float64(1.8),
        jnp.float64(0.4),
        mode="point",
        method="auto",
        return_diagnostics=True,
    )
    barnes_diag = api.eval_point(
        "ifj_barnesdoublegamma_diagnostics",
        jnp.asarray(0.3 + 0.05j, dtype=jnp.complex128),
        jnp.float64(1.0),
        dps=60,
        max_m_cap=96,
    )

    assert jnp.isfinite(gamma_value)
    assert jnp.isfinite(bessel_value)
    assert gamma_diag.method in {"quadrature", "recurrence", "aitken", "wynn", "auto", "high_precision_refine"}
    assert isinstance(gamma_diag.fallback_used, bool)
    assert bessel_diag.method in {"quadrature", "high_precision_refine", "auto"}
    assert isinstance(bessel_diag.fallback_used, bool)
    assert barnes_diag.m_used <= barnes_diag.max_m_cap
    assert barnes_diag.n_shift >= 1
