import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import stable_kernels


def test_stable_kernel_provider_barnes_wrappers_match_ifj_family() -> None:
    z = jnp.asarray(1.7 + 0.1j, dtype=jnp.complex128)
    tau = jnp.asarray(0.5, dtype=jnp.float64)

    value = stable_kernels.barnesdoublegamma(z, tau, dps=60)
    log_value = stable_kernels.log_barnesdoublegamma(z, tau, dps=60)

    assert jnp.allclose(value, api.eval_point("ifj_barnesdoublegamma", z, tau, dps=60))
    assert jnp.allclose(log_value, api.eval_point("ifj_log_barnesdoublegamma", z, tau, dps=60))
    assert jnp.allclose(value, jnp.exp(log_value), rtol=1e-8, atol=1e-8)


def test_stable_kernel_provider_barnes_batch_and_prewarm_surfaces() -> None:
    z = jnp.asarray([1.7 + 0.1j, 1.9 + 0.05j], dtype=jnp.complex128)
    tau = jnp.asarray([0.5, 0.5], dtype=jnp.float64)

    batch = stable_kernels.barnesdoublegamma_batch(z, tau, dps=50, pad_to=8)
    log_batch = stable_kernels.log_barnesdoublegamma_batch(z, tau, dps=50, pad_to=8)
    warm = stable_kernels.warm_special_function_point_kernels(dtype="float64", pad_to=8)

    assert batch.shape[0] == 8
    assert log_batch.shape[0] == 8
    assert jnp.allclose(batch[:2], jnp.exp(log_batch[:2]), rtol=1e-8, atol=1e-8)
    for name in (
        "incomplete_gamma_upper",
        "incomplete_bessel_k",
        "arb_hypgeom_1f1",
        "arb_hypgeom_u",
        "ifj_log_barnesdoublegamma",
    ):
        assert name in warm
        assert warm[name] >= 0.0


def test_stable_kernel_provider_barnes_aliases_are_self_consistent() -> None:
    z = jnp.asarray([1.7 + 0.1j, 1.9 + 0.05j], dtype=jnp.complex128)
    tau = jnp.asarray([0.5, 0.6], dtype=jnp.float64)

    value = stable_kernels.provider_barnesdoublegamma(z[0], tau[0], dps=60)
    log_value = stable_kernels.provider_log_barnesdoublegamma(z[0], tau[0], dps=60)
    batch = stable_kernels.provider_barnesdoublegamma_batch(z, tau, dps=60, pad_to=8)
    log_batch = stable_kernels.provider_log_barnesdoublegamma_batch(z, tau, dps=60, pad_to=8)

    assert jnp.allclose(value, jnp.exp(log_value), rtol=1e-8, atol=1e-8)
    assert batch.shape[0] == 8
    assert log_batch.shape[0] == 8
    assert jnp.allclose(batch[:2], jnp.exp(log_batch[:2]), rtol=1e-8, atol=1e-8)


def test_stable_kernel_provider_incomplete_bessel_wrappers_match_public_api() -> None:
    nu = jnp.asarray(0.5, dtype=jnp.float64)
    z = jnp.asarray(1.1, dtype=jnp.float64)
    lower = jnp.asarray(0.4, dtype=jnp.float64)
    upper = jnp.asarray(1.0, dtype=jnp.float64)

    val_k, diag_k = stable_kernels.provider_incomplete_bessel_k(
        nu, z, lower, mode="point", method="auto", return_diagnostics=True
    )
    val_i, diag_i = stable_kernels.provider_incomplete_bessel_i(
        nu, z, upper, mode="point", method="auto", return_diagnostics=True
    )

    ref_k, ref_diag_k = api.incomplete_bessel_k(nu, z, lower, mode="point", method="auto", return_diagnostics=True)
    ref_i, ref_diag_i = api.incomplete_bessel_i(nu, z, upper, mode="point", method="auto", return_diagnostics=True)

    assert jnp.allclose(val_k, ref_k, rtol=1e-10, atol=1e-10)
    assert diag_k.method == ref_diag_k.method
    assert diag_k.fallback_used == ref_diag_k.fallback_used
    assert jnp.allclose(val_i, ref_i, rtol=1e-10, atol=1e-10)
    assert diag_i.method == ref_diag_i.method
    assert diag_i.fallback_used == ref_diag_i.fallback_used


def test_stable_kernel_subset_lists_provider_aliases() -> None:
    kernels = stable_kernels.list_supported_kernels()
    assert "provider_incomplete_bessel_i" in kernels
    assert "provider_incomplete_bessel_k" in kernels
    assert "provider_barnesdoublegamma" in kernels
    assert "provider_log_barnesdoublegamma" in kernels
