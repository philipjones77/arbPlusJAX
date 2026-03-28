import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di


def _real_interval_batch(n: int = 5) -> jnp.ndarray:
    x = jnp.linspace(0.1, 0.5, n, dtype=jnp.float64)
    return di.interval(x, x + 0.01)


def _complex_box_batch(n: int = 5) -> jnp.ndarray:
    x = jnp.linspace(0.1, 0.5, n, dtype=jnp.float64)
    re = di.interval(x, x + 0.01)
    im = di.interval(0.05 * x, 0.05 * x + 0.01)
    return acb_core.acb_box(re, im)


def test_interval_service_binder_matches_eval_interval_batch_for_basic_and_rigorous() -> None:
    x = _real_interval_batch()
    y = _real_interval_batch()
    z = _complex_box_batch()

    real_bound = api.bind_interval_batch("arb_add", mode="basic", dtype="float64", shape_bucket_multiple=8, backend="cpu")
    complex_bound = api.bind_interval_batch("acb_exp", mode="rigorous", dtype="float64", shape_bucket_multiple=8, backend="cpu")

    real_out = real_bound(x, y)
    complex_out = complex_bound(z)

    expected_real = api.eval_interval_batch("arb_add", x, y, mode="basic", dtype="float64", pad_to=8)
    expected_complex = api.eval_interval_batch("acb_exp", z, mode="rigorous", dtype="float64", pad_to=8, prec_bits=53)

    assert real_out.shape == x.shape
    assert complex_out.shape == z.shape
    assert jnp.allclose(real_out, expected_real)
    assert jnp.allclose(complex_out, expected_complex)


def test_interval_service_policy_helper_chooses_backend_and_effective_pad() -> None:
    policy_small = api.choose_interval_batch_policy(
        batch_size=32,
        dtype="float64",
        mode="basic",
        shape_bucket_multiple=64,
        backend="auto",
        min_gpu_batch_size=128,
    )
    policy_large = api.choose_interval_batch_policy(
        batch_size=256,
        dtype="float64",
        mode="rigorous",
        shape_bucket_multiple=64,
        backend="auto",
        min_gpu_batch_size=128,
    )

    assert policy_small.effective_pad_to == 64
    assert policy_large.effective_pad_to == 256
    assert policy_small.mode == "basic"
    assert policy_large.mode == "rigorous"
    if "gpu" in api._available_backends():
        assert policy_small.chosen_backend == "cpu"
        assert policy_large.chosen_backend == "gpu"


def test_interval_service_bound_callable_exposes_diagnostics_and_policy() -> None:
    x = _real_interval_batch()
    bound = api.bind_interval_batch(
        "arb_add",
        mode="basic",
        dtype="float64",
        shape_bucket_multiple=8,
        backend="cpu",
    )

    diag_before = bound.diagnostics(x, x)
    out = bound(x, x)
    diag_after = bound.last_diagnostics
    policy = bound.policy(batch_size=5)

    assert out.shape == x.shape
    assert diag_before.chosen_backend == "cpu"
    assert diag_before.effective_pad_to == 8
    assert diag_after.jit_enabled is False
    assert diag_after.mode == "basic"
    assert policy.effective_pad_to == 8
    assert policy.requested_backend == "cpu"


def test_interval_service_diagnostics_wrappers_report_backend_and_compile_state() -> None:
    api._interval_batch_bound_jit_fn.cache_clear()
    x = _real_interval_batch()
    bound = api.bind_interval_batch_jit_with_diagnostics(
        "arb_add",
        mode="basic",
        dtype="float64",
        shape_bucket_multiple=8,
        backend="cpu",
    )

    _, diag1 = bound(x, x)
    _, diag2 = bound(x, x)

    assert diag1.jit_enabled is True
    assert diag1.chosen_backend == "cpu"
    assert diag1.effective_pad_to == 8
    assert diag1.compiled_this_call is True
    assert diag2.compiled_this_call is False


def test_interval_service_shape_bucket_jit_binder_matches_eval_interval_batch() -> None:
    api._interval_batch_bound_jit_fn.cache_clear()
    bound = api.bind_interval_batch_jit(
        "acb_exp",
        mode="rigorous",
        dtype="float64",
        shape_bucket_multiple=8,
        backend="cpu",
        prec_bits=53,
    )
    z5 = _complex_box_batch(5)
    z7 = _complex_box_batch(7)

    out5 = bound(z5)
    out7 = bound(z7)

    expected5 = api.eval_interval_batch("acb_exp", z5, mode="rigorous", dtype="float64", pad_to=8, prec_bits=53)
    expected7 = api.eval_interval_batch("acb_exp", z7, mode="rigorous", dtype="float64", pad_to=8, prec_bits=53)

    assert out5.shape == z5.shape
    assert out7.shape == z7.shape
    assert jnp.allclose(out5, expected5)
    assert jnp.allclose(out7, expected7)
    info = api._interval_batch_bound_jit_fn.cache_info()
    assert info.misses == 1


def test_interval_service_prewarm_entrypoint_returns_mode_diagnostics() -> None:
    diagnostics = api.prewarm_interval_mode_kernels(
        names=(("arb_add", "basic"), ("acb_exp", "rigorous")),
        dtype="float64",
        prec_bits=53,
        backend="cpu",
        batch_size=16,
        shape_bucket_multiple=8,
        min_gpu_batch_size=128,
    )

    assert set(diagnostics) == {"arb_add:basic", "acb_exp:rigorous"}
    assert diagnostics["arb_add:basic"].chosen_backend == "cpu"
    assert diagnostics["arb_add:basic"].jit_enabled is True
    assert diagnostics["acb_exp:rigorous"].effective_pad_to == 16
