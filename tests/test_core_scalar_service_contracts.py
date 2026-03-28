import jax.numpy as jnp

from arbplusjax import api


def test_core_scalar_service_binders_respect_dtype_padding_and_family_shapes():
    real32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    real64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    complex32 = jnp.array([1.0 + 0.5j, 2.0 - 0.25j, 3.0 + 0.75j], dtype=jnp.complex64)
    ints_a = jnp.array([[1, 2], [3, 5], [8, 13]], dtype=jnp.int64)
    ints_b = jnp.array([[2, 3], [5, 8], [13, 21]], dtype=jnp.int64)

    arf32 = api.bind_point_batch("arf_add", dtype="float32", pad_to=8)(real32, real32)
    arf64 = api.bind_point_batch("arf_add", dtype="float64", pad_to=8)(real32, real32)
    acf32 = api.bind_point_batch("acf_mul", dtype="float32", pad_to=8)(complex32, complex32)
    acf64 = api.bind_point_batch("acf_mul", dtype="float64", pad_to=8)(complex32, complex32)
    fmpr64 = api.bind_point_batch("fmpr_mul", dtype="float64", pad_to=8)(real64, real64)
    fmpzi = api.bind_point_batch("fmpzi_add", pad_to=8)(ints_a, ints_b)
    fpwrap64 = api.bind_point_batch("arb_fpwrap_double_exp", dtype="float64", pad_to=8)(real32)

    assert arf32.dtype == jnp.float32
    assert arf64.dtype == jnp.float64
    assert acf32.dtype == jnp.complex64
    assert acf64.dtype == jnp.complex128
    assert fmpr64.dtype == jnp.float64
    assert fmpzi.dtype == jnp.int64
    assert fpwrap64.dtype == jnp.float64
    assert arf32.shape == real32.shape
    assert acf32.shape == complex32.shape
    assert fmpzi.shape == ints_a.shape


def test_core_scalar_service_chunked_binders_match_nonchunked_api_results():
    real = jnp.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=jnp.float32)
    imag = jnp.array([0.25 + 0.1j, 0.5 - 0.15j, 0.75 + 0.2j, 1.0 + 0.3j, 1.25 - 0.1j], dtype=jnp.complex64)
    ints_a = jnp.array([[1, 2], [3, 5], [8, 13], [21, 34], [55, 89]], dtype=jnp.int64)
    ints_b = jnp.array([[2, 3], [5, 8], [13, 21], [34, 55], [89, 144]], dtype=jnp.int64)

    arf_bound = api.bind_point_batch("arf_add", dtype="float32", pad_to=8, chunk_size=2)
    acf_bound = api.bind_point_batch("acf_add", dtype="float32", pad_to=8, chunk_size=2)
    fmpr_bound = api.bind_point_batch("fmpr_add", dtype="float32", pad_to=8, chunk_size=2)
    fmpzi_bound = api.bind_point_batch("fmpzi_sub", pad_to=8, chunk_size=2)
    fpwrap_bound = api.bind_point_batch("arb_fpwrap_cdouble_log", dtype="float32", pad_to=8, chunk_size=2)

    assert jnp.allclose(arf_bound(real, real), api.eval_point_batch("arf_add", real, real, dtype="float32", pad_to=8))
    assert jnp.allclose(acf_bound(imag, imag), api.eval_point_batch("acf_add", imag, imag, dtype="float32", pad_to=8))
    assert jnp.allclose(fmpr_bound(real, real), api.eval_point_batch("fmpr_add", real, real, dtype="float32", pad_to=8))
    assert jnp.array_equal(fmpzi_bound(ints_b, ints_a), api.eval_point_batch("fmpzi_sub", ints_b, ints_a, pad_to=8))
    assert jnp.allclose(
        fpwrap_bound(imag),
        api.eval_point_batch("arb_fpwrap_cdouble_log", imag, dtype="float32", pad_to=8),
        equal_nan=True,
    )


def test_core_scalar_service_binders_are_safe_for_repeated_calls():
    real = jnp.linspace(0.1, 1.0, 7, dtype=jnp.float32)
    bound = api.bind_point_batch("arb_fpwrap_double_exp", dtype="float32", pad_to=16)
    expected = api.eval_point_batch("arb_fpwrap_double_exp", real, dtype="float32", pad_to=16)

    for _ in range(5):
        out = bound(real)
        assert out.shape == real.shape
        assert jnp.allclose(out, expected)


def test_core_scalar_service_shape_bucket_binder_matches_eval_point_batch_for_varying_lengths():
    bound = api.bind_point_batch("arf_add", dtype="float32", shape_bucket_multiple=8)

    x5 = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)
    x7 = jnp.linspace(0.1, 0.7, 7, dtype=jnp.float32)

    out5 = bound(x5, x5)
    out7 = bound(x7, x7)

    expected5 = api.eval_point_batch("arf_add", x5, x5, dtype="float32", pad_to=8)
    expected7 = api.eval_point_batch("arf_add", x7, x7, dtype="float32", pad_to=8)

    assert out5.shape == x5.shape
    assert out7.shape == x7.shape
    assert jnp.allclose(out5, expected5)
    assert jnp.allclose(out7, expected7)


def test_core_scalar_service_shape_bucket_jit_binder_matches_eval_point_batch_for_varying_lengths():
    api._point_batch_bound_jit_fn.cache_clear()
    bound = api.bind_point_batch_jit("arb_fpwrap_double_exp", dtype="float32", shape_bucket_multiple=8)

    x5 = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)
    x7 = jnp.linspace(0.1, 0.7, 7, dtype=jnp.float32)

    out5 = bound(x5)
    out7 = bound(x7)

    expected5 = api.eval_point_batch("arb_fpwrap_double_exp", x5, dtype="float32", pad_to=8)
    expected7 = api.eval_point_batch("arb_fpwrap_double_exp", x7, dtype="float32", pad_to=8)

    assert out5.shape == x5.shape
    assert out7.shape == x7.shape
    assert jnp.allclose(out5, expected5)
    assert jnp.allclose(out7, expected7)
    info = api._point_batch_bound_jit_fn.cache_info()
    assert info.misses == 1


def test_core_scalar_service_policy_helper_chooses_backend_and_effective_pad():
    policy_small = api.choose_point_batch_policy(
        batch_size=64,
        dtype="float32",
        backend="auto",
        shape_bucket_multiple=128,
        min_gpu_batch_size=256,
    )
    policy_large = api.choose_point_batch_policy(
        batch_size=512,
        dtype="float32",
        backend="auto",
        shape_bucket_multiple=128,
        min_gpu_batch_size=256,
    )

    assert policy_small.effective_pad_to == 128
    assert policy_large.effective_pad_to == 512
    assert policy_small.chosen_backend in {"cpu", "gpu"}
    assert policy_large.chosen_backend in {"cpu", "gpu"}
    if "gpu" in api._available_backends():
        assert policy_small.chosen_backend == "cpu"
        assert policy_large.chosen_backend == "gpu"


def test_core_scalar_service_diagnostics_wrappers_report_backend_and_compile_state():
    api._point_batch_bound_jit_fn.cache_clear()
    bound = api.bind_point_batch_jit_with_diagnostics(
        "arb_fpwrap_double_exp",
        dtype="float32",
        shape_bucket_multiple=8,
        backend="cpu",
    )
    x = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)

    _, diag1 = bound(x)
    _, diag2 = bound(x)

    assert diag1.jit_enabled is True
    assert diag1.chosen_backend == "cpu"
    assert diag1.effective_pad_to == 8
    assert diag1.compiled_this_call is True
    assert diag2.compiled_this_call is False


def test_core_scalar_service_bound_callable_exposes_diagnostics_and_policy():
    bound = api.bind_point_batch(
        "arf_add",
        dtype="float32",
        shape_bucket_multiple=8,
        backend="cpu",
    )
    x = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)

    diag_before = bound.diagnostics(x, x)
    out = bound(x, x)
    diag_after = bound.last_diagnostics
    policy = bound.policy(batch_size=5)

    assert out.shape == x.shape
    assert diag_before.chosen_backend == "cpu"
    assert diag_before.effective_pad_to == 8
    assert diag_after.jit_enabled is False
    assert policy.effective_pad_to == 8
    assert policy.requested_backend == "cpu"


def test_core_scalar_service_prewarm_entrypoint_returns_scalar_family_diagnostics():
    diagnostics = api.prewarm_core_point_kernels(
        names=("arf_add", "acf_mul"),
        dtype="float32",
        backend="cpu",
        batch_size=16,
        shape_bucket_multiple=8,
        min_gpu_batch_size=128,
    )

    assert set(diagnostics) == {"arf_add", "acf_mul"}
    assert diagnostics["arf_add"].chosen_backend == "cpu"
    assert diagnostics["arf_add"].jit_enabled is True
    assert diagnostics["acf_mul"].effective_pad_to == 16
