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
