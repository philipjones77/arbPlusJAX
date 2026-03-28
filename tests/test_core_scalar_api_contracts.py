import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import core_wrappers


def test_core_scalar_api_eval_point_routes_representative_scalar_families():
    real = jnp.array([1.0, 2.0], dtype=jnp.float32)
    imag = jnp.array([1.0 + 2.0j, 2.0 - 1.0j], dtype=jnp.complex64)
    ints_a = jnp.array([[1, 2], [3, 5]], dtype=jnp.int64)
    ints_b = jnp.array([[4, 6], [7, 9]], dtype=jnp.int64)

    arf = api.eval_point("arf_add", real, real)
    acf = api.eval_point("acf_mul", imag, imag)
    fmpr = api.eval_point("fmpr_mul", real, real)
    fmpzi = api.eval_point("fmpzi_add", ints_a, ints_b)
    fpwrap = api.eval_point("arb_fpwrap_double_exp", real)

    assert arf.dtype == jnp.float32
    assert acf.dtype == jnp.complex64
    assert fmpr.dtype == jnp.float32
    assert fmpzi.dtype == jnp.int64
    assert fpwrap.dtype == jnp.float32
    assert arf.shape == real.shape
    assert acf.shape == imag.shape
    assert fmpr.shape == real.shape
    assert fmpzi.shape == ints_a.shape
    assert fpwrap.shape == real.shape


def test_core_scalar_api_batch_aliases_match_point_results():
    real = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)
    imag = jnp.array([0.25 + 0.1j, 0.5 - 0.15j, 0.75 + 0.2j], dtype=jnp.complex64)
    ints_a = jnp.array([[1, 2], [3, 4], [5, 7]], dtype=jnp.int64)
    ints_b = jnp.array([[2, 3], [4, 6], [7, 10]], dtype=jnp.int64)

    assert jnp.allclose(api.eval_point_batch("arf_add", real, real), api.eval_point("arf_add", real, real))
    assert jnp.allclose(api.eval_point_batch("acf_add", imag, imag), api.eval_point("acf_add", imag, imag))
    assert jnp.allclose(api.eval_point_batch("fmpr_add", real, real), api.eval_point("fmpr_add", real, real))
    assert jnp.array_equal(api.eval_point_batch("fmpzi_sub", ints_b, ints_a), api.eval_point("fmpzi_sub", ints_b, ints_a))
    assert jnp.allclose(
        api.eval_point_batch("arb_fpwrap_cdouble_log", imag),
        api.eval_point("arb_fpwrap_cdouble_log", imag),
        equal_nan=True,
    )


def test_core_scalar_api_metadata_exposes_public_contract_fields():
    names = set(api.list_public_functions())
    for name in ("arf_add", "acf_mul", "fmpr_add", "fmpzi_add", "arb_fpwrap_double_exp"):
        assert name in names
        meta = api.get_public_function_metadata(name)
        assert meta.name == name
        assert meta.family == "core"
        assert isinstance(meta.implementation_options, tuple)
        assert meta.default_implementation in meta.implementation_options


def test_core_scalar_mode_wrappers_include_lazy_jitted_complex_precision_exports():
    # Some `acb_core.*_prec` functions are exposed through a lazy-JIT decorator.
    # The mode-surface generator must still preserve them as public wrappers.
    for name in (
        "acb_abs_mode",
        "acb_add_mode",
        "acb_mul_mode",
        "acb_div_mode",
        "acb_log1p_mode",
        "acb_expm1_mode",
    ):
        assert hasattr(core_wrappers, name), name
