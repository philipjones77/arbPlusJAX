import jax
import jax.numpy as jnp

from arbplusjax import baseline_wrappers as bw
from arbplusjax import double_interval as di
from arbplusjax import api


from tests._test_checks import _check
def _interval(lo: float, hi: float):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float):
    return jnp.array([re_lo, re_hi, im_lo, im_hi], dtype=jnp.float64)


def test_basic_modes_real():
    x = _interval(0.2, 0.3)
    for mode in ("basic", "rigorous", "adaptive"):
        y = bw.arb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (2,))
        _check(bool(y[0] <= y[1]))


def test_basic_modes_complex():
    x = _box(0.2, 0.3, -0.1, 0.1)
    for mode in ("basic", "rigorous", "adaptive"):
        y = bw.acb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (4,))
        _check(bool(y[0] <= y[1]))
        _check(bool(y[2] <= y[3]))


def test_basic_grad_path():
    x = _interval(0.2, 0.3)

    def loss(t):
        xt = di.interval(t, t)
        y = bw.arb_log_mp(xt, mode="basic", dps=50)
        return jnp.sum(y)

    g = jax.grad(loss)(jnp.float64(0.25))
    _check(bool(jnp.isfinite(g)))


def test_top10_wrappers_and_api_shapes():
    x = _interval(0.2, 0.3)
    y = _interval(1.1, 1.2)
    z = _interval(-0.4, -0.2)

    unary = (
        bw.arb_abs_mp,
        bw.arb_inv_mp,
        bw.arb_log1p_mp,
        bw.arb_expm1_mp,
    )
    for fn in unary:
        for mode in ("basic", "rigorous", "adaptive"):
            out = fn(x if fn is not bw.arb_inv_mp else y, mode=mode, dps=50)
            _check(out.shape == (2,))
            _check(bool(out[0] <= out[1]))

    bivariate = (bw.arb_add_mp, bw.arb_sub_mp, bw.arb_mul_mp, bw.arb_div_mp)
    for fn in bivariate:
        for mode in ("basic", "rigorous", "adaptive"):
            out = fn(x, y, mode=mode, dps=50)
            _check(out.shape == (2,))
            _check(bool(out[0] <= out[1]))

    for mode in ("basic", "rigorous", "adaptive"):
        out = bw.arb_fma_mp(x, y, z, mode=mode, dps=50)
        _check(out.shape == (2,))
        _check(bool(out[0] <= out[1]))
        s, c = bw.arb_sin_cos_mp(x, mode=mode, dps=50)
        _check(s.shape == (2,))
        _check(c.shape == (2,))

    _check(api.eval_interval("add", x, y, mode="basic", dps=50).shape == (2,))
    _check(api.eval_interval("fma", x, y, z, mode="basic", dps=50).shape == (2,))
    s_api, c_api = api.eval_interval("sin_cos", x, mode="basic", dps=50)
    _check(s_api.shape == (2,))
    _check(c_api.shape == (2,))


def test_api_dtype_policy_rejects_mixed_float_dtypes():
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y64 = jnp.array([3.0, 4.0], dtype=jnp.float64)
    try:
        _ = api.eval_point("add", x32, y64)
        _check(False)
    except ValueError as exc:
        _check("Mixed floating dtypes" in str(exc))


def test_api_dtype_policy_allows_explicit_dtype_override():
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y64 = jnp.array([3.0, 4.0], dtype=jnp.float64)
    out32 = api.eval_point("add", x32, y64, dtype="float32")
    out64 = api.eval_point("add", x32, y64, dtype="float64")
    _check(out32.dtype == jnp.float32)
    _check(out64.dtype == jnp.float64)


def test_api_bind_batch_respects_dtype_lock():
    fn64 = api.bind_point_batch("add", dtype="float64")
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y32 = jnp.array([3.0, 4.0], dtype=jnp.float32)
    out = fn64(x32, y32)
    _check(out.dtype == jnp.float64)


def test_api_interval_batch_float32_path():
    x = jnp.array([[0.1, 0.2], [0.2, 0.3]], dtype=jnp.float32)
    y = jnp.array([[0.3, 0.4], [0.4, 0.5]], dtype=jnp.float32)
    out = api.eval_interval_batch("add", x, y, mode="basic", dtype="float32")
    _check(out.dtype == jnp.float32)


def test_api_point_batch_optional_padding_trims_output():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    out = api.eval_point_batch("add", x, y, dtype="float32", pad_to=8)
    _check(out.dtype == jnp.float32)
    _check(out.shape == (3,))
    _check(bool(jnp.allclose(out, jnp.array([5.0, 7.0, 9.0], dtype=jnp.float32))))


def test_api_bind_interval_batch_optional_padding():
    fn = api.bind_interval_batch("add", mode="basic", dtype="float32", pad_to=4)
    x = jnp.array([[0.1, 0.2], [0.2, 0.3]], dtype=jnp.float32)
    y = jnp.array([[0.3, 0.4], [0.4, 0.5]], dtype=jnp.float32)
    out = fn(x, y)
    _check(out.dtype == jnp.float32)
    _check(out.shape == (2, 2))


def test_api_padding_rejects_smaller_target():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    try:
        _ = api.eval_point_batch("add", x, y, dtype="float32", pad_to=2)
        _check(False)
    except ValueError as exc:
        _check("pad_to must be >=" in str(exc))
