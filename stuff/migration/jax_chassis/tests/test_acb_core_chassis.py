import jax
import jax.numpy as jnp

from arbjax import acb_core
from arbjax import double_interval as di


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    x = jnp.array(
        [
            [0.3, 0.4, -0.2, 0.1],
            [1.2, 1.5, 0.05, 0.2],
            [-0.7, 0.1, -0.3, -0.1],
        ],
        dtype=jnp.float64,
    )
    for fn in (
        acb_core.acb_exp_batch_jit,
        acb_core.acb_log_batch_jit,
        acb_core.acb_sqrt_batch_jit,
        acb_core.acb_sin_batch_jit,
        acb_core.acb_cos_batch_jit,
        acb_core.acb_tan_batch_jit,
        acb_core.acb_sinh_batch_jit,
        acb_core.acb_cosh_batch_jit,
        acb_core.acb_tanh_batch_jit,
    ):
        out = fn(x)
        assert out.shape == (3, 4)
        assert bool(jnp.all(out[:, 0] <= out[:, 1]))
        assert bool(jnp.all(out[:, 2] <= out[:, 3]))

    y = acb_core.acb_add(x, x)
    z = acb_core.acb_mul(x, x)
    denom = acb_core.acb_box(
        di.interval(jnp.array([0.8, 0.9, 1.1]), jnp.array([0.9, 1.0, 1.2])),
        di.interval(jnp.array([0.2, 0.1, 0.05]), jnp.array([0.25, 0.2, 0.1])),
    )
    w = acb_core.acb_div(x, denom)
    assert y.shape == (3, 4)
    assert z.shape == (3, 4)
    assert w.shape == (3, 4)


def test_grad_paths_on_point_intervals():
    def grad_real_part(fn, re0: float, im0: float):
        def loss(t):
            tt = jnp.asarray(t, dtype=jnp.float64)
            y = fn(acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(im0), jnp.float64(im0))))
            return jnp.real(acb_core.acb_midpoint(y))

        g = jax.grad(loss)(jnp.float64(re0))
        assert bool(jnp.isfinite(g))

    grad_real_part(acb_core.acb_exp, 0.4, 0.2)
    grad_real_part(acb_core.acb_log, 1.3, 0.1)
    grad_real_part(acb_core.acb_sqrt, 1.8, 0.05)
    grad_real_part(acb_core.acb_sin, 0.6, -0.3)
    grad_real_part(acb_core.acb_cos, 0.6, 0.2)
    grad_real_part(acb_core.acb_tan, 0.2, 0.1)
    grad_real_part(acb_core.acb_sinh, 0.6, -0.2)
    grad_real_part(acb_core.acb_cosh, 0.6, 0.1)
    grad_real_part(acb_core.acb_tanh, 0.6, 0.2)


def test_basic_ops_and_abs_arg():
    x = _box(0.2, 0.25, -0.3, -0.25)
    y = _box(0.1, 0.12, 0.05, 0.08)
    add = acb_core.acb_add(x, y)
    sub = acb_core.acb_sub(x, y)
    mul = acb_core.acb_mul(x, y)
    div = acb_core.acb_div(x, y)
    abs_int = acb_core.acb_abs(x)
    arg_int = acb_core.acb_arg(x)

    assert add.shape == (4,)
    assert sub.shape == (4,)
    assert mul.shape == (4,)
    assert div.shape == (4,)
    assert abs_int.shape == (2,)
    assert arg_int.shape == (2,)


def test_new_transcendentals_and_pow():
    x = _box(0.2, 0.25, -0.1, -0.05)
    sin_box, cos_box = acb_core.acb_sin_cos(x)
    sin_pi_box, cos_pi_box = acb_core.acb_sin_cos_pi(x)
    exp_inv, inv_exp = acb_core.acb_exp_invexp(x)
    pow_box = acb_core.acb_pow(x, _box(1.0, 1.0, 0.0, 0.0))
    root_box = acb_core.acb_root_ui(x, 2)
    sinc_box = acb_core.acb_sinc(x)
    sinc_pi_box = acb_core.acb_sinc_pi(x)
    log1p_box = acb_core.acb_log1p(x)
    rsqrt_box = acb_core.acb_rsqrt(x)
    cot_box = acb_core.acb_cot(x)

    assert sin_box.shape == (4,)
    assert cos_box.shape == (4,)
    assert sin_pi_box.shape == (4,)
    assert cos_pi_box.shape == (4,)
    assert exp_inv.shape == (4,)
    assert inv_exp.shape == (4,)
    assert pow_box.shape == (4,)
    assert root_box.shape == (4,)
    assert sinc_box.shape == (4,)
    assert sinc_pi_box.shape == (4,)
    assert log1p_box.shape == (4,)
    assert rsqrt_box.shape == (4,)
    assert cot_box.shape == (4,)


def test_real_ops_and_dot():
    x = _box(-0.4, -0.2, 0.1, 0.2)
    y = _box(0.1, 0.2, -0.1, -0.05)
    real_abs = acb_core.acb_real_abs(x)
    real_sgn = acb_core.acb_real_sgn(x)
    real_floor = acb_core.acb_real_floor(x)
    real_ceil = acb_core.acb_real_ceil(x)
    real_max = acb_core.acb_real_max(x, y)
    real_min = acb_core.acb_real_min(x, y)
    real_sqrtpos = acb_core.acb_real_sqrtpos(x)
    union_box = acb_core.acb_union(x, y)
    trimmed = acb_core.acb_trim(x)

    assert real_abs.shape == (4,)
    assert real_sgn.shape == (4,)
    assert real_floor.shape == (4,)
    assert real_ceil.shape == (4,)
    assert real_max.shape == (4,)
    assert real_min.shape == (4,)
    assert real_sqrtpos.shape == (4,)
    assert union_box.shape == (4,)
    assert trimmed.shape == (4,)

    xs = jnp.stack([x, y], axis=0)
    ys = jnp.stack([y, x], axis=0)
    init = acb_core.acb_zero()
    dot = acb_core.acb_dot_simple(init, 0, xs, ys)
    assert dot.shape == (4,)


def test_precision_semantics_wider_at_lower_precision():
    x = _box(0.123456789, 0.123456799, -0.3, -0.25)
    hi_exp = acb_core.acb_exp_prec(x, prec_bits=53)
    lo_exp = acb_core.acb_exp_prec(x, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo_exp), acb_core.acb_real(hi_exp)))
    assert bool(di.contains(acb_core.acb_imag(lo_exp), acb_core.acb_imag(hi_exp)))

    y = _box(0.6, 0.65, 0.1, 0.2)
    hi_sin = acb_core.acb_sin_prec(y, prec_bits=53)
    lo_sin = acb_core.acb_sin_prec(y, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo_sin), acb_core.acb_real(hi_sin)))
    assert bool(di.contains(acb_core.acb_imag(lo_sin), acb_core.acb_imag(hi_sin)))

    z = _box(0.2, 0.25, -0.15, -0.1)
    hi_tan = acb_core.acb_tan_prec(z, prec_bits=53)
    lo_tan = acb_core.acb_tan_prec(z, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo_tan), acb_core.acb_real(hi_tan)))
    assert bool(di.contains(acb_core.acb_imag(lo_tan), acb_core.acb_imag(hi_tan)))


def test_special_functions_shapes():
    x = _box(0.2, 0.25, 0.05, 0.1)
    s = _box(2.0, 2.0, 0.0, 0.0)
    a = _box(1.2, 1.2, 0.0, 0.0)
    z = _box(0.3, 0.3, 0.0, 0.0)

    assert acb_core.acb_gamma(x).shape == (4,)
    assert acb_core.acb_rgamma(x).shape == (4,)
    assert acb_core.acb_lgamma(x).shape == (4,)
    assert acb_core.acb_log_sin_pi(x).shape == (4,)
    assert acb_core.acb_digamma(x).shape == (4,)
    assert acb_core.acb_zeta(x).shape == (4,)
    assert acb_core.acb_hurwitz_zeta(s, a).shape == (4,)
    assert acb_core.acb_polygamma(1, x).shape == (4,)
    assert acb_core.acb_bernoulli_poly_ui(2, x).shape == (4,)
    assert acb_core.acb_polylog(s, z).shape == (4,)
    assert acb_core.acb_polylog_si(2, z).shape == (4,)
    assert acb_core.acb_agm(x, a).shape == (4,)
    assert acb_core.acb_agm1(x).shape == (4,)
    assert acb_core.acb_agm1_cpx(x).shape == (4,)
