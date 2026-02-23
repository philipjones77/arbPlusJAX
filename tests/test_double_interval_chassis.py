import jax
import jax.numpy as jnp

from arbplusjax import double_interval as di


from tests._test_checks import _check
def _positive_interval(lo: float, hi: float) -> jax.Array:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def test_jit_compiles_and_keeps_interval_order():
    x = _positive_interval(1.5, 2.0)
    y = _positive_interval(3.0, 4.0)
    out = di.fast_mul_jit(x, y)
    _check(out.shape == (2,))
    _check(bool(out[0] <= out[1]))


def test_batch_vectorization_shape_and_order():
    x = di.interval(jnp.array([1.0, -2.0, 4.0]), jnp.array([2.0, -1.0, 5.0]))
    y = di.interval(jnp.array([3.0, 3.0, -6.0]), jnp.array([4.0, 4.0, -5.0]))
    out = di.batch_fast_mul(x, y)
    _check(out.shape == (3, 2))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))


def test_grad_through_midpoint_on_smooth_subdomain():
    y = _positive_interval(2.0, 3.0)

    def loss(x_bounds: jax.Array) -> jax.Array:
        z = di.fast_mul(x_bounds, y)
        return di.midpoint(z)

    x = _positive_interval(1.5, 2.5)
    grad = jax.grad(loss)(x)
    _check(grad.shape == (2,))
    _check(bool(jnp.all(jnp.isfinite(grad))))


def test_precision_semantics_widen_interval_at_lower_prec():
    x = _positive_interval(1.23456789, 1.23456791)
    y = _positive_interval(2.34567891, 2.34567899)
    hi_prec = di.fast_mul_prec(x, y, prec_bits=53)
    lo_prec = di.fast_mul_prec(x, y, prec_bits=20)
    _check(bool(di.contains(lo_prec, hi_prec)))
