import jax
import jax.numpy as jnp

from arbplusjax import elementary as el

from tests._test_checks import _check


def test_constants_and_dtype_helpers():
    _check(bool(jnp.isfinite(el.PI)))
    _check(bool(jnp.isfinite(el.LOG_TWO_PI)))
    z = el.complex_promote(jnp.float64(1.0))
    _check(bool(jnp.issubdtype(z.dtype, jnp.complexfloating)))
    _check(bool(jnp.isfinite(el.eps(jnp.float64))))
    _check(bool(jnp.isfinite(el.tiny(jnp.float64))))
    _check(bool(jnp.isfinite(el.max_value(jnp.float64))))


def test_stable_log_domain_ops_are_finite():
    x = jnp.asarray([-20.0, -2.0, -0.8], dtype=jnp.float64)
    y = jnp.asarray([-25.0, -3.0, -1.2], dtype=jnp.float64)
    _check(bool(jnp.all(jnp.isfinite(el.logaddexp(x, y)))))
    _check(bool(jnp.all(jnp.isfinite(el.logsubexp(x, y)))))
    _check(bool(jnp.all(jnp.isfinite(el.log1mexp(x)))))
    _check(bool(jnp.all(jnp.isfinite(el.logexpm1(jnp.asarray([0.2, 1.0, 4.0], dtype=jnp.float64))))))


def test_complex_branch_helpers_and_grad():
    z = jnp.asarray([0.7 + 0.2j, 1.3 - 0.5j], dtype=jnp.complex128)
    s = jnp.asarray([0.4 + 0.1j, -0.2 + 0.3j], dtype=jnp.complex128)
    out = el.z_to_minus_s(z, s)
    _check(bool(jnp.all(jnp.isfinite(jnp.real(out)) & jnp.isfinite(jnp.imag(out)))))

    def loss(t):
        return jnp.real(el.cpow(1.2 + 0.0j, t + 0.0j))

    g = jax.grad(loss)(jnp.float64(0.7))
    _check(bool(jnp.isfinite(g)))
