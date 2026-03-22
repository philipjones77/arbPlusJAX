import jax
import jax.numpy as jnp

from arbplusjax.special.gamma.incomplete_gamma import incomplete_gamma_upper_point
from arbplusjax.special.gamma.incomplete_gamma_ad import incomplete_gamma_upper_switched_point_with_fingerprint
from arbplusjax.special.gamma.incomplete_gamma_ad import incomplete_gamma_upper_switched_z_vjp


def _rel_err(a, b):
    denom = jnp.maximum(jnp.abs(b), jnp.asarray(1e-6, dtype=jnp.float32))
    return jnp.abs(a - b) / denom


def test_incomplete_gamma_switched_smoke_fp32_value_and_fingerprint():
    s = jnp.float32(1.25)
    z = jnp.float32(0.15)
    value, fingerprint = incomplete_gamma_upper_switched_point_with_fingerprint(s, z)
    ref = jnp.asarray(incomplete_gamma_upper_point(s, z), dtype=jnp.float32)

    assert float(_rel_err(jnp.asarray(value, dtype=jnp.float32), ref)) <= 1e-5
    assert int(fingerprint.regime_code) in (0, 1, 2)
    assert int(fingerprint.method_code) >= 0
    assert int(fingerprint.work_units) >= 0
    assert float(fingerprint.adjoint_residual) == 0.0


def test_incomplete_gamma_switched_smoke_fp32_regime_stable_under_tiny_perturbation():
    s = jnp.float32(2.0)
    z = jnp.float32(3.0)
    _, fp0 = incomplete_gamma_upper_switched_point_with_fingerprint(s, z)
    _, fp1 = incomplete_gamma_upper_switched_point_with_fingerprint(s, z + jnp.float32(1e-6))

    assert int(fp0.regime_code) == int(fp1.regime_code)
    assert int(fp0.method_code) == int(fp1.method_code)


def test_incomplete_gamma_switched_smoke_fp32_grad_is_finite():
    s = jnp.float32(1.75)
    z = jnp.float32(0.6)
    grad = jax.grad(lambda zz: incomplete_gamma_upper_switched_z_vjp(s, zz))(z)
    assert bool(jnp.isfinite(grad))
