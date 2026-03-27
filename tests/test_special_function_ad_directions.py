from __future__ import annotations

import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import bessel_kernels as bk
from arbplusjax import double_gamma


def test_incomplete_gamma_supports_argument_and_parameter_ad() -> None:
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)

    ds = jax.grad(lambda sv: api.incomplete_gamma_upper(sv, z, mode="point", method="quadrature"))(s)
    dz = jax.grad(lambda zv: api.incomplete_gamma_upper(s, zv, mode="point", method="quadrature"))(z)

    assert jnp.isfinite(ds)
    assert jnp.isfinite(dz)
    assert jnp.isclose(ds, api.incomplete_gamma_upper_parameter_derivative(s, z, method="quadrature"), rtol=5e-3, atol=5e-4)
    assert jnp.isclose(dz, api.incomplete_gamma_upper_argument_derivative(s, z), rtol=5e-3, atol=5e-4)


def test_hypgeom_1f1_supports_argument_and_parameter_ad() -> None:
    a = jnp.float64(1.25)
    b = jnp.float64(2.25)
    z = jnp.float64(0.3)

    da = jax.grad(lambda av: api.eval_point("hypgeom.arb_hypgeom_1f1", av, b, z))(a)
    dz = jax.grad(lambda zv: api.eval_point("hypgeom.arb_hypgeom_1f1", a, b, zv))(z)

    assert jnp.isfinite(da)
    assert jnp.isfinite(dz)


def test_hypgeom_u_supports_argument_and_parameter_ad_in_stable_regime() -> None:
    a = jnp.float64(1.0)
    b = jnp.float64(1.5)
    z = jnp.float64(0.2)

    da = jax.grad(lambda av: api.eval_point("hypgeom.arb_hypgeom_u", av, b, z))(a)
    dz = jax.grad(lambda zv: api.eval_point("hypgeom.arb_hypgeom_u", a, b, zv))(z)

    assert jnp.isfinite(da)
    assert jnp.isfinite(dz)


def test_barnes_ifj_supports_argument_and_tau_ad() -> None:
    z = jnp.asarray(1.1 + 0.05j, dtype=jnp.complex128)
    tau = jnp.float64(1.0)

    dx = jax.jacfwd(
        lambda xv: jnp.real(double_gamma.ifj_barnesdoublegamma(jnp.asarray(xv + 0.05j, dtype=jnp.complex128), tau, dps=60))
    )(jnp.float64(1.1))
    dtau = jax.jacfwd(lambda tv: jnp.real(double_gamma.ifj_barnesdoublegamma(z, tv, dps=60)))(tau)

    assert jnp.isfinite(dx)
    assert jnp.isfinite(dtau)


def test_bessel_point_kernel_supports_argument_and_order_ad() -> None:
    nu = jnp.float32(0.4)
    z = jnp.float32(2.5)

    dnu = jax.grad(lambda nv: bk.real_bessel_eval_j(nv, z))(nu)
    dz = jax.grad(lambda zv: bk.real_bessel_eval_j(nu, zv))(z)

    assert jnp.isfinite(dnu)
    assert jnp.isfinite(dz)
