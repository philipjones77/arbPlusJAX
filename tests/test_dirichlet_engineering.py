import jax
import jax.numpy as jnp

from arbplusjax import acb_core, acb_dirichlet, api, dirichlet, double_interval as di, point_wrappers


def _allclose(a, b, rtol=1e-10, atol=1e-10):
    return bool(jnp.allclose(jnp.asarray(a), jnp.asarray(b), rtol=rtol, atol=atol, equal_nan=True))


def _real_batch():
    lo = jnp.linspace(jnp.float64(1.2), jnp.float64(1.8), 4)
    hi = lo + jnp.float64(0.05)
    return di.interval(lo, hi)


def _complex_batch():
    re_lo = jnp.linspace(jnp.float64(1.2), jnp.float64(1.8), 4)
    re_hi = re_lo + jnp.float64(0.05)
    im_lo = jnp.linspace(jnp.float64(-0.2), jnp.float64(0.1), 4)
    im_hi = im_lo + jnp.float64(0.04)
    return acb_core.acb_box(di.interval(re_lo, re_hi), di.interval(im_lo, im_hi))


def test_dirichlet_real_batch_jit_and_prec_paths_are_stable():
    s = _real_batch()
    zeta_batch = dirichlet.dirichlet_zeta_batch(s, n_terms=24)
    zeta_jit = dirichlet.dirichlet_zeta_batch_jit(s, n_terms=24)
    eta_batch = dirichlet.dirichlet_eta_batch(s, n_terms=24)
    eta_jit = dirichlet.dirichlet_eta_batch_jit(s, n_terms=24)
    assert _allclose(zeta_batch, zeta_jit)
    assert _allclose(eta_batch, eta_jit)

    zeta_prec = dirichlet.dirichlet_zeta_batch_prec_jit(s, n_terms=24, prec_bits=53)
    eta_prec = dirichlet.dirichlet_eta_batch_prec_jit(s, n_terms=24, prec_bits=53)
    zeta_rig = dirichlet.dirichlet_zeta_rigorous(s, n_terms=24)
    eta_rig = dirichlet.dirichlet_eta_rigorous(s, n_terms=24)
    assert bool(jnp.all(di.contains(zeta_prec, zeta_batch)))
    assert bool(jnp.all(di.contains(eta_prec, eta_batch)))
    assert bool(jnp.all(di.contains(zeta_rig, zeta_batch)))
    assert eta_rig.shape == eta_batch.shape
    assert bool(jnp.all(jnp.isfinite(eta_rig)))
    assert bool(jnp.all(eta_rig[..., 0] <= eta_rig[..., 1]))


def test_dirichlet_real_ad_and_functional_relation_hold_on_safe_slice():
    zeta_grad = jax.grad(lambda x: jnp.squeeze(di.midpoint(dirichlet.dirichlet_zeta(di.interval(x, x), n_terms=24))))
    eta_grad = jax.grad(lambda x: jnp.squeeze(di.midpoint(dirichlet.dirichlet_eta(di.interval(x, x), n_terms=24))))
    for x in (jnp.float64(1.5), jnp.float64(2.0), jnp.float64(2.5)):
        assert jnp.isfinite(zeta_grad(x))
        assert jnp.isfinite(eta_grad(x))

    s = _real_batch()
    sm = di.midpoint(s)
    zeta_mid = di.midpoint(dirichlet.dirichlet_zeta_batch(s, n_terms=32))
    eta_mid = di.midpoint(dirichlet.dirichlet_eta_batch(s, n_terms=32))
    expected = (1.0 - jnp.exp((1.0 - sm) * jnp.log(2.0))) * zeta_mid
    assert _allclose(eta_mid, expected, rtol=1e-9, atol=1e-9)


def test_acb_dirichlet_point_fastpaths_and_api_bindings_match():
    s = _complex_batch()
    direct_fixed = point_wrappers.acb_dirichlet_zeta_batch_fixed_point(s, n_terms=48)
    direct_padded = point_wrappers.acb_dirichlet_zeta_batch_padded_point(s, pad_to=8, n_terms=48)
    assert _allclose(direct_fixed, direct_padded[: s.shape[0]])

    api_fixed = api.eval_point_batch("acb_dirichlet_zeta", s, n_terms=48)
    api_padded = api.eval_point_batch("acb_dirichlet_zeta", s, pad_to=8, n_terms=48)
    assert _allclose(api_fixed, direct_fixed)
    assert _allclose(api_padded, direct_fixed)

    jit_bound = api.bind_point_batch_jit("acb_dirichlet_zeta", dtype="float64", pad_to=8, n_terms=48)
    jit_out = jit_bound(s)
    assert _allclose(jit_out, direct_fixed)

    eta_fixed = point_wrappers.acb_dirichlet_eta_batch_fixed_point(s, n_terms=48)
    eta_api = api.eval_point_batch("acb_dirichlet_eta", s, pad_to=8, n_terms=48)
    assert _allclose(eta_api, eta_fixed)


def test_acb_dirichlet_prec_and_ad_are_stable():
    s = _complex_batch()
    zeta_prec = acb_dirichlet.acb_dirichlet_zeta_batch_prec_jit(s, n_terms=48, prec_bits=53)
    eta_prec = acb_dirichlet.acb_dirichlet_eta_batch_prec_jit(s, n_terms=48, prec_bits=53)
    zeta = acb_dirichlet.acb_dirichlet_zeta_batch_jit(s, n_terms=48)
    eta = acb_dirichlet.acb_dirichlet_eta_batch_jit(s, n_terms=48)
    assert bool(jnp.all(di.contains(acb_core.acb_real(zeta_prec), acb_core.acb_real(zeta))))
    assert bool(jnp.all(di.contains(acb_core.acb_imag(zeta_prec), acb_core.acb_imag(zeta))))
    assert bool(jnp.all(di.contains(acb_core.acb_real(eta_prec), acb_core.acb_real(eta))))
    assert bool(jnp.all(di.contains(acb_core.acb_imag(eta_prec), acb_core.acb_imag(eta))))

    def loss(x):
        s_box = acb_core.acb_box(di.interval(x, x), di.interval(jnp.float64(0.2), jnp.float64(0.2)))
        return jnp.real(acb_core.acb_midpoint(acb_dirichlet.acb_dirichlet_zeta(s_box, n_terms=48)))

    grad = jax.grad(loss)
    for x in (jnp.float64(1.5), jnp.float64(2.0), jnp.float64(2.5)):
        assert jnp.isfinite(grad(x))
