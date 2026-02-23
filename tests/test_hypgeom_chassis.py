import jax
import jax.numpy as jnp

from arbplusjax import hypgeom


from tests._test_checks import _check
def test_arb_rising_jit_and_vectorization():
    x = jnp.array([[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]], dtype=jnp.float64)
    out = hypgeom.arb_hypgeom_rising_ui_batch_jit(x, n=6)
    _check(out.shape == (3, 2))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))


def test_acb_rising_jit_and_order():
    x = jnp.array([[1.0, 1.1, 0.5, 0.6], [2.0, 2.2, 0.1, 0.2]], dtype=jnp.float64)
    out = hypgeom.acb_hypgeom_rising_ui_batch_jit(x, n=5)
    _check(out.shape == (2, 4))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))
    _check(bool(jnp.all(out[:, 2] <= out[:, 3])))


def test_grad_real_rising_midpoint():
    x = jnp.array([1.2, 1.3], dtype=jnp.float64)

    def loss(v):
        y = hypgeom.arb_hypgeom_rising_ui(v, n=7)
        return jnp.sum(y)

    g = jax.grad(loss)(x)
    _check(g.shape == (2,))
    _check(bool(jnp.all(jnp.isfinite(g))))


def test_grad_complex_rising_midpoint_abs2():
    x = jnp.array([1.1, 1.2, 0.2, 0.3], dtype=jnp.float64)

    def loss(v):
        y = hypgeom.acb_hypgeom_rising_ui(v, n=4)
        m = hypgeom.acb_midpoint(y)
        return jnp.real(m * jnp.conj(m))

    g = jax.grad(loss)(x)
    _check(g.shape == (4,))
    _check(bool(jnp.all(jnp.isfinite(g))))


def test_lgamma_jit_real_and_complex():
    xr = jnp.array([1.3, 1.6], dtype=jnp.float64)
    xc = jnp.array([1.1, 1.3, 0.4, 0.5], dtype=jnp.float64)
    yr = hypgeom.arb_hypgeom_lgamma(xr)
    yc = hypgeom.acb_hypgeom_lgamma(xc)
    _check(yr.shape == (2,))
    _check(yc.shape == (4,))
    _check(bool(yr[0] <= yr[1]))
    _check(bool(yc[0] <= yc[1]))
    _check(bool(yc[2] <= yc[3]))


def test_lgamma_grad_paths():
    xr = jnp.array([1.2, 1.25], dtype=jnp.float64)
    xc = jnp.array([1.4, 1.45, 0.3, 0.35], dtype=jnp.float64)

    def real_loss(v):
        y = hypgeom.arb_hypgeom_lgamma(v)
        return jnp.sum(y)

    def complex_loss(v):
        y = hypgeom.acb_hypgeom_lgamma(v)
        m = hypgeom.acb_midpoint(y)
        return jnp.real(m * jnp.conj(m))

    gr = jax.grad(real_loss)(xr)
    gc = jax.grad(complex_loss)(xc)
    _check(bool(jnp.all(jnp.isfinite(gr))))
    _check(bool(jnp.all(jnp.isfinite(gc))))


def test_gamma_rgamma_jit_shapes():
    xr = jnp.array([1.2, 1.3], dtype=jnp.float64)
    xc = jnp.array([1.1, 1.2, 0.2, 0.25], dtype=jnp.float64)
    gr = hypgeom.arb_hypgeom_gamma(xr)
    rr = hypgeom.arb_hypgeom_rgamma(xr)
    gc = hypgeom.acb_hypgeom_gamma(xc)
    rc = hypgeom.acb_hypgeom_rgamma(xc)
    _check(gr.shape == (2,))
    _check(rr.shape == (2,))
    _check(gc.shape == (4,))
    _check(rc.shape == (4,))
    _check(bool(gr[0] <= gr[1] and rr[0] <= rr[1]))
    _check(bool(gc[0] <= gc[1] and gc[2] <= gc[3]))
    _check(bool(rc[0] <= rc[1] and rc[2] <= rc[3]))


def test_gamma_rgamma_grad_paths():
    xr = jnp.array([1.25, 1.3], dtype=jnp.float64)
    xc = jnp.array([1.35, 1.4, 0.2, 0.25], dtype=jnp.float64)

    def real_gamma_loss(v):
        y = hypgeom.arb_hypgeom_gamma(v)
        return jnp.sum(y)

    def real_rgamma_loss(v):
        y = hypgeom.arb_hypgeom_rgamma(v)
        return jnp.sum(y)

    def cplx_gamma_loss(v):
        y = hypgeom.acb_hypgeom_gamma(v)
        m = hypgeom.acb_midpoint(y)
        return jnp.real(m * jnp.conj(m))

    def cplx_rgamma_loss(v):
        y = hypgeom.acb_hypgeom_rgamma(v)
        m = hypgeom.acb_midpoint(y)
        return jnp.real(m * jnp.conj(m))

    g1 = jax.grad(real_gamma_loss)(xr)
    g2 = jax.grad(real_rgamma_loss)(xr)
    g3 = jax.grad(cplx_gamma_loss)(xc)
    g4 = jax.grad(cplx_rgamma_loss)(xc)
    _check(bool(jnp.all(jnp.isfinite(g1))))
    _check(bool(jnp.all(jnp.isfinite(g2))))
    _check(bool(jnp.all(jnp.isfinite(g3))))
    _check(bool(jnp.all(jnp.isfinite(g4))))


def test_erf_family_jit_shapes():
    xr = jnp.array([-0.8, 0.9], dtype=jnp.float64)
    xc = jnp.array([-0.6, 0.7, -0.5, 0.6], dtype=jnp.float64)
    er = hypgeom.arb_hypgeom_erf(xr)
    ec = hypgeom.acb_hypgeom_erf(xc)
    cr = hypgeom.arb_hypgeom_erfc(xr)
    cc = hypgeom.acb_hypgeom_erfc(xc)
    ir = hypgeom.arb_hypgeom_erfi(xr)
    ic = hypgeom.acb_hypgeom_erfi(xc)
    _check(er.shape == (2,) and cr.shape == (2,) and ir.shape == (2,))
    _check(ec.shape == (4,) and cc.shape == (4,) and ic.shape == (4,))
    _check(bool(er[0] <= er[1] and cr[0] <= cr[1] and ir[0] <= ir[1]))
    _check(bool(ec[0] <= ec[1] and ec[2] <= ec[3]))
    _check(bool(cc[0] <= cc[1] and cc[2] <= cc[3]))
    _check(bool(ic[0] <= ic[1] and ic[2] <= ic[3]))


def test_erf_family_grad_paths():
    xr = jnp.array([-0.7, 0.8], dtype=jnp.float64)
    xc = jnp.array([-0.6, 0.65, -0.4, 0.45], dtype=jnp.float64)

    def real_loss(v):
        return jnp.sum(hypgeom.arb_hypgeom_erf(v)) + jnp.sum(hypgeom.arb_hypgeom_erfc(v)) + jnp.sum(hypgeom.arb_hypgeom_erfi(v))

    def complex_loss(v):
        y = hypgeom.acb_hypgeom_erf(v)
        z = hypgeom.acb_hypgeom_erfc(v)
        w = hypgeom.acb_hypgeom_erfi(v)
        m = hypgeom.acb_midpoint(y) + hypgeom.acb_midpoint(z) + hypgeom.acb_midpoint(w)
        return jnp.real(m * jnp.conj(m))

    gr = jax.grad(real_loss)(xr)
    gc = jax.grad(complex_loss)(xc)
    _check(bool(jnp.all(jnp.isfinite(gr))))
    _check(bool(jnp.all(jnp.isfinite(gc))))


def test_barnesg_shapes_and_grad_paths():
    xr = jnp.array([1.2, 1.4], dtype=jnp.float64)
    xc = jnp.array([1.1, 1.3, 0.2, 0.25], dtype=jnp.float64)

    gr = hypgeom.arb_hypgeom_barnesg(xr)
    gc = hypgeom.acb_hypgeom_barnesg(xc)
    _check(gr.shape == (2,))
    _check(gc.shape == (4,))
    _check(bool(gr[0] <= gr[1]))
    _check(bool(gc[0] <= gc[1] and gc[2] <= gc[3]))

    def real_loss(v):
        return jnp.sum(hypgeom.arb_hypgeom_barnesg(v))

    def complex_loss(v):
        y = hypgeom.acb_hypgeom_barnesg(v)
        m = hypgeom.acb_midpoint(y)
        return jnp.real(m * jnp.conj(m))

    gr2 = jax.grad(real_loss)(xr)
    gc2 = jax.grad(complex_loss)(xc)
    _check(bool(jnp.all(jnp.isfinite(gr2))))
    _check(bool(jnp.all(jnp.isfinite(gc2))))


def test_erfinv_erfcinv_shapes_and_grad_paths():
    xr = jnp.array([-0.6, 0.7], dtype=jnp.float64)
    xr_erfc = jnp.array([0.2, 1.7], dtype=jnp.float64)

    erfi = hypgeom.arb_hypgeom_erfinv(xr)
    erfc = hypgeom.arb_hypgeom_erfcinv(xr_erfc)
    _check(erfi.shape == (2,))
    _check(erfc.shape == (2,))
    _check(bool(erfi[0] <= erfi[1]))
    _check(bool(erfc[0] <= erfc[1]))

    def loss(v):
        return jnp.sum(hypgeom.arb_hypgeom_erfinv(v))

    g = jax.grad(loss)(xr)
    _check(g.shape == (2,))
    _check(bool(jnp.all(jnp.isfinite(g))))


def test_1f1_2f1_jit_shapes():
    a = jnp.array([1.1, 1.2], dtype=jnp.float64)
    b = jnp.array([2.0, 2.1], dtype=jnp.float64)
    c = jnp.array([2.4, 2.5], dtype=jnp.float64)
    z = jnp.array([-0.2, 0.3], dtype=jnp.float64)
    ac = jnp.array([1.0, 1.1, 0.1, 0.2], dtype=jnp.float64)
    bc = jnp.array([2.0, 2.1, 0.1, 0.2], dtype=jnp.float64)
    cc = jnp.array([2.5, 2.6, 0.1, 0.2], dtype=jnp.float64)
    zc = jnp.array([-0.3, 0.3, -0.2, 0.2], dtype=jnp.float64)

    r0 = hypgeom.arb_hypgeom_0f1(b, z, regularized=False)
    r1 = hypgeom.arb_hypgeom_1f1(a, b, z)
    rm = hypgeom.arb_hypgeom_m(a, b, z, regularized=True)
    r2 = hypgeom.arb_hypgeom_2f1(a, b, c, z)
    c0 = hypgeom.acb_hypgeom_0f1(bc, zc, regularized=False)
    c1 = hypgeom.acb_hypgeom_1f1(ac, bc, zc)
    cm = hypgeom.acb_hypgeom_m(ac, bc, zc, regularized=True)
    c2 = hypgeom.acb_hypgeom_2f1(ac, bc, cc, zc)
    _check(r0.shape == (2,) and r1.shape == (2,) and rm.shape == (2,) and r2.shape == (2,))
    _check(c0.shape == (4,) and c1.shape == (4,) and cm.shape == (4,) and c2.shape == (4,))
    _check(bool(r0[0] <= r0[1] and r1[0] <= r1[1] and rm[0] <= rm[1] and r2[0] <= r2[1]))
    _check(bool(c0[0] <= c0[1] and c0[2] <= c0[3]))
    _check(bool(c1[0] <= c1[1] and c1[2] <= c1[3]))
    _check(bool(cm[0] <= cm[1] and cm[2] <= cm[3]))
    _check(bool(c2[0] <= c2[1] and c2[2] <= c2[3]))


def test_1f1_2f1_grad_paths():
    a = jnp.array([1.1, 1.15], dtype=jnp.float64)
    b = jnp.array([2.1, 2.2], dtype=jnp.float64)
    c = jnp.array([2.6, 2.7], dtype=jnp.float64)
    z = jnp.array([-0.2, 0.25], dtype=jnp.float64)
    ac = jnp.array([1.1, 1.2, 0.1, 0.2], dtype=jnp.float64)
    bc = jnp.array([2.2, 2.3, 0.1, 0.2], dtype=jnp.float64)
    cc = jnp.array([2.8, 2.9, 0.1, 0.2], dtype=jnp.float64)
    zc = jnp.array([-0.2, 0.2, -0.15, 0.15], dtype=jnp.float64)

    def real_loss(v):
        return (
            jnp.sum(hypgeom.arb_hypgeom_0f1(b, v, regularized=False))
            + jnp.sum(hypgeom.arb_hypgeom_m(a, b, v, regularized=True))
            + jnp.sum(hypgeom.arb_hypgeom_1f1(a, b, v))
            + jnp.sum(hypgeom.arb_hypgeom_2f1(a, b, c, v, regularized=True))
        )

    def cplx_loss(v):
        u = hypgeom.acb_hypgeom_0f1(bc, v, regularized=False)
        y = hypgeom.acb_hypgeom_m(ac, bc, v, regularized=True)
        w = hypgeom.acb_hypgeom_2f1(ac, bc, cc, v, regularized=True)
        q = hypgeom.acb_hypgeom_1f1(ac, bc, v)
        m = hypgeom.acb_midpoint(u) + hypgeom.acb_midpoint(y) + hypgeom.acb_midpoint(w) + hypgeom.acb_midpoint(q)
        return jnp.real(m * jnp.conj(m))

    gr = jax.grad(real_loss)(z)
    gc = jax.grad(cplx_loss)(zc)
    _check(bool(jnp.all(jnp.isfinite(gr))))
    _check(bool(jnp.all(jnp.isfinite(gc))))


def test_u_and_integration_paths():
    a = jnp.array([1.1, 1.2], dtype=jnp.float64)
    b = jnp.array([1.8, 1.9], dtype=jnp.float64)
    z = jnp.array([0.2, 0.3], dtype=jnp.float64)
    ac = jnp.array([1.1, 1.2, 0.1, 0.2], dtype=jnp.float64)
    bc = jnp.array([1.6, 1.7, 0.1, 0.2], dtype=jnp.float64)
    zc = jnp.array([0.2, 0.3, 0.1, 0.15], dtype=jnp.float64)

    ur = hypgeom.arb_hypgeom_u(a, b, z)
    ui = hypgeom.arb_hypgeom_u_integration(a, b, z)
    uc = hypgeom.acb_hypgeom_u(ac, bc, zc)
    uci = hypgeom.acb_hypgeom_u_integration(ac, bc, zc)
    _check(ur.shape == (2,) and ui.shape == (2,))
    _check(uc.shape == (4,) and uci.shape == (4,))

    def real_loss(v):
        return jnp.sum(hypgeom.arb_hypgeom_u(a, b, v)) + jnp.sum(hypgeom.arb_hypgeom_1f1_integration(a, b, v, regularized=True))

    def cplx_loss(v):
        u = hypgeom.acb_hypgeom_u(ac, bc, v)
        w = hypgeom.acb_hypgeom_2f1_integration(ac, bc, ac, v, regularized=True)
        m = hypgeom.acb_midpoint(u) + hypgeom.acb_midpoint(w)
        return jnp.real(m * jnp.conj(m))

    gr = jax.grad(real_loss)(z)
    gc = jax.grad(cplx_loss)(zc)
    _check(bool(jnp.all(jnp.isfinite(gr))))
    _check(bool(jnp.all(jnp.isfinite(gc))))


def test_bessel_shapes_and_grad_paths():
    nu = jnp.array([0.5, 0.6], dtype=jnp.float64)
    z = jnp.array([0.4, 0.5], dtype=jnp.float64)
    nu_c = jnp.array([0.5, 0.6, 0.1, 0.2], dtype=jnp.float64)
    z_c = jnp.array([0.4, 0.5, 0.2, 0.25], dtype=jnp.float64)

    j = hypgeom.arb_hypgeom_bessel_j(nu, z)
    y = hypgeom.arb_hypgeom_bessel_y(nu, z)
    i = hypgeom.arb_hypgeom_bessel_i(nu, z)
    k = hypgeom.arb_hypgeom_bessel_k(nu, z)
    _check(j.shape == (2,) and y.shape == (2,) and i.shape == (2,) and k.shape == (2,))

    jc = hypgeom.acb_hypgeom_bessel_j(nu_c, z_c)
    yc = hypgeom.acb_hypgeom_bessel_y(nu_c, z_c)
    ic = hypgeom.acb_hypgeom_bessel_i(nu_c, z_c)
    kc = hypgeom.acb_hypgeom_bessel_k(nu_c, z_c)
    _check(jc.shape == (4,) and yc.shape == (4,) and ic.shape == (4,) and kc.shape == (4,))

    def real_loss(v):
        return jnp.sum(hypgeom.arb_hypgeom_bessel_j(nu, v)) + jnp.sum(hypgeom.arb_hypgeom_bessel_i(nu, v))

    def cplx_loss(v):
        m = hypgeom.acb_midpoint(hypgeom.acb_hypgeom_bessel_j(nu_c, v))
        return jnp.real(m * jnp.conj(m))

    gr = jax.grad(real_loss)(z)
    gc = jax.grad(cplx_loss)(z_c)
    _check(bool(jnp.all(jnp.isfinite(gr))))
    _check(bool(jnp.all(jnp.isfinite(gc))))

def test_precision_semantics_for_hypgeom_kernels():
    xr = jnp.array([1.4, 1.41], dtype=jnp.float64)
    xc = jnp.array([1.2, 1.25, 0.3, 0.35], dtype=jnp.float64)

    r_hi = hypgeom.arb_hypgeom_rising_ui_prec(xr, n=8, prec_bits=53)
    r_lo = hypgeom.arb_hypgeom_rising_ui_prec(xr, n=8, prec_bits=20)
    _check(bool((r_lo[0] <= r_hi[0]) and (r_lo[1] >= r_hi[1])))

    lg_hi = hypgeom.acb_hypgeom_lgamma_prec(xc, prec_bits=53)
    lg_lo = hypgeom.acb_hypgeom_lgamma_prec(xc, prec_bits=20)
    _check(bool(lg_lo[0] <= lg_hi[0] and lg_lo[1] >= lg_hi[1]))
    _check(bool(lg_lo[2] <= lg_hi[2] and lg_lo[3] >= lg_hi[3]))

    gg_hi = hypgeom.arb_hypgeom_gamma_prec(xr, prec_bits=53)
    gg_lo = hypgeom.arb_hypgeom_gamma_prec(xr, prec_bits=20)
    _check(bool(gg_lo[0] <= gg_hi[0] and gg_lo[1] >= gg_hi[1]))

    ef_hi = hypgeom.arb_hypgeom_erf_prec(xr, prec_bits=53)
    ef_lo = hypgeom.arb_hypgeom_erf_prec(xr, prec_bits=20)
    _check(bool(ef_lo[0] <= ef_hi[0] and ef_lo[1] >= ef_hi[1]))

    a = jnp.array([1.1, 1.2], dtype=jnp.float64)
    b = jnp.array([2.1, 2.2], dtype=jnp.float64)
    z = jnp.array([-0.2, 0.25], dtype=jnp.float64)
    z0_hi = hypgeom.arb_hypgeom_0f1_prec(b, z, prec_bits=53, regularized=True)
    z0_lo = hypgeom.arb_hypgeom_0f1_prec(b, z, prec_bits=20, regularized=True)
    _check(bool(z0_lo[0] <= z0_hi[0] and z0_lo[1] >= z0_hi[1]))

    m_hi = hypgeom.arb_hypgeom_m_prec(a, b, z, prec_bits=53, regularized=True)
    m_lo = hypgeom.arb_hypgeom_m_prec(a, b, z, prec_bits=20, regularized=True)
    _check(bool(m_lo[0] <= m_hi[0] and m_lo[1] >= m_hi[1]))

    f_hi = hypgeom.arb_hypgeom_1f1_prec(a, b, z, prec_bits=53)
    f_lo = hypgeom.arb_hypgeom_1f1_prec(a, b, z, prec_bits=20)
    _check(bool(f_lo[0] <= f_hi[0] and f_lo[1] >= f_hi[1]))


def test_additional_specials_shapes():
    x = jnp.array([0.2, 0.3], dtype=jnp.float64)
    s, c = hypgeom.arb_hypgeom_fresnel(x, normalized=True)
    _check(s.shape == (2,) and c.shape == (2,))
    _check(bool(s[0] <= s[1] and c[0] <= c[1]))

    for fn in (
        hypgeom.arb_hypgeom_ei,
        hypgeom.arb_hypgeom_si,
        hypgeom.arb_hypgeom_ci,
        hypgeom.arb_hypgeom_shi,
        hypgeom.arb_hypgeom_chi,
        hypgeom.arb_hypgeom_li,
        hypgeom.arb_hypgeom_dilog,
    ):
        out = fn(x)
        _check(out.shape == (2,))
        _check(bool(out[0] <= out[1]))


def test_new_arb_hypgeom_stage2_shapes():
    x = jnp.array([0.2, 0.3], dtype=jnp.float64)
    s = jnp.array([1.0, 1.0], dtype=jnp.float64)
    a = jnp.array([0.5, 0.5], dtype=jnp.float64)
    b = jnp.array([1.5, 1.5], dtype=jnp.float64)

    ai, aip, bi, bip = hypgeom.arb_hypgeom_airy(x)
    _check(ai.shape == (2,) and aip.shape == (2,) and bi.shape == (2,) and bip.shape == (2,))

    _check(hypgeom.arb_hypgeom_expint(s, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_gamma_lower(s, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_gamma_upper(s, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_beta_lower(a, b, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_chebyshev_t(3, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_chebyshev_u(3, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_laguerre_l(3, a, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_hermite_h(4, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_legendre_p(3, a, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_legendre_q(2, a, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_jacobi_p(3, a, b, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_gegenbauer_c(3, a, x).shape == (2,))
    _check(hypgeom.arb_hypgeom_central_bin_ui(5).shape == (2,))


def test_hypgeom_series_and_helpers_shapes():
    x = jnp.array([0.2, 0.3], dtype=jnp.float64)
    s = jnp.array([1.0, 1.0], dtype=jnp.float64)
    a = jnp.array([0.5, 0.5], dtype=jnp.float64)
    b = jnp.array([1.5, 1.5], dtype=jnp.float64)
    eta = jnp.array([0.0, 0.0], dtype=jnp.float64)

    _check(hypgeom.arb_hypgeom_rising_ui_jet(x, 3, 4).shape == (4, 2))
    _check(hypgeom.arb_hypgeom_gamma_lower_series(s, x, 4).shape == (4, 2))
    _check(hypgeom.arb_hypgeom_gamma_upper_series(s, x, 4).shape == (4, 2))
    _check(hypgeom.arb_hypgeom_beta_lower_series(a, b, x, 4).shape == (4, 2))
    _check(hypgeom.arb_hypgeom_erf_series(x, 4).shape == (4, 2))
    _check(hypgeom.arb_hypgeom_ei_series(x, 4).shape == (4, 2))
    s_coef, c_coef = hypgeom.arb_hypgeom_fresnel_series(x, 4)
    _check(s_coef.shape == (4, 2) and c_coef.shape == (4, 2))

    ai0, aip0, bi0, bip0 = hypgeom.arb_hypgeom_airy_zero(1)
    _check(ai0.shape == (2,) and aip0.shape == (2,) and bi0.shape == (2,) and bip0.shape == (2,))

    f, g = hypgeom.arb_hypgeom_coulomb(a, eta, x)
    _check(f.shape == (2,) and g.shape == (2,))

    root, weight = hypgeom.arb_hypgeom_legendre_p_ui_root(4, 2)
    _check(root.shape == (2,) and weight.shape == (2,))
