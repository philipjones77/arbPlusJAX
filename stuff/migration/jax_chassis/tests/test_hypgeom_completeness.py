import jax.numpy as jnp

from arbjax import hypgeom


def _assert_interval(x: jnp.ndarray, length: int | None = None) -> None:
    arr = jnp.asarray(x)
    assert arr.shape[-1] == 2
    if length is not None:
        assert arr.shape[0] == length
    assert bool(jnp.all(arr[..., 0] <= arr[..., 1]))


def _assert_tuple_intervals(vals: tuple, length: int | None = None) -> None:
    assert isinstance(vals, tuple)
    for item in vals:
        _assert_interval(item, length=length)


def test_series_and_jet_helpers_shapes():
    length = 6
    x = jnp.array([0.9, 1.1], dtype=jnp.float64)
    s = jnp.array([1.2, 1.3], dtype=jnp.float64)
    z = jnp.array([0.2, 0.3], dtype=jnp.float64)

    _assert_interval(hypgeom.arb_hypgeom_rising_ui_jet_powsum(x, n=4, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_jet_rs(x, n=4, m=2, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_jet_bs(x, n=4, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_jet(x, n=4, length=length), length=length)

    _assert_interval(hypgeom.arb_hypgeom_gamma_lower_series(s, z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_gamma_upper_series(s, z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_beta_lower_series(s, s, z, length=length), length=length)

    _assert_tuple_intervals(hypgeom.arb_hypgeom_fresnel_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_ei_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_si_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_ci_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_shi_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_chi_series(z, length=length), length=length)

    z_pos = jnp.array([1.2, 1.3], dtype=jnp.float64)
    _assert_interval(hypgeom.arb_hypgeom_li_series(z_pos, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_erf_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_erfc_series(z, length=length), length=length)
    _assert_interval(hypgeom.arb_hypgeom_erfi_series(z, length=length), length=length)

    _assert_tuple_intervals(hypgeom.arb_hypgeom_airy_jet(z, length=length), length=length)
    _assert_tuple_intervals(hypgeom.arb_hypgeom_airy_series(z, length=length), length=length)

    l = jnp.array([1.0, 1.1], dtype=jnp.float64)
    eta = jnp.array([0.1, 0.2], dtype=jnp.float64)
    _assert_tuple_intervals(hypgeom.arb_hypgeom_coulomb_jet(l, eta, z, length=length), length=length)
    _assert_tuple_intervals(hypgeom.arb_hypgeom_coulomb_series(l, eta, z, length=length), length=length)


def test_lower_level_helpers_shapes():
    x = jnp.array([0.8, 1.0], dtype=jnp.float64)
    n = jnp.array([3.0, 3.0], dtype=jnp.float64)
    _assert_interval(hypgeom.arb_hypgeom_rising(x, n))
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_rs(x, n=3, m=2))
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_bs(x, n=3))
    _assert_interval(hypgeom.arb_hypgeom_rising_ui_rec(x, n=3))

    _assert_interval(hypgeom.arb_hypgeom_gamma_fmpq(3, 2))
    _assert_interval(hypgeom.arb_hypgeom_gamma_fmpz(5))
    _assert_interval(hypgeom.arb_hypgeom_gamma_stirling(x))
    _assert_interval(hypgeom.arb_hypgeom_gamma_stirling(x, reciprocal=True))
    _assert_interval(hypgeom.arb_hypgeom_gamma_stirling_sum_horner(x, n=4))
    _assert_interval(hypgeom.arb_hypgeom_gamma_stirling_sum_improved(x, n=4, k=2))

    root, weight = hypgeom.arb_hypgeom_legendre_p_ui_root(3, 2)
    _assert_interval(root)
    _assert_interval(weight)
    x2sub1 = x * x - 1.0
    d1, d2 = hypgeom.arb_hypgeom_legendre_p_ui_deriv_bound(3, x, x2sub1)
    _assert_interval(d1)
    _assert_interval(d2)


def test_pfq_helpers_shapes():
    a = jnp.array([1.1, 1.4], dtype=jnp.float64)
    b = jnp.array([2.2], dtype=jnp.float64)
    z = jnp.array([0.1, 0.2], dtype=jnp.float64)

    _assert_interval(hypgeom.arb_hypgeom_pfq(a, b, z))
    _assert_interval(hypgeom.arb_hypgeom_sum_fmpq_arb(a, b, z, reciprocal=False, n_terms=8))
    _assert_interval(hypgeom.arb_hypgeom_sum_fmpq_arb_forward(a, b, z, reciprocal=False, n_terms=8))
    _assert_interval(hypgeom.arb_hypgeom_sum_fmpq_arb_rs(a, b, z, reciprocal=False, n_terms=8))
    _assert_interval(hypgeom.arb_hypgeom_sum_fmpq_arb_bs(a, b, z, reciprocal=False, n_terms=8))

    re, im = hypgeom.arb_hypgeom_sum_fmpq_imag_arb(a, b, z, reciprocal=False, n_terms=8)
    _assert_interval(re)
    _assert_interval(im)
    re, im = hypgeom.arb_hypgeom_sum_fmpq_imag_arb_forward(a, b, z, reciprocal=False, n_terms=8)
    _assert_interval(re)
    _assert_interval(im)
    re, im = hypgeom.arb_hypgeom_sum_fmpq_imag_arb_rs(a, b, z, reciprocal=False, n_terms=8)
    _assert_interval(re)
    _assert_interval(im)
    re, im = hypgeom.arb_hypgeom_sum_fmpq_imag_arb_bs(a, b, z, reciprocal=False, n_terms=8)
    _assert_interval(re)
    _assert_interval(im)
