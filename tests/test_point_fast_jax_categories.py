from __future__ import annotations

import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import srb_mat


def _real_interval_batch(values: jax.Array) -> jax.Array:
    return di.interval(values, values)


def test_core_scalar_point_fast_jax_surface_has_compiled_batch_and_vmap_parity() -> None:
    x = jnp.linspace(0.1, 1.2, 8, dtype=jnp.float32)
    bound = api.bind_point_batch_jit("arb_fpwrap_double_exp", dtype="float32", pad_to=8)

    got = bound(x)
    expected = jax.vmap(lambda t: api.eval_point("arb_fpwrap_double_exp", t, dtype="float32"))(x)

    assert got.dtype == jnp.float32
    assert got.shape == x.shape
    assert jnp.allclose(got, expected)
    assert jnp.allclose(bound(x), got)


def test_core_scalar_alias_fastpaths_have_compiled_batch_parity() -> None:
    x = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
    y = jnp.linspace(0.5, 1.2, 8, dtype=jnp.float64)
    bound_abs = api.bind_point_batch_jit("arb_abs", dtype="float64", pad_to=8)
    bound_add = api.bind_point_batch_jit("arb_add", dtype="float64", pad_to=8)
    positive = jnp.linspace(0.5, 1.5, 8, dtype=jnp.float64)
    bound_mul = api.bind_point_batch_jit("arb_mul", dtype="float64", pad_to=8)
    bound_exp = api.bind_point_batch_jit("arb_exp", dtype="float64", pad_to=8)
    bound_log = api.bind_point_batch_jit("arb_log", dtype="float64", pad_to=8)

    got_abs = bound_abs(x)
    ref_abs = jax.vmap(lambda t: api.eval_point("arb_abs", t, dtype="float64"))(x)
    got_add = bound_add(x, y)
    ref_add = jax.vmap(lambda a, b: api.eval_point("arb_add", a, b, dtype="float64"))(x, y)
    got_mul = bound_mul(x, y)
    ref_mul = jax.vmap(lambda a, b: api.eval_point("arb_mul", a, b, dtype="float64"))(x, y)
    got_exp = bound_exp(x)
    ref_exp = jax.vmap(lambda t: api.eval_point("arb_exp", t, dtype="float64"))(x)
    got_log = bound_log(positive)
    ref_log = jax.vmap(lambda t: api.eval_point("arb_log", t, dtype="float64"))(positive)

    assert jnp.allclose(got_abs, ref_abs)
    assert jnp.allclose(got_add, ref_add)
    assert jnp.allclose(got_mul, ref_mul)
    assert jnp.allclose(got_exp, ref_exp)
    assert jnp.allclose(got_log, ref_log)


def test_core_scalar_transcendental_alias_fastpaths_have_compiled_batch_parity() -> None:
    xr = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float64)
    xc = jnp.asarray(
        [0.1 + 0.05j, 0.2 + 0.1j, 0.3 + 0.15j, 0.4 + 0.2j, 0.5 + 0.1j, 0.6 + 0.05j, 0.7 + 0.15j, 0.8 + 0.1j],
        dtype=jnp.complex128,
    )
    bound_real = api.bind_point_batch_jit("arb_acos", dtype="float64", pad_to=8)
    bound_real_expm1 = api.bind_point_batch_jit("arb_expm1", dtype="float64", pad_to=8)
    bound_complex = api.bind_point_batch_jit("acb_log1p", pad_to=8)
    bound_complex_trig = api.bind_point_batch_jit("acb_sin_pi", pad_to=8)

    got_real = bound_real(xr)
    ref_real = jax.vmap(lambda t: api.eval_point("arb_acos", t, dtype="float64"))(xr)
    got_real_expm1 = bound_real_expm1(xr)
    ref_real_expm1 = jax.vmap(lambda t: api.eval_point("arb_expm1", t, dtype="float64"))(xr)
    got_complex = bound_complex(xc)
    ref_complex = jax.vmap(lambda z: api.eval_point("acb_log1p", z))(xc)
    got_complex_trig = bound_complex_trig(xc)
    ref_complex_trig = jax.vmap(lambda z: api.eval_point("acb_sin_pi", z))(xc)

    assert jnp.allclose(got_real, ref_real)
    assert jnp.allclose(got_real_expm1, ref_real_expm1)
    assert jnp.allclose(got_complex, ref_complex)
    assert jnp.allclose(got_complex_trig, ref_complex_trig)


def test_core_scalar_static_and_tuple_fastpaths_have_compiled_batch_parity() -> None:
    xr = jnp.linspace(0.4, 1.1, 8, dtype=jnp.float64)
    yr = jnp.linspace(1.0, 1.7, 8, dtype=jnp.float64)
    zr = jnp.linspace(-0.3, 0.4, 8, dtype=jnp.float64)
    xc = jnp.asarray(
        [0.4 + 0.05j, 0.5 + 0.1j, 0.6 + 0.15j, 0.7 + 0.2j, 0.8 + 0.1j, 0.9 + 0.05j, 1.0 + 0.15j, 1.1 + 0.1j],
        dtype=jnp.complex128,
    )
    yc = jnp.asarray(
        [1.0 + 0.1j, 1.1 + 0.05j, 1.2 + 0.2j, 1.3 + 0.1j, 1.4 + 0.05j, 1.5 + 0.2j, 1.6 + 0.1j, 1.7 + 0.05j],
        dtype=jnp.complex128,
    )
    zc = jnp.asarray(
        [-0.3 + 0.02j, -0.2 + 0.04j, -0.1 + 0.06j, 0.0 + 0.08j, 0.1 + 0.02j, 0.2 + 0.04j, 0.3 + 0.06j, 0.4 + 0.08j],
        dtype=jnp.complex128,
    )

    bound_fma = api.bind_point_batch_jit("arb_fma", dtype="float64", pad_to=8)
    bound_pow_ui = api.bind_point_batch_jit("arb_pow_ui", dtype="float64", pad_to=8, n=3)
    bound_root_ui = api.bind_point_batch_jit("arb_root_ui", dtype="float64", pad_to=8, k=3)
    bound_sin_cos = api.bind_point_batch_jit("arb_sin_cos", dtype="float64", pad_to=8)
    bound_sinh_cosh = api.bind_point_batch_jit("arb_sinh_cosh", dtype="float64", pad_to=8)
    bound_complex_pow_ui = api.bind_point_batch_jit("acb_pow_ui", pad_to=8, n=3)
    bound_complex_sin_cos = api.bind_point_batch_jit("acb_sin_cos", pad_to=8)

    got_fma = bound_fma(xr, yr, zr)
    ref_fma = jax.vmap(lambda a, b, c: api.eval_point("arb_fma", a, b, c, dtype="float64"))(xr, yr, zr)
    got_pow_ui = bound_pow_ui(xr)
    ref_pow_ui = jax.vmap(lambda t: api.eval_point("arb_pow_ui", t, dtype="float64", n=3))(xr)
    got_root_ui = bound_root_ui(xr)
    ref_root_ui = jax.vmap(lambda t: api.eval_point("arb_root_ui", t, dtype="float64", k=3))(xr)
    got_sin_cos = bound_sin_cos(xr)
    ref_sin_cos = jax.vmap(lambda t: api.eval_point("arb_sin_cos", t, dtype="float64"))(xr)
    got_sinh_cosh = bound_sinh_cosh(xr)
    ref_sinh_cosh = jax.vmap(lambda t: api.eval_point("arb_sinh_cosh", t, dtype="float64"))(xr)
    got_complex_pow_ui = bound_complex_pow_ui(xc)
    ref_complex_pow_ui = jax.vmap(lambda z: api.eval_point("acb_pow_ui", z, n=3))(xc)
    got_complex_sin_cos = bound_complex_sin_cos(xc)
    ref_complex_sin_cos = jax.vmap(lambda z: api.eval_point("acb_sin_cos", z))(xc)

    assert jnp.allclose(got_fma, ref_fma)
    assert jnp.allclose(got_pow_ui, ref_pow_ui)
    assert jnp.allclose(got_root_ui, ref_root_ui)
    assert isinstance(got_sin_cos, tuple) and isinstance(ref_sin_cos, tuple)
    assert jnp.allclose(got_sin_cos[0], ref_sin_cos[0])
    assert jnp.allclose(got_sin_cos[1], ref_sin_cos[1])
    assert isinstance(got_sinh_cosh, tuple) and isinstance(ref_sinh_cosh, tuple)
    assert jnp.allclose(got_sinh_cosh[0], ref_sinh_cosh[0])
    assert jnp.allclose(got_sinh_cosh[1], ref_sinh_cosh[1])
    assert jnp.allclose(got_complex_pow_ui, ref_complex_pow_ui)
    assert isinstance(got_complex_sin_cos, tuple) and isinstance(ref_complex_sin_cos, tuple)
    assert jnp.allclose(got_complex_sin_cos[0], ref_complex_sin_cos[0])
    assert jnp.allclose(got_complex_sin_cos[1], ref_complex_sin_cos[1])


def test_complex_parameterized_core_fastpaths_have_compiled_batch_parity() -> None:
    z = jnp.asarray(
        [0.3 + 0.05j, 0.4 + 0.1j, 0.5 + 0.15j, 0.6 + 0.2j, 0.7 + 0.05j, 0.8 + 0.1j, 0.9 + 0.15j, 1.0 + 0.2j],
        dtype=jnp.complex128,
    )
    s = jnp.asarray(
        [2.0 + 0.0j, 2.1 + 0.05j, 2.2 + 0.1j, 2.3 + 0.15j, 2.4 + 0.0j, 2.5 + 0.05j, 2.6 + 0.1j, 2.7 + 0.15j],
        dtype=jnp.complex128,
    )
    a = jnp.asarray(
        [0.5 + 0.1j, 0.55 + 0.12j, 0.6 + 0.14j, 0.65 + 0.16j, 0.7 + 0.1j, 0.75 + 0.12j, 0.8 + 0.14j, 0.85 + 0.16j],
        dtype=jnp.complex128,
    )
    x = jnp.asarray(
        [1.0 + 0.1j, 1.1 + 0.15j, 1.2 + 0.2j, 1.3 + 0.25j, 1.4 + 0.1j, 1.5 + 0.15j, 1.6 + 0.2j, 1.7 + 0.25j],
        dtype=jnp.complex128,
    )
    y = jnp.asarray(
        [2.0 + 0.2j, 2.1 + 0.25j, 2.2 + 0.3j, 2.3 + 0.35j, 2.4 + 0.2j, 2.5 + 0.25j, 2.6 + 0.3j, 2.7 + 0.35j],
        dtype=jnp.complex128,
    )

    bound_addmul = api.bind_point_batch_jit("acb_addmul", pad_to=8)
    bound_hurwitz = api.bind_point_batch_jit("acb_hurwitz_zeta", pad_to=8)
    bound_bernoulli = api.bind_point_batch_jit("acb_bernoulli_poly_ui", pad_to=8, n=3)
    bound_polylog = api.bind_point_batch_jit("acb_polylog_si", pad_to=8, s=2)

    got_addmul = bound_addmul(x, y, z)
    ref_addmul = jax.vmap(lambda a0, b0, c0: api.eval_point("acb_addmul", a0, b0, c0))(x, y, z)
    got_hurwitz = bound_hurwitz(s, a)
    ref_hurwitz = jax.vmap(lambda s0, a0: api.eval_point("acb_hurwitz_zeta", s0, a0))(s, a)
    got_bernoulli = bound_bernoulli(z)
    ref_bernoulli = jax.vmap(lambda z0: api.eval_point("acb_bernoulli_poly_ui", z0, n=3))(z)
    got_polylog = bound_polylog(z)
    ref_polylog = jax.vmap(lambda z0: api.eval_point("acb_polylog_si", z0, s=2))(z)

    assert jnp.allclose(got_addmul, ref_addmul)
    assert jnp.allclose(got_hurwitz, ref_hurwitz)
    assert jnp.allclose(got_bernoulli, ref_bernoulli)
    assert jnp.allclose(got_polylog, ref_polylog, equal_nan=True)


def test_complex_analytic_fastpaths_have_compiled_batch_parity() -> None:
    x = jnp.asarray(
        [1.2 + 0.1j, 1.25 + 0.12j, 1.3 + 0.14j, 1.35 + 0.16j, 1.4 + 0.1j, 1.45 + 0.12j, 1.5 + 0.14j, 1.55 + 0.16j],
        dtype=jnp.complex128,
    )
    y = jnp.asarray(
        [0.6 + 0.2j, 0.65 + 0.18j, 0.7 + 0.16j, 0.75 + 0.14j, 0.8 + 0.2j, 0.85 + 0.18j, 0.9 + 0.16j, 0.95 + 0.14j],
        dtype=jnp.complex128,
    )
    z = jnp.asarray(
        [2.0 + 0.1j, 2.1 + 0.12j, 2.2 + 0.14j, 2.3 + 0.16j, 2.4 + 0.1j, 2.5 + 0.12j, 2.6 + 0.14j, 2.7 + 0.16j],
        dtype=jnp.complex128,
    )
    w = jnp.asarray(
        [0.3 + 0.1j, 0.35 + 0.08j, 0.4 + 0.06j, 0.45 + 0.04j, 0.5 + 0.1j, 0.55 + 0.08j, 0.6 + 0.06j, 0.65 + 0.04j],
        dtype=jnp.complex128,
    )

    bound_agm = api.bind_point_batch_jit("acb_agm", pad_to=8)
    bound_agm1 = api.bind_point_batch_jit("acb_agm1", pad_to=8)
    bound_agm1_cpx = api.bind_point_batch_jit("acb_agm1_cpx", pad_to=8)
    bound_polylog = api.bind_point_batch_jit("acb_polylog", pad_to=8)
    bound_zeta = api.bind_point_batch_jit("acb_zeta", pad_to=8)
    got_agm = bound_agm(x, y)
    ref_agm = jax.vmap(lambda a0, b0: api.eval_point("acb_agm", a0, b0))(x, y)
    got_agm1 = bound_agm1(y)
    ref_agm1 = jax.vmap(lambda y0: api.eval_point("acb_agm1", y0))(y)
    got_agm1_cpx = bound_agm1_cpx(y)
    ref_agm1_cpx = jax.vmap(lambda y0: api.eval_point("acb_agm1_cpx", y0))(y)
    got_polylog = bound_polylog(z, w)
    ref_polylog = jax.vmap(lambda s0, z0: api.eval_point("acb_polylog", s0, z0))(z, w)
    got_zeta = bound_zeta(z)
    ref_zeta = jax.vmap(lambda z0: api.eval_point("acb_zeta", z0))(z)

    assert jnp.allclose(got_agm, ref_agm)
    assert jnp.allclose(got_agm1, ref_agm1)
    assert jnp.allclose(got_agm1_cpx, ref_agm1_cpx)
    assert jnp.allclose(got_polylog, ref_polylog, equal_nan=True)
    assert jnp.allclose(got_zeta, ref_zeta)


def test_complex_batch_helper_and_barnes_fastpaths_have_compiled_batch_parity() -> None:
    s = jnp.asarray(
        [0.5 + 0.1j, 0.55 + 0.12j, 0.6 + 0.14j, 0.65 + 0.16j, 0.7 + 0.1j, 0.75 + 0.12j, 0.8 + 0.14j, 0.85 + 0.16j],
        dtype=jnp.complex128,
    )
    m = jnp.asarray(
        [0.2 + 0.1j, 0.25 + 0.12j, 0.3 + 0.14j, 0.35 + 0.16j, 0.4 + 0.1j, 0.45 + 0.12j, 0.5 + 0.14j, 0.55 + 0.16j],
        dtype=jnp.complex128,
    )
    z = jnp.asarray(
        [1.2 + 0.1j, 1.25 + 0.12j, 1.3 + 0.14j, 1.35 + 0.16j, 1.4 + 0.1j, 1.45 + 0.12j, 1.5 + 0.14j, 1.55 + 0.16j],
        dtype=jnp.complex128,
    )

    bound_zeta_helper = api.bind_point_batch_jit("acb_dirichlet_zeta_batch_fixed", pad_to=8)
    bound_elliptic_helper = api.bind_point_batch_jit("acb_elliptic_k_batch_fixed", pad_to=8)
    bound_barnes = api.bind_point_batch_jit("acb_barnes_g", pad_to=8)
    bound_log_barnes = api.bind_point_batch_jit("acb_log_barnes_g", pad_to=8)

    got_zeta_helper = bound_zeta_helper(s)
    ref_zeta_helper = api.eval_point("acb_dirichlet_zeta_batch_fixed", s)
    got_elliptic_helper = bound_elliptic_helper(m)
    ref_elliptic_helper = api.eval_point("acb_elliptic_k_batch_fixed", m)
    got_barnes = bound_barnes(z)
    ref_barnes = jax.vmap(lambda z0: api.eval_point("acb_barnes_g", z0))(z)
    got_log_barnes = bound_log_barnes(z)
    ref_log_barnes = jax.vmap(lambda z0: api.eval_point("acb_log_barnes_g", z0))(z)

    assert jnp.allclose(got_zeta_helper, ref_zeta_helper)
    assert jnp.allclose(got_elliptic_helper, ref_elliptic_helper)
    assert jnp.allclose(got_barnes, ref_barnes)
    assert jnp.allclose(got_log_barnes, ref_log_barnes)


def test_interval_precision_point_fast_jax_surface_has_compiled_batch_and_basic_containment() -> None:
    x = jnp.linspace(0.2, 1.1, 8, dtype=jnp.float64)
    bound = api.bind_point_batch_jit("arb_exp", dtype="float64", pad_to=8)

    got = bound(x)
    expected = jax.vmap(lambda t: api.eval_point("arb_exp", t, dtype="float64"))(x)
    enclosure = api.eval_interval_batch(
        "arb_exp",
        _real_interval_batch(x),
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
    )

    assert got.shape == x.shape
    assert jnp.allclose(got, expected)
    assert jnp.all(di.contains(enclosure, _real_interval_batch(got)))


def test_dense_matrix_point_fast_jax_surface_has_compiled_batch_and_vmap_parity() -> None:
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
            [[5.0, 0.75], [0.75, 4.0]],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray(
        [
            [1.0, 2.0],
            [1.5, -0.5],
            [0.75, 0.25],
            [0.25, 1.25],
        ],
        dtype=jnp.float64,
    )
    dense_interval = _real_interval_batch(dense)
    rhs_interval = _real_interval_batch(rhs)
    bound = api.bind_point_batch_jit("arb_mat_matvec", dtype="float64", pad_to=4)

    got = bound(dense_interval, rhs_interval)
    expected = jax.vmap(lambda a, x: api.eval_point("arb_mat_matvec", a, x, dtype="float64"))(dense_interval, rhs_interval)

    assert got.shape == rhs.shape
    assert jnp.allclose(got, expected)


def test_dense_matrix_alias_fastpaths_have_compiled_batch_parity() -> None:
    left = _real_interval_batch(
        jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.5, -0.5], [0.25, 0.75]],
                [[0.5, 1.0], [1.5, 2.0]],
            ],
            dtype=jnp.float64,
        )
    )
    right = _real_interval_batch(
        jnp.asarray(
            [
                [[0.25, 0.5], [0.75, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[2.0, 1.0], [0.0, 1.0]],
            ],
            dtype=jnp.float64,
        )
    )
    bound = api.bind_point_batch_jit("arb_mat_add", dtype="float64", pad_to=4)

    got = bound(left, right)
    expected = jax.vmap(lambda a, b: api.eval_point("arb_mat_add", a, b, dtype="float64"))(left, right)

    assert jnp.allclose(got, expected)


def test_sparse_point_fast_jax_surface_has_compiled_cached_batch_and_vmap_parity() -> None:
    dense = jnp.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    sparse = srb_mat.srb_mat_from_dense_csr(dense)
    plan = api.eval_point("srb_mat_matvec_cached_prepare", sparse)
    rhs = jnp.stack(
        [
            jnp.array([1.0, -1.0, 0.5], dtype=jnp.float64),
            jnp.array([2.0, 0.5, -0.25], dtype=jnp.float64),
            jnp.array([-0.5, 1.5, 1.0], dtype=jnp.float64),
            jnp.array([0.25, -0.75, 0.5], dtype=jnp.float64),
        ],
        axis=0,
    )
    bound = api.bind_point_batch_jit("srb_mat_matvec_cached_apply", dtype="float64", pad_to=4)

    got = bound(plan, rhs)
    expected = jax.vmap(lambda x: api.eval_point("srb_mat_matvec_cached_apply", plan, x, dtype="float64"))(rhs)

    assert got.shape == rhs.shape
    assert jnp.allclose(got, expected)


def test_matrix_free_point_fast_jax_surface_has_compiled_operator_apply_and_logdet_parity() -> None:
    mat = jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64))
    a = di.interval(mat, mat)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    rhs = di.interval(
        jnp.asarray(
            [
                [1.0, 0.5, -0.25, 0.75],
                [0.5, -1.0, 0.25, 0.0],
                [-0.25, 0.75, 1.0, -0.5],
                [1.25, 0.0, -0.5, 0.5],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [
                [1.0, 0.5, -0.25, 0.75],
                [0.5, -1.0, 0.25, 0.0],
                [-0.25, 0.75, 1.0, -0.5],
                [1.25, 0.0, -0.5, 0.5],
            ],
            dtype=jnp.float64,
        ),
    )
    compiled_apply = jax.jit(lambda batch: jax.vmap(lambda x: jrb_mat.jrb_mat_operator_plan_apply(plan, x))(batch))

    got = compiled_apply(rhs)
    expected = jax.vmap(lambda x: jrb_mat.jrb_mat_operator_plan_apply(plan, x))(rhs)
    probes = jnp.stack([rhs[0], rhs[1]], axis=0)
    logdet_jit = jrb_mat.jrb_mat_logdet_slq_point_jit(plan, probes, 2)
    logdet_ref = jrb_mat.jrb_mat_logdet_slq_point(plan, probes, 2)

    assert got.shape == rhs.shape
    assert jnp.allclose(got, expected)
    assert jnp.allclose(logdet_jit, logdet_ref, rtol=1e-6, atol=1e-6)


def test_special_function_point_fast_jax_surface_has_compiled_batch_and_basic_containment() -> None:
    s = jnp.asarray([1.5, 2.0, 2.5, 3.0], dtype=jnp.float64)
    z = jnp.asarray([0.5, 0.8, 1.1, 1.4], dtype=jnp.float64)
    bound = api.bind_point_batch_jit(
        "incomplete_gamma_upper",
        dtype="float64",
        pad_to=4,
        method="quadrature",
        regularized=True,
    )

    got = bound(s, z)
    expected = jax.vmap(
        lambda s_i, z_i: api.eval_point(
            "incomplete_gamma_upper",
            s_i,
            z_i,
            dtype="float64",
            method="quadrature",
            regularized=True,
        )
    )(s, z)
    assert got.shape == s.shape
    assert jnp.allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(bound(s, z), got, rtol=1e-6, atol=1e-6)


def test_special_function_direct_fastpaths_cover_bessel_and_gamma_point_batches() -> None:
    nu = jnp.asarray([0.25, 0.5, 0.75, 1.0], dtype=jnp.float64)
    zr = jnp.asarray([0.5, 0.8, 1.1, 1.4], dtype=jnp.float64)
    xr = jnp.asarray([0.8, 1.1, 1.4, 1.7], dtype=jnp.float64)

    bessel_batch = api.bind_point_batch_jit("arb_bessel_k", dtype="float64", pad_to=4)
    gamma_batch = api.bind_point_batch_jit("arb_gamma", dtype="float64", pad_to=4)

    got_bessel = bessel_batch(nu, zr)
    ref_bessel = jax.vmap(lambda n, z: api.eval_point("arb_bessel_k", n, z, dtype="float64"))(nu, zr)
    got_gamma = gamma_batch(xr)
    ref_gamma = jax.vmap(lambda x: api.eval_point("arb_gamma", x, dtype="float64"))(xr)

    assert jnp.allclose(got_bessel, ref_bessel, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(got_gamma, ref_gamma, rtol=1e-6, atol=1e-6)


def test_point_api_jit_and_batch_paths_support_static_method_kwargs() -> None:
    s = jnp.asarray([1.5, 2.0, 2.5, 3.0], dtype=jnp.float64)
    z = jnp.asarray([0.5, 0.8, 1.1, 1.4], dtype=jnp.float64)

    single = api.eval_point(
        "incomplete_gamma_upper",
        jnp.asarray(2.5, dtype=jnp.float64),
        jnp.asarray(0.8, dtype=jnp.float64),
        jit=True,
        dtype="float64",
        method="quadrature",
        regularized=True,
    )
    batch = api.eval_point_batch(
        "incomplete_gamma_upper",
        s,
        z,
        dtype="float64",
        pad_to=4,
        method="quadrature",
        regularized=True,
    )
    bound = api.bind_point_batch(
        "incomplete_gamma_upper",
        dtype="float64",
        pad_to=4,
        method="quadrature",
        regularized=True,
    )

    assert jnp.isfinite(single)
    assert batch.shape == s.shape
    assert jnp.allclose(bound(s, z), batch, rtol=1e-6, atol=1e-6)
