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
