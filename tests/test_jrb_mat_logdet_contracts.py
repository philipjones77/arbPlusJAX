import json

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse

from arbplusjax import double_interval as di
from arbplusjax import jrb_mat

from tests._test_checks import _check


def _point_interval(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return di.interval(xx, xx)


def _diag_operator(diag: jax.Array):
    dense = jnp.diag(jnp.asarray(diag, dtype=diag.dtype))
    return jrb_mat.jrb_mat_dense_operator(di.interval(dense, dense))


def _dense_operator(a: jax.Array):
    aa = jnp.asarray(a)
    return jrb_mat.jrb_mat_dense_operator(di.interval(aa, aa))


def _rademacher_probes(key: jax.Array, num_probes: int, n: int, dtype=jnp.float64) -> jax.Array:
    mids = random.rademacher(key, (num_probes, n), dtype=dtype)
    return jax.vmap(_point_interval)(mids)


def _make_spd(key: jax.Array, n: int, cond: float = 1e2, dtype=jnp.float64) -> jax.Array:
    q, _ = jnp.linalg.qr(random.normal(key, (n, n), dtype=dtype))
    eigs = jnp.exp(jnp.linspace(jnp.log(jnp.asarray(1.0 / cond, dtype=dtype)), 0.0, n, dtype=dtype))
    return (q.T * eigs) @ q


def _slq_logdet(mat: jax.Array, key: jax.Array, num_probes: int, steps: int, dtype=jnp.float64) -> jax.Array:
    a = jnp.asarray(mat, dtype=dtype)
    op = _dense_operator(a)
    probes = _rademacher_probes(key, num_probes, a.shape[0], dtype=dtype)
    return jrb_mat.jrb_mat_logdet_slq_point(op, probes, steps)


def test_slq_logdet_small_diagonal_exactness_contract():
    for n in (4, 8):
        diag = jnp.exp(jnp.linspace(jnp.log(0.01), 0.0, n, dtype=jnp.float64))
        exact = jnp.sum(jnp.log(diag))
        est = jrb_mat.jrb_mat_logdet_slq_point(
            _diag_operator(diag),
            _rademacher_probes(random.PRNGKey(n), 16, n),
            n,
        )
        _check(bool(jnp.allclose(est, exact, rtol=1e-12, atol=1e-12)))


def test_slq_logdet_tail_sensitivity_contract():
    diag = jnp.asarray([0.2, 0.4, 0.8, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=jnp.float64)
    base = jrb_mat.jrb_mat_logdet_slq_point(
        _diag_operator(diag),
        _rademacher_probes(random.PRNGKey(0), 16, diag.shape[0]),
        diag.shape[0],
    )
    top_idx = jnp.asarray([6, 7], dtype=jnp.int32)
    for factor in (0.5, 2.0):
        scaled = diag.at[top_idx].multiply(factor)
        est = jrb_mat.jrb_mat_logdet_slq_point(
            _diag_operator(scaled),
            _rademacher_probes(random.PRNGKey(0), 16, diag.shape[0]),
            diag.shape[0],
        )
        expected_delta = jnp.asarray(top_idx.shape[0], dtype=jnp.float64) * jnp.log(jnp.asarray(factor, dtype=jnp.float64))
        _check(bool(jnp.allclose(est - base, expected_delta, rtol=1e-12, atol=1e-12)))


def test_slq_logdet_budget_contract_variance_drops_with_more_probes():
    a = _make_spd(random.PRNGKey(11), 8, cond=1e2)
    op = _dense_operator(a)

    def sample_estimates(num_probes: int) -> jax.Array:
        vals = []
        for seed in range(8):
            probes = _rademacher_probes(random.PRNGKey(seed), num_probes, 8)
            vals.append(jrb_mat.jrb_mat_logdet_slq_point(op, probes, 8))
        return jnp.asarray(vals, dtype=jnp.float64)

    small = sample_estimates(4)
    large = sample_estimates(32)
    _check(bool(jnp.std(large) < jnp.std(small)))


def test_slq_logdet_reproducibility_and_dtype_stability_contract():
    diag64 = jnp.asarray([0.3, 0.75, 1.0, 1.8, 3.0, 5.0], dtype=jnp.float64)
    key = random.PRNGKey(123)
    probes64 = _rademacher_probes(key, 16, diag64.shape[0], dtype=jnp.float64)
    op64 = _diag_operator(diag64)
    out_a = jrb_mat.jrb_mat_logdet_slq_point(op64, probes64, diag64.shape[0])
    out_b = jrb_mat.jrb_mat_logdet_slq_point(op64, probes64, diag64.shape[0])
    _check(bool(jnp.allclose(out_a, out_b, rtol=0.0, atol=0.0)))
    _check(bool(jnp.isfinite(out_a)))

    diag32 = diag64.astype(jnp.float32)
    probes32 = _rademacher_probes(key, 16, diag32.shape[0], dtype=jnp.float32)
    op32 = _diag_operator(diag32)
    out32 = jrb_mat.jrb_mat_logdet_slq_point(op32, probes32, diag32.shape[0])
    _check(bool(jnp.isfinite(out32)))
    _check(bool(jnp.allclose(out32.astype(jnp.float64), out_a, rtol=1e-5, atol=1e-5)))


def test_leja_hutchpp_sparse_diagonal_exactness_contract():
    diag = jnp.asarray([0.4, 0.8, 1.5, 3.0], dtype=jnp.float64)
    dense = jnp.diag(diag)
    bcoo = jsparse.BCOO.fromdense(dense)
    op = jrb_mat.jrb_mat_bcoo_operator(bcoo)
    bounds = jrb_mat.jrb_mat_bcoo_gershgorin_bounds(bcoo)
    sketch = jnp.stack([_point_interval(row) for row in jnp.eye(diag.shape[0], dtype=jnp.float64)], axis=0)
    residual = jnp.zeros((0, diag.shape[0], 2), dtype=jnp.float64)
    est = jrb_mat.jrb_mat_logdet_leja_hutchpp_point(
        op,
        sketch,
        residual,
        degree=16,
        spectral_bounds=bounds,
        candidate_count=96,
    )
    exact = jnp.sum(jnp.log(diag))
    _check(bool(jnp.allclose(est, exact, rtol=1e-6, atol=1e-6)))


def test_leja_hutchpp_sparse_diagonal_auto_bounds_and_adaptive_degree_contract():
    diag = jnp.asarray([0.4, 0.8, 1.5, 3.0], dtype=jnp.float64)
    bcoo = jsparse.BCOO.fromdense(jnp.diag(diag))
    sketch = jnp.stack([_point_interval(row) for row in jnp.eye(diag.shape[0], dtype=jnp.float64)], axis=0)
    residual = jnp.zeros((0, diag.shape[0], 2), dtype=jnp.float64)
    est, diag_info = jrb_mat.jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
        bcoo,
        sketch,
        residual,
        degree=8,
        max_degree=20,
        min_degree=4,
        candidate_count=128,
        bounds_steps=4,
    )
    exact = jnp.sum(jnp.log(diag))
    _check(bool(jnp.allclose(est, exact, rtol=1e-6, atol=1e-6)))
    _check(int(diag_info.steps) >= 4)
    _check(int(diag_info.steps) <= 20)
