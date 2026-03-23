import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import point_wrappers


def _real_interval_batch(values: jax.Array) -> jax.Array:
    return di.interval(values, values)


def _complex_box_batch(real: jax.Array, imag: jax.Array) -> jax.Array:
    return acb_core.acb_box(di.interval(real, real), di.interval(imag, imag))


def test_matrix_service_binders_cover_point_and_interval_paths():
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
        ],
        dtype=jnp.float64,
    )
    vec = jnp.asarray(
        [
            [1.0, 2.0],
            [1.5, -0.5],
            [0.75, 0.25],
        ],
        dtype=jnp.float64,
    )
    dense_interval = _real_interval_batch(dense)
    vec_interval = _real_interval_batch(vec)
    complex_dense = _complex_box_batch(dense, 0.25 * dense)
    complex_vec = _complex_box_batch(vec, -0.1 * vec)

    det_point = api.bind_point_batch("arb_mat_det", dtype="float64", pad_to=8)(dense_interval)
    solve_basic = api.bind_interval_batch(
        "arb_mat_solve",
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
    )(dense_interval, vec_interval)
    complex_matvec = api.bind_point_batch("acb_mat_matvec", dtype="float64", pad_to=8)(complex_dense, complex_vec)

    assert det_point.shape == (dense.shape[0],)
    assert solve_basic.shape == vec.shape + (2,)
    assert complex_matvec.shape == vec.shape
    assert jnp.allclose(complex_matvec, point_wrappers.acb_mat_matvec_batch_fixed_point(complex_dense, complex_vec))


def test_matrix_service_chunked_binders_match_nonchunked_api_results():
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
            [[5.0, 0.75], [0.75, 4.0]],
            [[6.0, -0.5], [-0.5, 5.0]],
        ],
        dtype=jnp.float64,
    )
    vec = jnp.asarray(
        [
            [1.0, 2.0],
            [1.5, -0.5],
            [0.75, 0.25],
            [0.25, 1.25],
            [2.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    dense_interval = _real_interval_batch(dense)
    vec_interval = _real_interval_batch(vec)

    det_bound = api.bind_point_batch("arb_mat_det", dtype="float64", pad_to=8, chunk_size=2)
    solve_bound = api.bind_interval_batch(
        "arb_mat_solve",
        mode="basic",
        dtype="float64",
        pad_to=8,
        prec_bits=53,
        chunk_size=2,
    )

    assert jnp.allclose(
        det_bound(dense_interval),
        api.eval_point_batch("arb_mat_det", dense_interval, dtype="float64", pad_to=8),
    )
    assert jnp.allclose(
        solve_bound(dense_interval, vec_interval),
        api.eval_interval_batch(
            "arb_mat_solve",
            dense_interval,
            vec_interval,
            mode="basic",
            dtype="float64",
            pad_to=8,
            prec_bits=53,
        ),
    )


def test_matrix_service_binders_are_safe_for_repeated_calls():
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
            [[5.0, 0.75], [0.75, 4.0]],
        ],
        dtype=jnp.float64,
    )
    dense_interval = _real_interval_batch(dense)
    bound = api.bind_point_batch("arb_mat_det", dtype="float64", pad_to=8)
    expected = api.eval_point_batch("arb_mat_det", dense_interval, dtype="float64", pad_to=8)

    for _ in range(5):
        out = bound(dense_interval)
        assert out.shape == expected.shape
        assert jnp.allclose(out, expected)
