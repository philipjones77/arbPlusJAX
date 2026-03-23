import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import point_wrappers


def _real_interval(x):
    return di.interval(x, x)


def _complex_box(z):
    return acb_core.acb_box(
        di.interval(jnp.real(z), jnp.real(z)),
        di.interval(jnp.imag(z), jnp.imag(z)),
    )


def test_real_point_wrappers_broadcast_and_tuple_outputs():
    x = jnp.asarray([[0.2], [0.5]], dtype=jnp.float64)
    y = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64)

    added = point_wrappers.arb_add_point(x, y)
    s, c = point_wrappers.arb_sin_cos_point(added)

    assert added.shape == (2, 3)
    assert s.shape == (2, 3)
    assert c.shape == (2, 3)
    assert jnp.allclose(added, x + y)
    assert jnp.allclose(s, jnp.sin(x + y))
    assert jnp.allclose(c, jnp.cos(x + y))


def test_complex_point_wrappers_broadcast_and_tuple_outputs():
    x = jnp.asarray([[0.2 + 0.1j], [0.5 - 0.2j]], dtype=jnp.complex128)
    y = jnp.asarray([[1.0 - 0.3j, 2.0 + 0.25j, 3.0 - 0.1j]], dtype=jnp.complex128)

    added = point_wrappers.acb_add_point(x, y)
    s, c = point_wrappers.acb_sin_cos_point(added)

    assert added.shape == (2, 3)
    assert s.shape == (2, 3)
    assert c.shape == (2, 3)
    assert jnp.allclose(added, x + y)
    assert jnp.allclose(s, jnp.sin(x + y))
    assert jnp.allclose(c, jnp.cos(x + y))


def test_real_matrix_point_wrapper_batch_padding_and_plan_paths():
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray(
        [
            [1.0, 2.0],
            [1.5, -0.5],
            [0.75, 0.25],
        ],
        dtype=jnp.float64,
    )
    dense_interval = _real_interval(dense)
    rhs_interval = _real_interval(rhs)

    fixed = point_wrappers.arb_mat_matvec_batch_fixed_point(dense_interval, rhs_interval)
    padded = point_wrappers.arb_mat_matvec_batch_padded_point(dense_interval, rhs_interval, pad_to=4)
    cache = point_wrappers.arb_mat_matvec_cached_prepare_point(dense_interval)
    cached = point_wrappers.arb_mat_matvec_cached_apply_batch_fixed_point(cache, rhs_interval)

    assert fixed.shape == (3, 2)
    assert padded.shape == (4, 2)
    assert jnp.allclose(padded[:3], fixed)
    assert jnp.allclose(padded[3], fixed[-1])
    assert jnp.allclose(cached, fixed)

    lu_plan = point_wrappers.arb_mat_dense_lu_solve_plan_prepare_point(dense_interval[0])
    solved_direct = point_wrappers.arb_mat_solve_point(dense_interval[0], rhs_interval[0])
    solved_plan = point_wrappers.arb_mat_dense_lu_solve_plan_apply_point(lu_plan, rhs_interval[0])

    assert lu_plan.rows == 2
    assert lu_plan.algebra == "arb"
    assert jnp.allclose(solved_plan, solved_direct)


def test_complex_matrix_point_wrapper_batch_padding_and_plan_paths():
    dense = jnp.asarray(
        [
            [[3.0 + 0.0j, 0.5 + 0.2j], [0.5 - 0.2j, 2.0 + 0.0j]],
            [[4.0 + 0.0j, -1.0 + 0.1j], [-1.0 - 0.1j, 3.5 + 0.0j]],
            [[2.5 + 0.0j, 0.25 - 0.15j], [0.25 + 0.15j, 2.25 + 0.0j]],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.asarray(
        [
            [1.0 + 0.25j, 2.0 - 0.1j],
            [1.5 - 0.5j, -0.5 + 0.2j],
            [0.75 + 0.4j, 0.25 - 0.3j],
        ],
        dtype=jnp.complex128,
    )
    dense_box = _complex_box(dense)
    rhs_box = _complex_box(rhs)

    fixed = point_wrappers.acb_mat_matvec_batch_fixed_point(dense_box, rhs_box)
    padded = point_wrappers.acb_mat_matvec_batch_padded_point(dense_box, rhs_box, pad_to=4)
    cache = point_wrappers.acb_mat_matvec_cached_prepare_point(dense_box)
    cached = point_wrappers.acb_mat_matvec_cached_apply_batch_fixed_point(cache, rhs_box)

    assert fixed.shape == (3, 2)
    assert padded.shape == (4, 2)
    assert jnp.allclose(padded[:3], fixed)
    assert jnp.allclose(padded[3], fixed[-1])
    assert jnp.allclose(cached, fixed)

    hpd_plan = point_wrappers.acb_mat_dense_hpd_solve_plan_prepare_point(dense_box[0])
    solved_direct = point_wrappers.acb_mat_hpd_solve_point(dense_box[0], rhs_box[0])
    solved_plan = point_wrappers.acb_mat_dense_hpd_solve_plan_apply_point(hpd_plan, rhs_box[0])

    assert hpd_plan.rows == 2
    assert hpd_plan.algebra == "acb"
    assert hpd_plan.structure == "hermitian"
    assert jnp.allclose(solved_plan, solved_direct)


def test_public_api_point_batch_surfaces_match_point_wrapper_fastpaths():
    dense = jnp.asarray(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[4.0, -1.0], [-1.0, 3.5]],
            [[2.5, 0.25], [0.25, 2.25]],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray(
        [
            [1.0 + 0.25j, 2.0 - 0.1j],
            [1.5 - 0.5j, -0.5 + 0.2j],
            [0.75 + 0.4j, 0.25 - 0.3j],
        ],
        dtype=jnp.complex128,
    )
    dense_interval = _real_interval(dense)
    rhs_box = _complex_box(rhs)
    dense_box = _complex_box(
        jnp.asarray(
            [
                [[3.0 + 0.0j, 0.5 + 0.2j], [0.5 - 0.2j, 2.0 + 0.0j]],
                [[4.0 + 0.0j, -1.0 + 0.1j], [-1.0 - 0.1j, 3.5 + 0.0j]],
                [[2.5 + 0.0j, 0.25 - 0.15j], [0.25 + 0.15j, 2.25 + 0.0j]],
            ],
            dtype=jnp.complex128,
        )
    )

    det_api = api.eval_point_batch("arb_mat_det", dense_interval, dtype="float64", pad_to=4)
    det_direct = point_wrappers.arb_mat_det_batch_fixed_point(dense_interval)
    matvec_bound = api.bind_point_batch("acb_mat_matvec", dtype="float64", pad_to=4)
    matvec_api = matvec_bound(dense_box, rhs_box)
    matvec_direct = point_wrappers.acb_mat_matvec_batch_fixed_point(dense_box, rhs_box)

    metadata = {entry.name: entry for entry in api.list_public_function_metadata(family="matrix")}
    arb_det = metadata["arb_mat_det"]
    acb_matvec = metadata["acb_mat_matvec"]

    assert jnp.allclose(det_api, det_direct)
    assert jnp.allclose(matvec_api, matvec_direct)
    assert arb_det.point_support is True
    assert arb_det.module == "point_wrappers"
    assert arb_det.qualified_name == "point_wrappers.arb_mat_det_point"
    assert acb_matvec.point_support is True
    assert acb_matvec.family == "matrix"
    assert acb_matvec.qualified_name == "point_wrappers.acb_mat_matvec_point"
