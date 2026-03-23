import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import transform_common as tc


def test_transform_common_axes_sizes_and_mode_grid_contracts():
    assert tc.is_power_of_two(8) is True
    assert tc.is_power_of_two(6) is False
    assert tc.canonical_axes(3, None, "axes") == (0, 1, 2)
    assert tc.canonical_axes(3, (-1, 1), "axes") == (2, 1)
    assert tc.smooth_good_size(7) == 8

    grid = tc.mode_grid((2, 3))
    assert grid.shape == (6, 2)
    assert jnp.array_equal(grid[0], jnp.asarray([0.0, 0.0], dtype=tc.TRANSFORM_REAL_DTYPE))
    assert jnp.array_equal(grid[-1], jnp.asarray([1.0, 2.0], dtype=tc.TRANSFORM_REAL_DTYPE))


def test_transform_common_unit_points_and_stencils_wrap_and_normalize():
    points = tc.as_unit_points_matrix(jnp.asarray([1.25, -0.25], dtype=jnp.float64), 1, "points")
    assert points.shape == (2, 1)
    assert jnp.all(points >= 0.0)
    assert jnp.all(points < 1.0)

    idx, weights = tc.lanczos_stencil_axis(jnp.asarray([0.1, 0.9], dtype=jnp.float64), 8, 3)
    assert idx.shape == weights.shape
    assert jnp.allclose(jnp.sum(weights, axis=1), 1.0, atol=1e-6)

    stencil_pairs = tc.lanczos_stencils(jnp.asarray([[0.1, 0.2], [0.9, 0.8]], dtype=jnp.float64), (8, 6), 2, "stencils")
    assert len(stencil_pairs) == 2
    assert stencil_pairs[0][0].shape[0] == 2
    assert tc.empty_stencil_arrays(3)[0].shape == (3, 0)


def test_transform_common_complex_and_box_coercions_have_stable_shapes():
    vec = tc.as_complex_vector(jnp.asarray([1.0 + 0.0j, 2.0 + 1.0j]), "vec")
    arr = tc.as_complex_array(jnp.asarray([[1.0, 2.0]]), "arr")
    real = tc.as_real_array(jnp.asarray([1.0, 2.0]), "real")
    box = tc.point_box(jnp.asarray(1.0 + 2.0j))
    box_vec = tc.as_box_vector(jnp.stack([box, box], axis=0), "box_vec")
    box_arr = tc.as_box_array(box, "box_arr")

    assert vec.shape == (2,)
    assert arr.dtype == tc.TRANSFORM_COMPLEX_DTYPE
    assert real.dtype == tc.TRANSFORM_REAL_DTYPE
    assert box.shape == (4,)
    assert box_vec.shape == (2, 4)
    assert box_arr.shape == (4,)


def test_transform_common_box_linear_apply_contains_midpoint_action():
    matrix = jnp.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=tc.TRANSFORM_COMPLEX_DTYPE)
    z0 = tc.point_box(jnp.asarray(1.0 + 0.5j))
    z1 = tc.point_box(jnp.asarray(-0.5 + 0.25j))
    x_box = jnp.stack([z0, z1], axis=0)

    out_box = tc.box_linear_apply(matrix, x_box)
    midpoint = matrix @ acb_core.acb_midpoint(x_box)

    assert out_box.shape == (2, 4)
    assert jnp.all(acb_core.acb_real(out_box)[:, 0] <= jnp.real(midpoint))
    assert jnp.all(acb_core.acb_real(out_box)[:, 1] >= jnp.real(midpoint))
    assert jnp.all(acb_core.acb_imag(out_box)[:, 0] <= jnp.imag(midpoint))
    assert jnp.all(acb_core.acb_imag(out_box)[:, 1] >= jnp.imag(midpoint))
