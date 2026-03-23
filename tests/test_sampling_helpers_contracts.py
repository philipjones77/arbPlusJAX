from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import sampling_helpers as sh


def test_sampling_helpers_ball_box_conversions_round_trip_reasonably() -> None:
    interval = di.interval(jnp.float64(1.0), jnp.float64(3.0))
    mid, rad = sh.ball_from_interval(interval)
    assert jnp.isclose(mid, 2.0)
    assert jnp.isclose(rad, 1.0)

    box = sh.box_from_ball(jnp.complex128(2.0 + 1.0j), jnp.float64(0.5))
    mid_box, rad_box = sh.ball_from_box(box)
    assert jnp.isclose(mid_box, 2.0 + 1.0j)
    assert jnp.isclose(rad_box, 0.5)


def test_sampling_helpers_adaptive_and_grid_enclosures_cover_sampled_values() -> None:
    x = di.interval(jnp.float64(1.0), jnp.float64(2.0))
    y = di.interval(jnp.float64(-1.0), jnp.float64(1.0))
    real_env = sh.adaptive_real(lambda t: t**2, x, eps=1.0e-6, samples=9)
    bi_env = sh.sample_interval_bivariate_grid(lambda a, b: a + b, x, y, samples=5)

    assert real_env[0] <= 1.0 <= real_env[1]
    assert real_env[0] <= 4.0 <= real_env[1]
    assert bi_env[0] <= 0.0 <= bi_env[1]
    assert bi_env[0] <= 3.0 <= bi_env[1]


def test_sampling_helpers_complex_enclosures_cover_expected_ranges() -> None:
    x = acb_core.acb_box(di.interval(jnp.float64(1.0), jnp.float64(1.0)), di.interval(jnp.float64(0.0), jnp.float64(0.0)))
    y = acb_core.acb_box(di.interval(jnp.float64(0.0), jnp.float64(0.0)), di.interval(jnp.float64(1.0), jnp.float64(1.0)))

    adaptive = sh.adaptive_complex(lambda z: z**2, x, eps=1.0e-6, samples=8)
    candidates = sh.sample_box_bivariate_candidates(lambda a, b: a + b, x, y)

    adaptive_mid, _ = sh.ball_from_box(adaptive)
    candidates_mid, _ = sh.ball_from_box(candidates)
    assert jnp.isclose(adaptive_mid, 1.0 + 0.0j)
    assert jnp.isclose(candidates_mid, 1.0 + 1.0j)
