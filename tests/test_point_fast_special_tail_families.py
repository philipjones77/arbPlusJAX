from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from arbplusjax import api


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    (
        (
            "incomplete_gamma_upper",
            (
                jnp.asarray([1.5, 2.0, 2.5, 3.0], dtype=jnp.float64),
                jnp.asarray([0.5, 0.8, 1.1, 1.4], dtype=jnp.float64),
            ),
            {"method": "quadrature", "regularized": True},
        ),
        (
            "incomplete_gamma_lower",
            (
                jnp.asarray([1.5, 2.0, 2.5, 3.0], dtype=jnp.float64),
                jnp.asarray([0.5, 0.8, 1.1, 1.4], dtype=jnp.float64),
            ),
            {"method": "quadrature", "regularized": True},
        ),
        (
            "incomplete_bessel_i",
            (
                jnp.asarray([0.0, 0.5, 1.0, 1.5], dtype=jnp.float64),
                jnp.asarray([0.8, 1.0, 1.2, 1.4], dtype=jnp.float64),
                jnp.asarray([0.7, 0.9, 1.1, 1.3], dtype=jnp.float64),
            ),
            {"method": "quadrature"},
        ),
        (
            "incomplete_bessel_k",
            (
                jnp.asarray([0.25, 0.5, 0.75, 1.0], dtype=jnp.float64),
                jnp.asarray([1.0, 1.2, 1.4, 1.6], dtype=jnp.float64),
                jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64),
            ),
            {"method": "quadrature"},
        ),
        (
            "laplace_bessel_k_tail",
            (
                jnp.asarray([0.5, 0.75, 1.0, 1.25], dtype=jnp.float64),
                jnp.asarray([1.5, 1.7, 1.9, 2.1], dtype=jnp.float64),
                jnp.asarray([0.75, 0.9, 1.05, 1.2], dtype=jnp.float64),
                jnp.asarray([0.4, 0.5, 0.6, 0.7], dtype=jnp.float64),
            ),
            {"method": "quadrature"},
        ),
    ),
)
def test_special_tail_point_fast_families_match_scalar_vmap_and_padding(
    name: str,
    args: tuple[jax.Array, ...],
    kwargs: dict[str, object],
) -> None:
    compiled = api.bind_point_batch_jit(name, dtype="float64", pad_to=8, **kwargs)
    bound = api.bind_point_batch(name, dtype="float64", pad_to=8, **kwargs)

    compiled_out = compiled(*args)
    bound_out = bound(*args)
    direct_out = api.eval_point_batch(name, *args, dtype="float64", pad_to=8, **kwargs)

    scalar_vmap = jax.vmap(lambda *aa: api.eval_point(name, *aa, jit=True, dtype="float64", **kwargs))(*args)

    assert compiled_out.shape == jnp.asarray(args[0]).shape
    assert jnp.allclose(compiled_out, bound_out, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled_out, direct_out, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled_out, scalar_vmap, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    (
        (
            "incomplete_gamma_upper",
            (
                jnp.asarray([1.5, 2.0, 2.5], dtype=jnp.float64),
                jnp.asarray([0.5, 0.8, 1.1], dtype=jnp.float64),
            ),
            {"method": "quadrature", "regularized": True},
        ),
        (
            "incomplete_bessel_k",
            (
                jnp.asarray([0.25, 0.5, 0.75], dtype=jnp.float64),
                jnp.asarray([1.0, 1.2, 1.4], dtype=jnp.float64),
                jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64),
            ),
            {"method": "quadrature"},
        ),
        (
            "laplace_bessel_k_tail",
            (
                jnp.asarray([0.5, 0.75, 1.0], dtype=jnp.float64),
                jnp.asarray([1.5, 1.7, 1.9], dtype=jnp.float64),
                jnp.asarray([0.75, 0.9, 1.05], dtype=jnp.float64),
                jnp.asarray([0.4, 0.5, 0.6], dtype=jnp.float64),
            ),
            {"method": "quadrature"},
        ),
    ),
)
def test_special_tail_point_fast_families_have_direct_batch_registry_entries(
    name: str,
    args: tuple[jax.Array, ...],
    kwargs: dict[str, object],
) -> None:
    assert name in api._DIRECT_POINT_BATCH_FASTPATHS

    direct = api._maybe_direct_point_batch_fastpath(name, args, pad_to=8, extra_kwargs=kwargs)
    generic = api.eval_point_batch(name, *args, dtype="float64", pad_to=8, **kwargs)

    assert direct is not None
    assert jnp.allclose(direct, generic, rtol=1e-6, atol=1e-6)
