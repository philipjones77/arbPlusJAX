from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import hypgeom


def _complex_box_exact(x: jnp.ndarray) -> jnp.ndarray:
    return acb_core.acb_box(
        di.interval(jnp.real(x), jnp.real(x)),
        di.interval(jnp.imag(x), jnp.imag(x)),
    )


def _real_family_midpoint(name: str, args: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    fn = getattr(hypgeom, name.removeprefix("hypgeom."))
    def _box_exact(x: jnp.ndarray) -> jnp.ndarray:
        return di.interval(x, x)
    if name.endswith("_pfq"):
        return jnp.asarray(
            [
                di.midpoint(
                    fn(
                        _box_exact(a_row),
                        _box_exact(b_row),
                        _box_exact(z_val),
                    )
                )
                for a_row, b_row, z_val in zip(args[0], args[1], args[2], strict=True)
            ],
            dtype=jnp.float64,
        )
    if len(args) == 2:
        return jnp.asarray([di.midpoint(fn(_box_exact(a_val), _box_exact(z_val))) for a_val, z_val in zip(args[0], args[1], strict=True)], dtype=jnp.float64)
    if len(args) == 3:
        return jnp.asarray(
            [di.midpoint(fn(_box_exact(a_val), _box_exact(b_val), _box_exact(z_val))) for a_val, b_val, z_val in zip(args[0], args[1], args[2], strict=True)],
            dtype=jnp.float64,
        )
    return jnp.asarray(
        [
            di.midpoint(fn(_box_exact(a_val), _box_exact(b_val), _box_exact(c_val), _box_exact(z_val)))
            for a_val, b_val, c_val, z_val in zip(args[0], args[1], args[2], args[3], strict=True)
        ],
        dtype=jnp.float64,
    )


def _complex_family_midpoint(name: str, args: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    fn = getattr(hypgeom, name.removeprefix("hypgeom."))
    if name.endswith("_pfq"):
        return jnp.asarray(
            [
                acb_core.acb_midpoint(fn(_complex_box_exact(a_row), _complex_box_exact(b_row), _complex_box_exact(z_val)))
                for a_row, b_row, z_val in zip(args[0], args[1], args[2], strict=True)
            ],
            dtype=jnp.complex128,
        )
    if len(args) == 2:
        return jnp.asarray(
            [acb_core.acb_midpoint(fn(_complex_box_exact(a_val), _complex_box_exact(z_val))) for a_val, z_val in zip(args[0], args[1], strict=True)],
            dtype=jnp.complex128,
        )
    if len(args) == 3:
        return jnp.asarray(
            [
                acb_core.acb_midpoint(fn(_complex_box_exact(a_val), _complex_box_exact(b_val), _complex_box_exact(z_val)))
                for a_val, b_val, z_val in zip(args[0], args[1], args[2], strict=True)
            ],
            dtype=jnp.complex128,
        )
    return jnp.asarray(
        [
            acb_core.acb_midpoint(fn(_complex_box_exact(a_val), _complex_box_exact(b_val), _complex_box_exact(c_val), _complex_box_exact(z_val)))
            for a_val, b_val, c_val, z_val in zip(args[0], args[1], args[2], args[3], strict=True)
        ],
        dtype=jnp.complex128,
    )


def test_real_hypgeom_point_surfaces_match_family_owned_exact_input_midpoints() -> None:
    onef1_a = jnp.asarray([1.1, 1.2, 1.3], dtype=jnp.float64)
    onef1_b = jnp.asarray([2.1, 2.2, 2.3], dtype=jnp.float64)
    onef1_z = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)

    twof1_c = jnp.asarray([2.8, 2.9, 3.0], dtype=jnp.float64)
    u_z = jnp.asarray([0.6, 0.8, 1.0], dtype=jnp.float64)
    pfq_a = jnp.asarray([[0.6, 0.9], [0.7, 1.0], [0.8, 1.1]], dtype=jnp.float64)
    pfq_b = jnp.asarray([[1.4], [1.5], [1.6]], dtype=jnp.float64)

    cases = [
        ("hypgeom.arb_hypgeom_0f1", (onef1_b, onef1_z)),
        ("hypgeom.arb_hypgeom_1f1", (onef1_a, onef1_b, onef1_z)),
        ("hypgeom.arb_hypgeom_2f1", (onef1_a, onef1_b, twof1_c, onef1_z)),
        ("hypgeom.arb_hypgeom_u", (onef1_a, onef1_b, u_z)),
        ("hypgeom.arb_hypgeom_pfq", (pfq_a, pfq_b, onef1_z)),
    ]

    for name, args in cases:
        point = api.bind_point_batch_jit(name, dtype="float64", pad_to=8)(*args)
        family = _real_family_midpoint(name, args)
        assert jnp.allclose(point, family, rtol=1e-10, atol=1e-10), name


def test_complex_hypgeom_point_surfaces_match_family_owned_exact_input_midpoints() -> None:
    onef1_a = jnp.asarray([1.1 + 0.0j, 1.2 + 0.0j, 1.3 + 0.0j], dtype=jnp.complex128)
    onef1_b = jnp.asarray([2.1 + 0.0j, 2.2 + 0.0j, 2.3 + 0.0j], dtype=jnp.complex128)
    onef1_z = jnp.asarray([0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j], dtype=jnp.complex128)

    twof1_c = jnp.asarray([2.8 + 0.0j, 2.9 + 0.0j, 3.0 + 0.0j], dtype=jnp.complex128)
    u_z = jnp.asarray([0.6 + 0.0j, 0.8 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)
    pfq_a = jnp.asarray([[0.6 + 0.0j, 0.9 + 0.0j], [0.7 + 0.0j, 1.0 + 0.0j], [0.8 + 0.0j, 1.1 + 0.0j]], dtype=jnp.complex128)
    pfq_b = jnp.asarray([[1.4 + 0.0j], [1.5 + 0.0j], [1.6 + 0.0j]], dtype=jnp.complex128)

    cases = [
        ("hypgeom.acb_hypgeom_0f1", (onef1_b, onef1_z)),
        ("hypgeom.acb_hypgeom_1f1", (onef1_a, onef1_b, onef1_z)),
        ("hypgeom.acb_hypgeom_2f1", (onef1_a, onef1_b, twof1_c, onef1_z)),
        ("hypgeom.acb_hypgeom_u", (onef1_a, onef1_b, u_z)),
        ("hypgeom.acb_hypgeom_pfq", (pfq_a, pfq_b, onef1_z)),
    ]

    for name, args in cases:
        point = api.bind_point_batch_jit(name, pad_to=8)(*args)
        family = _complex_family_midpoint(name, args)
        assert jnp.allclose(point, family, rtol=1e-10, atol=1e-10), name
