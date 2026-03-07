from __future__ import annotations

"""CUDA-lineage alternative Bessel-K helpers.

Provenance:
- classification: alternative
- preferred public family: cuda_besselk
- module lineage: CUDA/CubesselK-inspired alternative Bessel-K implementation
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

import jax
import jax.numpy as jnp

from . import ball_wrappers
from . import checks
from . import double_interval as di
from . import hypgeom
from . import precision

jax.config.update("jax_enable_x64", True)

PROVENANCE = {
    "classification": "alternative",
    "base_name": "besselk",
    "preferred_prefix": "cuda",
    "module_lineage": "CUDA/CubesselK-inspired alternative Bessel-K implementation",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}


def cuda_besselk_available() -> bool:
    return True


@jax.jit
def cuda_besselk_point(nu: jax.Array, z: jax.Array, nbins: int = 128) -> jax.Array:
    del nbins  # kept for API compatibility with the original CuBesselK knob
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)
    checks.check_equal(nu_v.shape, z_v.shape, "cuda_besselk.point.shape")
    return hypgeom._real_bessel_eval_k(nu_v, z_v)


def _pb(dps: int | None, prec_bits: int | None) -> int:
    if prec_bits is not None:
        return int(prec_bits)
    if dps is not None:
        return precision.dps_to_bits(dps)
    return precision.get_prec_bits()


def _point_to_interval(v: jax.Array) -> jax.Array:
    vv = jnp.asarray(v, dtype=jnp.float64)
    return jnp.stack([vv, vv], axis=-1)


def cuda_besselk(
    nu: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    nbins: int = 128,
) -> jax.Array:
    checks.check_in_set(mode, ("point", "basic", "rigorous", "adaptive"), "cuda_besselk.mode")
    pb = _pb(dps, prec_bits)
    if mode == "point":
        return cuda_besselk_point(nu, z, nbins=nbins)

    nu_iv = di.as_interval(nu)
    z_iv = di.as_interval(z)
    if mode == "basic":
        nu_mid = di.midpoint(nu_iv)
        z_mid = di.midpoint(z_iv)
        base = _point_to_interval(cuda_besselk_point(nu_mid, z_mid, nbins=nbins))
        return di.round_interval_outward(base, pb)
    if mode == "rigorous":
        return ball_wrappers.arb_ball_bessel_k(nu_iv, z_iv, prec_bits=pb)
    return ball_wrappers.arb_ball_bessel_k_adaptive(nu_iv, z_iv, prec_bits=pb)


__all__ = [
    "cuda_besselk_available",
    "cuda_besselk_point",
    "cuda_besselk",
]
