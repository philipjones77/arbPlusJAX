from __future__ import annotations

import jax
import jax.numpy as jnp

from . import jrb_mat as _jrb
from . import matrix_free_basic, matrix_free_core
from . import double_interval as di


def jrb_mat_logdet_estimate_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return _jrb.jrb_mat_logdet_slq_point(matvec, probes, steps)


def jrb_mat_logdet_estimate_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return jrb_mat_logdet_slq_basic(matvec, probes, steps, prec_bits=prec_bits)


def jrb_mat_heat_trace_slq_from_metadata_point(metadata: matrix_free_core.SlqQuadratureMetadata, time) -> jax.Array:
    return matrix_free_core.slq_heat_trace_from_metadata(metadata, time)


def jrb_mat_heat_trace_slq_point(matvec, probes: jax.Array, steps: int, *, time) -> jax.Array:
    metadata = _jrb._jrb_mat_slq_prepare_point(matvec, probes, steps, scalar_fun=jnp.log)
    return jrb_mat_heat_trace_slq_from_metadata_point(metadata, time)


def jrb_mat_heat_trace_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    time,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jrb_mat_heat_trace_slq_point,
        matvec,
        probes,
        steps,
        time=time,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_spectral_density_slq_from_metadata_point(
    metadata: matrix_free_core.SlqQuadratureMetadata,
    bin_edges: jax.Array,
    *,
    normalize: bool = False,
) -> jax.Array:
    return matrix_free_core.slq_spectral_density_from_metadata(metadata, bin_edges, normalize=normalize)


def jrb_mat_spectral_density_slq_point(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    bin_edges: jax.Array,
    normalize: bool = False,
) -> jax.Array:
    metadata = _jrb._jrb_mat_slq_prepare_point(matvec, probes, steps, scalar_fun=jnp.log)
    return jrb_mat_spectral_density_slq_from_metadata_point(metadata, bin_edges, normalize=normalize)


def jrb_mat_logdet_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        _jrb.jrb_mat_logdet_slq_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_det_slq_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return matrix_free_core.det_from_logdet(_jrb.jrb_mat_logdet_slq_point(matvec, probes, steps))


def jrb_mat_det_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jrb_mat_det_slq_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_logdet_slq_symmetric_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return _jrb.jrb_mat_logdet_slq_point(matvec, probes, steps)


def jrb_mat_det_slq_symmetric_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_det_slq_point(matvec, probes, steps)


def jrb_mat_logdet_slq_spd_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return _jrb.jrb_mat_logdet_slq_point(matvec, probes, steps)


def jrb_mat_det_slq_spd_point(matvec, probes: jax.Array, steps: int) -> jax.Array:
    return jrb_mat_det_slq_point(matvec, probes, steps)


def jrb_mat_logdet_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        _jrb.jrb_mat_logdet_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
        inflate_output=_jrb._jrb_inflate_basic_scalar,
        invalidate_output=_jrb._full_interval_like,
    )


def jrb_mat_det_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        _jrb.jrb_mat_det_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
        inflate_output=_jrb._jrb_inflate_basic_scalar,
        invalidate_output=_jrb._full_interval_like,
    )
