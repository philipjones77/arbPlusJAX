from __future__ import annotations

import jax
import jax.numpy as jnp

from . import jcb_mat as _jcb
from . import matrix_free_basic, matrix_free_core
from . import double_interval as di


def jcb_mat_logdet_estimate_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return _jcb.jcb_mat_logdet_slq_point(matvec, probes, steps, adjoint_matvec)


def jcb_mat_logdet_estimate_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return jcb_mat_logdet_slq_basic(
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        prec_bits=prec_bits,
    )


def jcb_mat_heat_trace_slq_hermitian_from_metadata_point(metadata: matrix_free_core.SlqQuadratureMetadata, time) -> jax.Array:
    return matrix_free_core.slq_heat_trace_from_metadata(metadata, time)


def jcb_mat_heat_trace_slq_hermitian_point(matvec, probes: jax.Array, steps: int, *, time) -> jax.Array:
    metadata = _jcb._jcb_mat_slq_prepare_hermitian_point(matvec, probes, steps, scalar_fun=jnp.log)
    return jcb_mat_heat_trace_slq_hermitian_from_metadata_point(metadata, time)


def jcb_mat_heat_trace_slq_hermitian_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    time,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jcb_mat_heat_trace_slq_hermitian_point,
        matvec,
        probes,
        steps,
        time=time,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_spectral_density_slq_hermitian_from_metadata_point(
    metadata: matrix_free_core.SlqQuadratureMetadata,
    bin_edges: jax.Array,
    *,
    normalize: bool = False,
) -> jax.Array:
    return matrix_free_core.slq_spectral_density_from_metadata(metadata, bin_edges, normalize=normalize)


def jcb_mat_spectral_density_slq_hermitian_point(
    matvec,
    probes: jax.Array,
    steps: int,
    *,
    bin_edges: jax.Array,
    normalize: bool = False,
) -> jax.Array:
    metadata = _jcb._jcb_mat_slq_prepare_hermitian_point(matvec, probes, steps, scalar_fun=jnp.log)
    return jcb_mat_spectral_density_slq_hermitian_from_metadata_point(metadata, bin_edges, normalize=normalize)


def jcb_mat_logdet_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        _jcb.jcb_mat_logdet_slq_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_det_slq_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(_jcb.jcb_mat_logdet_slq_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_det_slq_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jcb_mat_det_slq_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_logdet_slq_hermitian_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    used_adjoint = matvec if adjoint_matvec is None else adjoint_matvec
    return _jcb.mat_common.estimator_mean(
        probes,
        _jcb.acb_core.as_acb_box,
        lambda v: _jcb.jcb_mat_funm_trace_integrand_hermitian_point(
            matvec,
            v,
            jnp.log,
            steps=steps,
            adjoint_matvec=used_adjoint,
        ),
        probe_midpoint=_jcb.acb_core.acb_midpoint,
    )


def jcb_mat_logdet_slq_hpd_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return jcb_mat_logdet_slq_hermitian_point(matvec, probes, steps, adjoint_matvec)


def jcb_mat_det_slq_hermitian_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(jcb_mat_logdet_slq_hermitian_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_det_slq_hpd_point(matvec, probes: jax.Array, steps: int, adjoint_matvec=None) -> jax.Array:
    return matrix_free_core.det_from_logdet(jcb_mat_logdet_slq_hpd_point(matvec, probes, steps, adjoint_matvec))


def jcb_mat_logdet_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        _jcb.jcb_mat_logdet_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
        inflate_output=_jcb._jcb_inflate_basic_scalar,
        invalidate_output=_jcb._full_box_like,
    )


def jcb_mat_det_slq_with_diagnostics_basic(
    matvec,
    probes: jax.Array,
    steps: int,
    adjoint_matvec=None,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
):
    return matrix_free_basic.scalar_functional_with_diagnostics_basic(
        _jcb.jcb_mat_det_slq_with_diagnostics_point,
        matvec,
        probes,
        steps,
        adjoint_matvec=adjoint_matvec,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
        inflate_output=_jcb._jcb_inflate_basic_scalar,
        invalidate_output=_jcb._full_box_like,
    )
