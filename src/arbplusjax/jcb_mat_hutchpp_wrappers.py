from __future__ import annotations

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import jcb_mat as _jcb
from . import matrix_free_basic, matrix_free_core


def jcb_mat_hutchpp_trace_with_metadata_point(
    action_fn,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> matrix_free_core.HutchppTraceMetadata:
    return matrix_free_core.hutchpp_trace_with_metadata_projected_point(
        action_fn,
        sketch_probes,
        residual_probes,
        coerce_probes=_jcb.acb_core.as_acb_box,
        midpoint_value=_jcb.acb_core.acb_midpoint,
        point_from_midpoint=_jcb._jcb_point_box,
        basis_dtype=jnp.complex128,
        trace_inner=lambda q, fq_cols: jnp.trace(jnp.conjugate(q).T @ fq_cols),
        residual_project=lambda z, q: z - (z @ q) @ jnp.conjugate(q).T,
        quadratic_reduce=lambda z_proj, hz: jnp.sum(jnp.conjugate(z_proj) * hz, axis=-1),
        zero_scalar=jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def jcb_mat_rational_trace_hutchpp_prepare_point(
    matvec,
    sketch_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    adjoint_matvec=None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> matrix_free_core.RationalHutchppMetadata:
    del adjoint_matvec

    def action_fn(probe):
        return _jcb.jcb_mat_rational_action_point(
            matvec,
            probe,
            shifts=shifts,
            weights=weights,
            polynomial_coefficients=polynomial_coefficients,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            hermitian=hermitian,
            preconditioner=preconditioner,
        )

    deflation = _jcb.jcb_mat_deflated_operator_prepare_point(action_fn, sketch_probes)
    return matrix_free_core.make_rational_hutchpp_metadata(
        operator=matvec,
        deflation=deflation,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        preconditioner=preconditioner,
        tol=tol,
        atol=atol,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
        gradient_supported=True,
        implicit_adjoint=bool(hermitian),
        structured="hermitian" if hermitian else "general",
        algebra="jcb",
        maxiter=maxiter,
    )


def jcb_mat_rational_trace_hutchpp_from_metadata_point(
    metadata: matrix_free_core.RationalHutchppMetadata,
    residual_probes: jax.Array,
) -> matrix_free_core.HutchppTraceMetadata:
    def action_fn(probe):
        return _jcb.jcb_mat_rational_action_point(
            metadata.operator,
            probe,
            shifts=jnp.asarray(metadata.shifts),
            weights=jnp.asarray(metadata.weights),
            polynomial_coefficients=metadata.polynomial_coefficients,
            tol=jnp.asarray(metadata.tol, dtype=jnp.float64),
            atol=jnp.asarray(metadata.atol, dtype=jnp.float64),
            maxiter=metadata.maxiter,
            hermitian=(metadata.structured != "general"),
            preconditioner=metadata.preconditioner,
        )

    return _jcb.jcb_mat_trace_estimate_deflated_point(
        action_fn,
        metadata.deflation,
        residual_probes,
        target_stderr=None if metadata.target_stderr is None else jnp.asarray(metadata.target_stderr, dtype=jnp.float64),
        min_probes=None if metadata.min_probes is None else jnp.asarray(metadata.min_probes, dtype=jnp.int32),
        max_probes=None if metadata.max_probes is None else jnp.asarray(metadata.max_probes, dtype=jnp.int32),
        block_size=jnp.asarray(metadata.block_size, dtype=jnp.int32),
    )


def jcb_mat_rational_trace_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    adjoint_matvec=None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    metadata = jcb_mat_rational_trace_hutchpp_prepare_point(
        matvec,
        sketch_probes,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        adjoint_matvec=adjoint_matvec,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    estimate = jcb_mat_rational_trace_hutchpp_from_metadata_point(metadata, residual_probes)
    return jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(estimate), dtype=jnp.complex128)


def jcb_mat_rational_trace_hutchpp_basic(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    adjoint_matvec=None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    hermitian: bool = False,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jcb_mat_rational_trace_hutchpp_point,
        matvec,
        sketch_probes,
        residual_probes,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        adjoint_matvec=adjoint_matvec,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        hermitian=hermitian,
        preconditioner=preconditioner,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
        lift_scalar=_jcb._jcb_point_box,
        round_output=_jcb._jcb_round_basic,
        prec_bits=prec_bits,
    )


def jcb_mat_logdet_rational_hutchpp_prepare_point(*args, **kwargs) -> matrix_free_core.RationalHutchppMetadata:
    return jcb_mat_rational_trace_hutchpp_prepare_point(*args, **kwargs)


def jcb_mat_logdet_rational_hutchpp_from_metadata_point(
    metadata: matrix_free_core.RationalHutchppMetadata,
    residual_probes: jax.Array,
) -> matrix_free_core.HutchppTraceMetadata:
    return jcb_mat_rational_trace_hutchpp_from_metadata_point(metadata, residual_probes)


def jcb_mat_logdet_rational_hutchpp_point(*args, **kwargs) -> jax.Array:
    return jcb_mat_rational_trace_hutchpp_point(*args, **kwargs)


def jcb_mat_hutchpp_trace_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    metadata = jcb_mat_hutchpp_trace_with_metadata_point(action_fn, sketch_probes, residual_probes)
    return jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(metadata), dtype=jnp.complex128)


def jcb_mat_hutchpp_trace_estimate_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    return jcb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jcb_mat_hutchpp_trace_estimate_basic(
    action_fn,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    metadata = jcb_mat_hutchpp_trace_with_metadata_point(action_fn, sketch_probes, residual_probes)
    value = _jcb._jcb_round_basic(
        _jcb._jcb_point_box(jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(metadata), dtype=jnp.complex128)),
        prec_bits,
    )
    return _jcb._jcb_inflate_basic_scalar(value, metadata.statistics)


def jcb_mat_logdet_leja_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    action_fn = lambda v: _jcb.jcb_mat_log_action_leja_point(
        matvec,
        v,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    return jcb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, _jcb.JcbMatKrylovDiagnostics]:
    value = jcb_mat_logdet_leja_hutchpp_point(
        matvec,
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    reference = _jcb.acb_core.as_acb_box(sketch_probes)
    if reference.shape[0] > 0:
        _, action_diag = _jcb.jcb_mat_log_action_leja_with_diagnostics_point(
            matvec,
            reference[0],
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
        used_steps = action_diag.steps
        tail_norm = action_diag.tail_norm
    else:
        used_steps = jnp.asarray(max_degree if max_degree is not None else degree, dtype=jnp.int32)
        tail_norm = jnp.asarray(0.0, dtype=jnp.float64)
    diag = _jcb.JcbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(3, dtype=jnp.int32),
        steps=jnp.asarray(used_steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(used_steps, dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(0.0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(
            _jcb.acb_core.as_acb_box(sketch_probes).shape[0] + _jcb.acb_core.as_acb_box(residual_probes).shape[0],
            dtype=jnp.int32,
        ),
    )
    return value, diag


def jcb_mat_det_leja_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    return matrix_free_core.det_from_logdet(
        jcb_mat_logdet_leja_hutchpp_point(
            matvec,
            sketch_probes,
            residual_probes,
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
    )


def jcb_mat_det_leja_hutchpp_with_diagnostics_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array],
    candidate_count: int = 64,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, _jcb.JcbMatKrylovDiagnostics]:
    value, diag = jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        matvec,
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=spectral_bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )
    return matrix_free_core.det_from_logdet(value), diag


def jcb_mat_bcoo_logdet_leja_hutchpp_point(
    a: _jcb.sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    bounds = _jcb.jcb_mat_bcoo_gershgorin_bounds(a) if spectral_bounds is None else spectral_bounds
    return jcb_mat_logdet_leja_hutchpp_point(
        _jcb.jcb_mat_bcoo_operator(a),
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )


def jcb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
    a: _jcb.sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[jax.Array, _jcb.JcbMatKrylovDiagnostics]:
    bounds = _jcb.jcb_mat_bcoo_gershgorin_bounds(a) if spectral_bounds is None else spectral_bounds
    return jcb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        _jcb.jcb_mat_bcoo_operator(a),
        sketch_probes,
        residual_probes,
        degree=degree,
        spectral_bounds=bounds,
        candidate_count=candidate_count,
        max_degree=max_degree,
        min_degree=min_degree,
        rtol=rtol,
        atol=atol,
    )


def jcb_mat_bcoo_det_leja_hutchpp_point(
    a: _jcb.sparse_common.SparseBCOO,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    degree: int = 32,
    spectral_bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    candidate_count: int = 96,
    max_degree: int | None = None,
    min_degree: int = 8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> jax.Array:
    return matrix_free_core.det_from_logdet(
        jcb_mat_bcoo_logdet_leja_hutchpp_point(
            a,
            sketch_probes,
            residual_probes,
            degree=degree,
            spectral_bounds=spectral_bounds,
            candidate_count=candidate_count,
            max_degree=max_degree,
            min_degree=min_degree,
            rtol=rtol,
            atol=atol,
        )
    )
