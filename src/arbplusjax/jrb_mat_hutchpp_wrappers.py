from __future__ import annotations

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import jrb_mat as _jrb
from . import matrix_free_basic, matrix_free_core


def jrb_mat_hutchpp_trace_with_metadata_point(
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
        coerce_probes=di.as_interval,
        midpoint_value=di.midpoint,
        point_from_midpoint=_jrb._jrb_point_interval,
        basis_dtype=jnp.float64,
        trace_inner=lambda q, fq_cols: jnp.trace(q.T @ fq_cols),
        residual_project=lambda z, q: z - (z @ q) @ q.T,
        quadratic_reduce=lambda z_proj, hz: jnp.sum(z_proj * hz, axis=-1),
        zero_scalar=jnp.asarray(0.0, dtype=jnp.float64),
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def jrb_mat_rational_trace_hutchpp_prepare_point(
    matvec,
    sketch_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> matrix_free_core.RationalHutchppMetadata:
    def action_fn(probe):
        return _jrb.jrb_mat_rational_action_point(
            matvec,
            probe,
            shifts=shifts,
            weights=weights,
            polynomial_coefficients=polynomial_coefficients,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            symmetric=symmetric,
            preconditioner=preconditioner,
        )

    deflation = _jrb.jrb_mat_deflated_operator_prepare_point(action_fn, sketch_probes)
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
        implicit_adjoint=bool(symmetric),
        structured="symmetric" if symmetric else "general",
        algebra="jrb",
        maxiter=maxiter,
    )


def jrb_mat_rational_trace_hutchpp_from_metadata_point(
    metadata: matrix_free_core.RationalHutchppMetadata,
    residual_probes: jax.Array,
) -> matrix_free_core.HutchppTraceMetadata:
    def action_fn(probe):
        return _jrb.jrb_mat_rational_action_point(
            metadata.operator,
            probe,
            shifts=jnp.asarray(metadata.shifts),
            weights=jnp.asarray(metadata.weights),
            polynomial_coefficients=metadata.polynomial_coefficients,
            tol=jnp.asarray(metadata.tol, dtype=jnp.float64),
            atol=jnp.asarray(metadata.atol, dtype=jnp.float64),
            maxiter=metadata.maxiter,
            symmetric=(metadata.structured != "general"),
            preconditioner=metadata.preconditioner,
        )

    return _jrb.jrb_mat_trace_estimate_deflated_point(
        action_fn,
        metadata.deflation,
        residual_probes,
        target_stderr=None if metadata.target_stderr is None else jnp.asarray(metadata.target_stderr, dtype=jnp.float64),
        min_probes=None if metadata.min_probes is None else jnp.asarray(metadata.min_probes, dtype=jnp.int32),
        max_probes=None if metadata.max_probes is None else jnp.asarray(metadata.max_probes, dtype=jnp.int32),
        block_size=jnp.asarray(metadata.block_size, dtype=jnp.int32),
    )


def jrb_mat_rational_trace_hutchpp_point(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    metadata = jrb_mat_rational_trace_hutchpp_prepare_point(
        matvec,
        sketch_probes,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    estimate = jrb_mat_rational_trace_hutchpp_from_metadata_point(metadata, residual_probes)
    return jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(estimate), dtype=jnp.float64)


def jrb_mat_rational_trace_hutchpp_basic(
    matvec,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    symmetric: bool = True,
    preconditioner=None,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    return matrix_free_basic.scalar_functional_basic(
        jrb_mat_rational_trace_hutchpp_point,
        matvec,
        sketch_probes,
        residual_probes,
        shifts=shifts,
        weights=weights,
        polynomial_coefficients=polynomial_coefficients,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        symmetric=symmetric,
        preconditioner=preconditioner,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
        lift_scalar=_jrb._jrb_point_interval,
        round_output=_jrb._jrb_round_basic,
        prec_bits=prec_bits,
    )


def jrb_mat_logdet_rational_hutchpp_prepare_point(*args, **kwargs) -> matrix_free_core.RationalHutchppMetadata:
    return jrb_mat_rational_trace_hutchpp_prepare_point(*args, **kwargs)


def jrb_mat_logdet_rational_hutchpp_from_metadata_point(
    metadata: matrix_free_core.RationalHutchppMetadata,
    residual_probes: jax.Array,
) -> matrix_free_core.HutchppTraceMetadata:
    return jrb_mat_rational_trace_hutchpp_from_metadata_point(metadata, residual_probes)


def jrb_mat_logdet_rational_hutchpp_point(*args, **kwargs) -> jax.Array:
    return jrb_mat_rational_trace_hutchpp_point(*args, **kwargs)


def jrb_mat_hutchpp_trace_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    metadata = jrb_mat_hutchpp_trace_with_metadata_point(action_fn, sketch_probes, residual_probes)
    return jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(metadata), dtype=jnp.float64)


def jrb_mat_hutchpp_trace_estimate_point(action_fn, sketch_probes: jax.Array, residual_probes: jax.Array) -> jax.Array:
    return jrb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jrb_mat_hutchpp_trace_estimate_basic(
    action_fn,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    metadata = jrb_mat_hutchpp_trace_with_metadata_point(action_fn, sketch_probes, residual_probes)
    value = _jrb._jrb_round_basic(
        _jrb._jrb_point_interval(jnp.asarray(matrix_free_core.hutchpp_trace_from_metadata(metadata), dtype=jnp.float64)),
        prec_bits,
    )
    return _jrb._jrb_inflate_basic_scalar(value, metadata.statistics)


def jrb_mat_logdet_leja_hutchpp_point(
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
    action_fn = lambda v: _jrb.jrb_mat_log_action_leja_point(
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
    return jrb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)


def jrb_mat_det_leja_hutchpp_point(
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
        jrb_mat_logdet_leja_hutchpp_point(
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


def jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
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
) -> tuple[jax.Array, _jrb.JrbMatKrylovDiagnostics]:
    value = jrb_mat_logdet_leja_hutchpp_point(
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
    reference = di.as_interval(sketch_probes)
    if reference.shape[0] > 0:
        _, action_diag = _jrb.jrb_mat_log_action_leja_with_diagnostics_point(
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
        algorithm_code = action_diag.algorithm_code
    else:
        used_steps = jnp.asarray(max_degree if max_degree is not None else degree, dtype=jnp.int32)
        tail_norm = jnp.asarray(0.0, dtype=jnp.float64)
        algorithm_code = jnp.asarray(3, dtype=jnp.int32)
    diag = _jrb.JrbMatKrylovDiagnostics(
        algorithm_code=jnp.asarray(algorithm_code, dtype=jnp.int32),
        steps=jnp.asarray(used_steps, dtype=jnp.int32),
        basis_dim=jnp.asarray(used_steps, dtype=jnp.int32),
        restart_count=jnp.asarray(0, dtype=jnp.int32),
        beta0=jnp.asarray(0.0, dtype=jnp.float64),
        tail_norm=jnp.asarray(tail_norm, dtype=jnp.float64),
        breakdown=jnp.asarray(False),
        used_adjoint=jnp.asarray(False),
        gradient_supported=jnp.asarray(True),
        probe_count=jnp.asarray(reference.shape[0] + di.as_interval(residual_probes).shape[0], dtype=jnp.int32),
    )
    return value, diag


def jrb_mat_det_leja_hutchpp_with_diagnostics_point(
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
) -> tuple[jax.Array, _jrb.JrbMatKrylovDiagnostics]:
    value, diag = jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
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


def jrb_mat_bcoo_logdet_leja_hutchpp_point(
    a: _jrb.sparse_common.SparseBCOO,
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
    bounds_steps: int = 16,
    bounds_safety_margin: float = 1.25,
) -> jax.Array:
    bounds = (
        _jrb.jrb_mat_bcoo_spectral_bounds_adaptive(
            a,
            steps=bounds_steps,
            safety_margin=bounds_safety_margin,
        )
        if spectral_bounds is None
        else spectral_bounds
    )
    return jrb_mat_logdet_leja_hutchpp_point(
        _jrb.jrb_mat_bcoo_operator(a),
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


def jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(
    a: _jrb.sparse_common.SparseBCOO,
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
    bounds_steps: int = 16,
    bounds_safety_margin: float = 1.25,
) -> tuple[jax.Array, _jrb.JrbMatKrylovDiagnostics]:
    bounds = (
        _jrb.jrb_mat_bcoo_spectral_bounds_adaptive(
            a,
            steps=bounds_steps,
            safety_margin=bounds_safety_margin,
        )
        if spectral_bounds is None
        else spectral_bounds
    )
    return jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(
        _jrb.jrb_mat_bcoo_operator(a),
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
