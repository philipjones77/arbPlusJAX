from __future__ import annotations

from collections import deque
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from . import matrix_free_core as _core


class DistanceKColoringDiagnostics(NamedTuple):
    distance_k: int
    color_count: int
    colors: jax.Array
    color_sizes: jax.Array


class ColoredInverseDiagonalDiagnostics(NamedTuple):
    distance_k: int
    color_count: int
    colors: jax.Array
    color_sizes: jax.Array
    block_size: int
    correction_probes: int
    stderr: jax.Array


def _symmetrized_adjacency_from_edges(
    rows: jax.Array,
    cols: jax.Array,
    *,
    size: int,
) -> list[set[int]]:
    rr = jnp.asarray(rows, dtype=jnp.int32).reshape(-1)
    cc = jnp.asarray(cols, dtype=jnp.int32).reshape(-1)
    adjacency = [set() for _ in range(int(size))]
    for r, c in zip(rr.tolist(), cc.tolist(), strict=False):
        if r < 0 or c < 0 or r >= size or c >= size or r == c:
            continue
        adjacency[r].add(c)
        adjacency[c].add(r)
    return adjacency


def greedy_distance_k_coloring_from_edges(
    rows: jax.Array,
    cols: jax.Array,
    *,
    size: int,
    distance_k: int = 1,
) -> DistanceKColoringDiagnostics:
    if distance_k <= 0:
        raise ValueError("distance_k must be > 0")
    adjacency = _symmetrized_adjacency_from_edges(rows, cols, size=size)
    colors = [-1] * int(size)
    for node in range(int(size)):
        forbidden: set[int] = set()
        visited = {node}
        frontier: deque[tuple[int, int]] = deque([(node, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth == distance_k:
                continue
            for nbr in adjacency[current]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                if colors[nbr] >= 0:
                    forbidden.add(colors[nbr])
                frontier.append((nbr, depth + 1))
        color = 0
        while color in forbidden:
            color += 1
        colors[node] = color
    color_arr = jnp.asarray(colors, dtype=jnp.int32)
    color_count = int(jnp.max(color_arr)) + 1 if size > 0 else 0
    color_sizes = jnp.bincount(color_arr, length=color_count)
    return DistanceKColoringDiagnostics(
        distance_k=int(distance_k),
        color_count=int(color_count),
        colors=color_arr,
        color_sizes=jnp.asarray(color_sizes, dtype=jnp.int32),
    )


def colored_inverse_diagonal_estimate_from_solve(
    solve_fn,
    *,
    size: int,
    rows: jax.Array,
    cols: jax.Array,
    dtype,
    distance_k: int = 1,
    block_size: int = 1,
    correction_probes: int = 0,
    key: jax.Array | None = None,
) -> tuple[jax.Array, ColoredInverseDiagonalDiagnostics]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if correction_probes < 0:
        raise ValueError("correction_probes must be >= 0")
    coloring = greedy_distance_k_coloring_from_edges(rows, cols, size=size, distance_k=distance_k)
    work_dtype = jnp.dtype(dtype)
    real_dtype = jnp.float64
    key = jax.random.PRNGKey(0) if key is None else jnp.asarray(key)
    diag_est = jnp.zeros((int(size),), dtype=work_dtype)
    for color in range(coloring.color_count):
        idx = jnp.asarray(jnp.nonzero(coloring.colors == color, size=int(size), fill_value=-1)[0], dtype=jnp.int32)
        valid = idx[idx >= 0]
        if valid.size == 0:
            continue
        key, subkey = jax.random.split(key)
        probe_values = jax.random.rademacher(subkey, (valid.shape[0], block_size), dtype=real_dtype).astype(work_dtype)
        probes = jnp.zeros((int(size), block_size), dtype=work_dtype).at[valid].set(probe_values)
        solved_cols = [jnp.asarray(solve_fn(probes[:, j])) for j in range(block_size)]
        solved = jnp.stack(solved_cols, axis=1)
        contrib = jnp.mean(probes * solved, axis=1)
        diag_est = diag_est.at[valid].set(contrib[valid])

    if correction_probes > 0:
        samples = []
        for _ in range(int(correction_probes)):
            key, subkey = jax.random.split(key)
            probe = jax.random.rademacher(subkey, (int(size),), dtype=real_dtype).astype(work_dtype)
            solved = jnp.asarray(solve_fn(probe))
            samples.append(probe * solved)
        _, _, stderr = probe_sample_statistics(jnp.stack(samples, axis=0))
    else:
        stderr = jnp.zeros((int(size),), dtype=real_dtype)

    diagnostics = ColoredInverseDiagonalDiagnostics(
        distance_k=int(coloring.distance_k),
        color_count=int(coloring.color_count),
        colors=coloring.colors,
        color_sizes=coloring.color_sizes,
        block_size=int(block_size),
        correction_probes=int(correction_probes),
        stderr=jnp.asarray(stderr, dtype=real_dtype),
    )
    return diag_est, diagnostics


def rational_spectral_action_midpoint(
    apply_operator,
    solve_shifted,
    x_mid: jax.Array,
    *,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    coeff_dtype,
) -> jax.Array:
    x_arr = jnp.asarray(x_mid)
    out_dtype = jnp.result_type(x_arr.dtype, coeff_dtype)
    acc = _core.polynomial_spectral_action_midpoint(
        apply_operator,
        x_arr,
        jnp.asarray([0.0], dtype=coeff_dtype) if polynomial_coefficients is None else polynomial_coefficients,
        coeff_dtype=coeff_dtype,
    )
    shifts_arr = jnp.asarray(shifts, dtype=coeff_dtype)
    weights_arr = jnp.asarray(weights, dtype=coeff_dtype)

    def body(carry, xs):
        shift, weight = xs
        resolved = jnp.asarray(solve_shifted(shift, x_arr), dtype=out_dtype)
        return carry + jnp.asarray(weight, dtype=out_dtype) * resolved, None

    value, _ = lax.scan(body, acc, (shifts_arr, weights_arr))
    return value


def probe_sample_statistics(samples: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    values = jnp.asarray(samples)
    if values.ndim == 0:
        raise ValueError("samples must have at least one probe axis")
    count = values.shape[0]
    if count <= 0:
        raise ValueError("samples must contain at least one probe")
    mean = jnp.mean(values, axis=0)
    centered = values - mean
    sq_norm = jnp.real(centered * jnp.conjugate(centered))
    variance = jnp.mean(sq_norm, axis=0) if count == 1 else jnp.sum(sq_norm, axis=0) / jnp.asarray(count - 1, dtype=jnp.float64)
    stderr = jnp.sqrt(variance / jnp.asarray(count, dtype=jnp.float64))
    return mean, variance, stderr


def make_probe_estimate_statistics(
    samples: jax.Array,
    *,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.ProbeEstimateStatistics:
    values = jnp.asarray(samples)
    mean, variance, stderr = probe_sample_statistics(values)
    probe_count = jnp.asarray(values.shape[0], dtype=jnp.int32)
    if target_stderr is None:
        recommended = probe_count
    else:
        recommended = adaptive_probe_count_from_pilot(
            values,
            target_stderr=target_stderr,
            min_probes=min_probes,
            max_probes=max_probes,
            block_size=block_size,
        )
    return _core.ProbeEstimateStatistics(
        mean=mean,
        variance=variance,
        stderr=stderr,
        probe_count=probe_count,
        recommended_probe_count=jnp.asarray(recommended, dtype=jnp.int32),
    )


def adaptive_probe_count_from_pilot(
    pilot_samples: jax.Array,
    *,
    target_stderr: float,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> jax.Array:
    values = jnp.asarray(pilot_samples)
    if values.ndim == 0:
        raise ValueError("pilot_samples must have a probe axis")
    count = int(values.shape[0])
    if count <= 0:
        raise ValueError("pilot_samples must contain at least one probe")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    _, _, stderr = probe_sample_statistics(values)
    stderr_scalar = jnp.max(jnp.asarray(stderr, dtype=jnp.float64))
    target = jnp.maximum(jnp.asarray(target_stderr, dtype=jnp.float64), jnp.asarray(1e-30, dtype=jnp.float64))
    required = jnp.ceil((stderr_scalar / target) ** 2 * jnp.asarray(count, dtype=jnp.float64)).astype(jnp.int32)
    required = jnp.maximum(required, jnp.asarray(count if min_probes is None else min_probes, dtype=jnp.int32))
    if max_probes is not None:
        required = jnp.minimum(required, jnp.asarray(max_probes, dtype=jnp.int32))
    block = jnp.asarray(block_size, dtype=jnp.int32)
    rounded = block * ((required + block - 1) // block)
    if max_probes is not None:
        rounded = jnp.minimum(rounded, jnp.asarray(max_probes, dtype=jnp.int32))
    return rounded


def probe_statistics_target_met(
    statistics: _core.ProbeEstimateStatistics,
    *,
    target_stderr: float,
) -> jax.Array:
    target = jnp.asarray(target_stderr, dtype=jnp.float64)
    return jnp.asarray(jnp.max(jnp.asarray(statistics.stderr, dtype=jnp.float64)) <= target)


def probe_statistics_should_stop(
    statistics: _core.ProbeEstimateStatistics,
    *,
    target_stderr: float,
    max_probes: int | None = None,
) -> jax.Array:
    met = probe_statistics_target_met(statistics, target_stderr=target_stderr)
    if max_probes is None:
        return met
    return jnp.asarray(met | (jnp.asarray(statistics.probe_count, dtype=jnp.int32) >= jnp.asarray(max_probes, dtype=jnp.int32)))


def probe_statistics_probe_deficit(statistics: _core.ProbeEstimateStatistics) -> jax.Array:
    recommended = jnp.asarray(statistics.recommended_probe_count, dtype=jnp.int32)
    current = jnp.asarray(statistics.probe_count, dtype=jnp.int32)
    return jnp.maximum(recommended - current, jnp.asarray(0, dtype=jnp.int32))


def probe_statistics_next_probe_count(
    statistics: _core.ProbeEstimateStatistics,
    *,
    block_size: int | None = None,
    max_probes: int | None = None,
) -> jax.Array:
    current = jnp.asarray(statistics.probe_count, dtype=jnp.int32)
    next_count = jnp.maximum(current, jnp.asarray(statistics.recommended_probe_count, dtype=jnp.int32))
    if block_size is not None:
        block = jnp.asarray(block_size, dtype=jnp.int32)
        next_count = block * ((next_count + block - 1) // block)
    if max_probes is not None:
        next_count = jnp.minimum(next_count, jnp.asarray(max_probes, dtype=jnp.int32))
    return next_count


def apply_action_over_probe_block_point(
    action_fn,
    probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
) -> jax.Array:
    coerced = coerce_probes(probes)
    outputs = jax.vmap(action_fn)(coerced)
    return midpoint_value(outputs)


def hutchpp_trace_with_metadata_projected_point(
    action_fn,
    sketch_probes: jax.Array,
    residual_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    basis_dtype,
    trace_inner,
    residual_project,
    quadratic_reduce,
    zero_scalar,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.HutchppTraceMetadata:
    sketch = coerce_probes(sketch_probes)
    residual = coerce_probes(residual_probes)
    n = int(sketch.shape[-2] if sketch.shape[0] > 0 else residual.shape[-2])

    if sketch.shape[0] > 0:
        y_cols = jnp.swapaxes(
            apply_action_over_probe_block_point(
                action_fn,
                sketch,
                coerce_probes=coerce_probes,
                midpoint_value=midpoint_value,
            ),
            0,
            1,
        )
        q, _ = jnp.linalg.qr(y_cols, mode="reduced")
        fq_cols = jnp.swapaxes(
            apply_action_over_probe_block_point(
                action_fn,
                jax.vmap(point_from_midpoint)(q.T),
                coerce_probes=coerce_probes,
                midpoint_value=midpoint_value,
            ),
            0,
            1,
        )
        trace_lr = trace_inner(q, fq_cols)
    else:
        q = jnp.zeros((n, 0), dtype=basis_dtype)
        trace_lr = zero_scalar

    if residual.shape[0] > 0:
        z = midpoint_value(residual)
        z_proj = residual_project(z, q)
        hz = apply_action_over_probe_block_point(
            action_fn,
            jax.vmap(point_from_midpoint)(z_proj),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        )
        residual_samples = quadratic_reduce(z_proj, hz)
    else:
        residual_samples = jnp.zeros((1,), dtype=jnp.asarray(zero_scalar).dtype)

    return make_hutchpp_trace_metadata(
        basis=q,
        low_rank_trace=trace_lr,
        residual_samples=residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def make_deflated_operator_metadata(
    *,
    basis: jax.Array,
    image: jax.Array,
    low_rank_trace,
) -> _core.DeflatedOperatorMetadata:
    return _core.DeflatedOperatorMetadata(
        basis=jnp.asarray(basis),
        image=jnp.asarray(image),
        low_rank_trace=low_rank_trace,
    )


def make_rational_hutchpp_metadata(
    *,
    operator,
    deflation: _core.DeflatedOperatorMetadata,
    shifts: jax.Array,
    weights: jax.Array,
    polynomial_coefficients: jax.Array | None = None,
    preconditioner=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
    gradient_supported: bool = True,
    implicit_adjoint: bool = False,
    structured: str = "",
    algebra: str = "",
    maxiter: int | None = None,
) -> _core.RationalHutchppMetadata:
    transpose_preconditioner = None
    if implicit_adjoint:
        transpose_preconditioner = _core.preconditioner_transpose_plan(
            preconditioner,
            algebra=algebra,
            conjugate=(structured == "hermitian"),
        )
    cached_adjoint_supported = bool(
        gradient_supported and implicit_adjoint and (preconditioner is None or transpose_preconditioner is not None)
    )
    return _core.RationalHutchppMetadata(
        operator=operator,
        deflation=deflation,
        shifts=jnp.asarray(shifts),
        weights=jnp.asarray(weights),
        polynomial_coefficients=None if polynomial_coefficients is None else jnp.asarray(polynomial_coefficients),
        preconditioner=preconditioner,
        transpose_preconditioner=transpose_preconditioner,
        tol=jnp.asarray(tol),
        atol=jnp.asarray(atol),
        target_stderr=None if target_stderr is None else jnp.asarray(target_stderr, dtype=jnp.float64),
        min_probes=None if min_probes is None else jnp.asarray(min_probes, dtype=jnp.int32),
        max_probes=None if max_probes is None else jnp.asarray(max_probes, dtype=jnp.int32),
        block_size=jnp.asarray(block_size, dtype=jnp.int32),
        gradient_supported=jnp.asarray(gradient_supported),
        implicit_adjoint=jnp.asarray(implicit_adjoint),
        cached_adjoint_supported=jnp.asarray(cached_adjoint_supported),
        structured=structured,
        algebra=algebra,
        maxiter=maxiter,
    )


def prepare_deflated_operator_metadata_point(
    action_fn,
    sketch_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    basis_dtype,
    trace_inner,
) -> _core.DeflatedOperatorMetadata:
    sketch = coerce_probes(sketch_probes)
    n = int(sketch.shape[-2])
    if sketch.shape[0] == 0:
        empty = jnp.zeros((n, 0), dtype=basis_dtype)
        return make_deflated_operator_metadata(
            basis=empty,
            image=empty,
            low_rank_trace=jnp.asarray(0.0, dtype=basis_dtype),
        )
    y_cols = jnp.swapaxes(
        apply_action_over_probe_block_point(
            action_fn,
            sketch,
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        ),
        0,
        1,
    )
    q, _ = jnp.linalg.qr(y_cols, mode="reduced")
    aq = jnp.swapaxes(
        apply_action_over_probe_block_point(
            action_fn,
            jax.vmap(point_from_midpoint)(q.T),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        ),
        0,
        1,
    )
    return make_deflated_operator_metadata(
        basis=q,
        image=aq,
        low_rank_trace=trace_inner(q, aq),
    )


def deflated_operator_apply_midpoint(
    x_mid: jax.Array,
    *,
    deflation: _core.DeflatedOperatorMetadata,
    apply_operator_midpoint,
    conjugate_inner: bool = False,
) -> jax.Array:
    x_arr = jnp.asarray(x_mid)
    ax = jnp.asarray(apply_operator_midpoint(x_arr))
    basis = jnp.asarray(deflation.basis)
    if basis.shape[1] == 0:
        return ax
    if conjugate_inner:
        coeffs = jnp.conjugate(basis).T @ x_arr
    else:
        coeffs = basis.T @ x_arr
    return ax - jnp.asarray(deflation.image) @ coeffs


def deflated_trace_estimate_from_metadata_point(
    action_fn,
    deflation: _core.DeflatedOperatorMetadata,
    residual_probes: jax.Array,
    *,
    coerce_probes,
    midpoint_value,
    point_from_midpoint,
    residual_project,
    quadratic_reduce,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.HutchppTraceMetadata:
    residual = coerce_probes(residual_probes)
    if residual.shape[0] > 0:
        z = midpoint_value(residual)
        z_proj = residual_project(z, jnp.asarray(deflation.basis))
        hz = apply_action_over_probe_block_point(
            lambda probe: point_from_midpoint(
                deflated_operator_apply_midpoint(
                    midpoint_value(probe),
                    deflation=deflation,
                    apply_operator_midpoint=lambda v_mid: midpoint_value(action_fn(point_from_midpoint(v_mid))),
                    conjugate_inner=jnp.iscomplexobj(jnp.asarray(deflation.basis)),
                )
            ),
            jax.vmap(point_from_midpoint)(z_proj),
            coerce_probes=coerce_probes,
            midpoint_value=midpoint_value,
        )
        residual_samples = quadratic_reduce(z_proj, hz)
    else:
        residual_samples = jnp.zeros((1,), dtype=jnp.asarray(deflation.low_rank_trace).dtype)
    return make_hutchpp_trace_metadata(
        basis=deflation.basis,
        low_rank_trace=deflation.low_rank_trace,
        residual_samples=residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def slq_nodes_weights(projected: jax.Array, beta0, *, hermitian: bool) -> tuple[jax.Array, jax.Array]:
    projected_arr = jnp.asarray(projected)
    beta0_arr = jnp.asarray(beta0)
    if hermitian:
        nodes, vecs = jnp.linalg.eigh(projected_arr)
        first_row = vecs[0, :]
        weights = (jnp.real(beta0_arr * jnp.conjugate(beta0_arr)) * jnp.real(first_row * jnp.conjugate(first_row))).astype(jnp.float64)
        return jnp.asarray(nodes), weights
    nodes, vecs = jnp.linalg.eig(projected_arr)
    first_row = vecs[0, :]
    weights = (beta0_arr * jnp.conjugate(beta0_arr)) * (first_row * jnp.conjugate(first_row))
    return jnp.asarray(nodes), jnp.asarray(weights)


def slq_scalar_quadrature(nodes: jax.Array, weights: jax.Array, scalar_fun, *, scalar_postprocess=None):
    values = weights * scalar_fun(nodes)
    total = jnp.sum(values)
    return total if scalar_postprocess is None else scalar_postprocess(total)


def slq_heat_trace(nodes: jax.Array, weights: jax.Array, time):
    time_value = jnp.asarray(time)
    return slq_scalar_quadrature(
        nodes,
        weights,
        lambda vals: jnp.exp(-time_value * vals),
    )


def slq_spectral_density(nodes: jax.Array, weights: jax.Array, bin_edges: jax.Array, *, normalize: bool = False) -> jax.Array:
    edges = jnp.asarray(bin_edges)
    vals = jnp.asarray(nodes)
    coeffs = jnp.real(jnp.asarray(weights, dtype=jnp.complex128))
    left = vals[:, None] >= edges[:-1][None, :]
    right = vals[:, None] < edges[1:][None, :]
    last_edge = edges[-1]
    last_bin = vals[:, None] == last_edge
    mask = (left & right) | (last_bin & (jnp.arange(edges.shape[0] - 1) == edges.shape[0] - 2)[None, :])
    hist = jnp.sum(jnp.where(mask, coeffs[:, None], 0.0), axis=0)
    if normalize:
        total = jnp.sum(hist)
        hist = jnp.where(total > 0, hist / total, hist)
    return hist


def slq_functional_statistics_from_metadata(
    metadata: _core.SlqQuadratureMetadata,
    scalar_function,
) -> _core.ProbeEstimateStatistics:
    values = jax.vmap(lambda nodes, weights: scalar_function(nodes, weights))(metadata.nodes, metadata.weights)
    return make_probe_estimate_statistics(values)


def slq_functional_mean_from_metadata(
    metadata: _core.SlqQuadratureMetadata,
    scalar_function,
):
    return slq_functional_statistics_from_metadata(metadata, scalar_function).mean


def slq_heat_trace_from_metadata(metadata: _core.SlqQuadratureMetadata, time):
    return slq_functional_mean_from_metadata(
        metadata,
        lambda nodes, weights: slq_heat_trace(nodes, weights, time),
    )


def slq_spectral_density_from_metadata(
    metadata: _core.SlqQuadratureMetadata,
    bin_edges: jax.Array,
    *,
    normalize: bool = False,
) -> jax.Array:
    return jnp.mean(
        jax.vmap(lambda nodes, weights: slq_spectral_density(nodes, weights, bin_edges, normalize=normalize))(metadata.nodes, metadata.weights),
        axis=0,
    )


def hutchpp_trace_from_metadata(metadata: _core.HutchppTraceMetadata):
    return metadata.low_rank_trace + metadata.residual_trace


def rational_hutchpp_probe_deficit(
    metadata: _core.RationalHutchppMetadata,
    estimate: _core.HutchppTraceMetadata,
) -> jax.Array:
    del metadata
    return probe_statistics_probe_deficit(estimate.statistics)


def rational_hutchpp_next_probe_count(
    metadata: _core.RationalHutchppMetadata,
    estimate: _core.HutchppTraceMetadata,
) -> jax.Array:
    max_probes = None if metadata.max_probes is None else int(jnp.asarray(metadata.max_probes))
    return probe_statistics_next_probe_count(
        estimate.statistics,
        block_size=int(jnp.asarray(metadata.block_size)),
        max_probes=max_probes,
    )


def rational_hutchpp_should_stop(
    metadata: _core.RationalHutchppMetadata,
    estimate: _core.HutchppTraceMetadata,
) -> jax.Array:
    if metadata.target_stderr is None:
        if metadata.max_probes is None:
            return jnp.asarray(False)
        return jnp.asarray(
            jnp.asarray(estimate.statistics.probe_count, dtype=jnp.int32)
            >= jnp.asarray(metadata.max_probes, dtype=jnp.int32)
        )
    return probe_statistics_should_stop(
        estimate.statistics,
        target_stderr=float(jnp.asarray(metadata.target_stderr)),
        max_probes=None if metadata.max_probes is None else int(jnp.asarray(metadata.max_probes)),
    )


def slq_prepare_metadata_point(
    lanczos_tridiag,
    probes: jax.Array,
    steps: int,
    *,
    coerce_probes,
    hermitian: bool,
    scalar_fun,
    scalar_postprocess=None,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.SlqQuadratureMetadata:
    coerced = coerce_probes(probes)
    projected, beta0 = jax.vmap(lambda v: lanczos_tridiag(v, steps)[1:])(coerced)
    sample_values = jax.vmap(
        lambda proj, beta: slq_scalar_quadrature(
            *slq_nodes_weights(proj, beta, hermitian=hermitian),
            scalar_fun,
            scalar_postprocess=scalar_postprocess,
        )
    )(projected, beta0)
    return make_slq_quadrature_metadata(
        projected,
        beta0,
        sample_values,
        hermitian=hermitian,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )


def make_slq_quadrature_metadata(
    projected: jax.Array,
    beta0,
    sample_values: jax.Array,
    *,
    hermitian: bool,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.SlqQuadratureMetadata:
    nodes, weights = jax.vmap(lambda proj, beta: slq_nodes_weights(proj, beta, hermitian=hermitian))(jnp.asarray(projected), jnp.asarray(beta0))
    stats = make_probe_estimate_statistics(
        sample_values,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    return _core.SlqQuadratureMetadata(
        projected=projected,
        beta0=beta0,
        nodes=nodes,
        weights=weights,
        statistics=stats,
        steps=jnp.asarray(jnp.asarray(projected).shape[-1], dtype=jnp.int32),
        hermitian=jnp.asarray(hermitian),
    )


def make_hutchpp_trace_metadata(
    *,
    basis: jax.Array,
    low_rank_trace,
    residual_samples: jax.Array,
    target_stderr: float | None = None,
    min_probes: int | None = None,
    max_probes: int | None = None,
    block_size: int = 1,
) -> _core.HutchppTraceMetadata:
    stats = make_probe_estimate_statistics(
        residual_samples,
        target_stderr=target_stderr,
        min_probes=min_probes,
        max_probes=max_probes,
        block_size=block_size,
    )
    return _core.HutchppTraceMetadata(
        basis=basis,
        low_rank_trace=low_rank_trace,
        residual_trace=stats.mean,
        statistics=stats,
    )
