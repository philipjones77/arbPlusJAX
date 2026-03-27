from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp

from .types import CurvatureSpec


MatvecFn = Callable[[Any], Any]


@dataclass(frozen=True)
class CurvatureOperator:
    shape: tuple[int, int]
    dtype: Any
    matvec: MatvecFn
    rmatvec: MatvecFn | None = None
    to_dense_fn: Callable[[], Any] | None = None
    diagonal_fn: Callable[[], Any] | None = None
    trace_fn: Callable[[], Any] | None = None
    solve_fn: Callable[..., Any] | None = None
    logdet_fn: Callable[..., Any] | None = None
    inverse_diagonal_fn: Callable[..., Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def transpose_matvec(self, v):
        fn = self.rmatvec if self.rmatvec is not None else self.matvec
        return fn(v)

    def to_dense(self):
        if self.to_dense_fn is None:
            raise NotImplementedError("dense materialization is not available")
        return self.to_dense_fn()

    def diagonal(self):
        if self.diagonal_fn is not None:
            return self.diagonal_fn()
        dense = self.to_dense()
        return jnp.diag(dense)

    def trace(self):
        if self.trace_fn is not None:
            return self.trace_fn()
        return jnp.trace(self.to_dense())

    def solve(self, b, **kwargs):
        if self.solve_fn is None:
            raise NotImplementedError("solve is not available")
        return self.solve_fn(b, **kwargs)

    def logdet(self, **kwargs):
        if self.logdet_fn is None:
            raise NotImplementedError("logdet is not available")
        return self.logdet_fn(**kwargs)

    def inverse_diagonal(self, **kwargs):
        if self.inverse_diagonal_fn is None:
            raise NotImplementedError("inverse diagonal is not available")
        return self.inverse_diagonal_fn(**kwargs)

    def is_symmetric(self) -> bool:
        return bool(self.metadata.get("symmetric", False))

    def is_psd(self) -> bool | None:
        return self.metadata.get("psd")


def make_curvature_operator(
    *,
    shape: tuple[int, int],
    dtype,
    matvec: MatvecFn,
    rmatvec: MatvecFn | None = None,
    to_dense_fn: Callable[[], Any] | None = None,
    diagonal_fn: Callable[[], Any] | None = None,
    trace_fn: Callable[[], Any] | None = None,
    solve_fn: Callable[..., Any] | None = None,
    logdet_fn: Callable[..., Any] | None = None,
    inverse_diagonal_fn: Callable[..., Any] | None = None,
    metadata: dict[str, Any] | None = None,
    spec: CurvatureSpec | None = None,
) -> CurvatureOperator:
    merged_metadata = {} if metadata is None else dict(metadata)
    if spec is not None:
        merged_metadata.setdefault("kind", spec.kind)
        merged_metadata.setdefault("representation", spec.representation)
        merged_metadata.setdefault("differentiation_mode", spec.differentiation_mode)
        merged_metadata.setdefault("damping", spec.damping)
        merged_metadata.setdefault("jitter", spec.jitter)
        merged_metadata.setdefault("symmetrize", spec.symmetrize)
        if spec.symmetrize:
            merged_metadata.setdefault("symmetric", True)
        if spec.enforce_psd:
            merged_metadata.setdefault("psd", True)
    return CurvatureOperator(
        shape=shape,
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        to_dense_fn=to_dense_fn,
        diagonal_fn=diagonal_fn,
        trace_fn=trace_fn,
        solve_fn=solve_fn,
        logdet_fn=logdet_fn,
        inverse_diagonal_fn=inverse_diagonal_fn,
        metadata=merged_metadata,
    )
