from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurvatureSpec:
    kind: str
    representation: str = "operator"
    differentiation_mode: str = "forward_over_reverse"
    damping: float = 0.0
    jitter: float = 0.0
    symmetrize: bool = True
    enforce_psd: bool = False
