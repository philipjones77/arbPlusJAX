from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TailEvaluationDiagnostics:
    method: str
    chunk_count: int
    panel_count: int
    recurrence_steps: int
    estimated_tail_remainder: float
    instability_flags: tuple[str, ...] = ()
    fallback_used: bool = False
    precision_warning: bool = False
    note: str = ""
