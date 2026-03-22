from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkMeasurement:
    name: str
    value: float | int | str | bool | None
    unit: str = ""
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_name: str
    concern: str
    category: str
    implementation: str
    operation: str
    device: str
    dtype: str
    cold_time_s: float | None = None
    warm_time_s: float | None = None
    recompile_time_s: float | None = None
    python_overhead_s: float | None = None
    memory_bytes: int | None = None
    accuracy_abs: float | None = None
    accuracy_rel: float | None = None
    residual: float | None = None
    ad_forward_time_s: float | None = None
    ad_backward_time_s: float | None = None
    ad_residual: float | None = None
    measurements: tuple[BenchmarkMeasurement, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["measurements"] = [asdict(item) for item in self.measurements]
        return payload


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: str = "v1"
    benchmark_name: str = ""
    concern: str = ""
    category: str = ""
    records: tuple[BenchmarkRecord, ...] = ()
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


def write_benchmark_report(path: Path, report: BenchmarkReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

