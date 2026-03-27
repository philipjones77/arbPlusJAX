from __future__ import annotations

import json
from functools import lru_cache

from . import api
from .function_provenance import engineering_status_for_public_name


DOWNSTREAM_KERNELS: dict[str, dict[str, object]] = {
    "gamma": {
        "public_name": "gamma",
        "family": "gamma",
        "notes": "Canonical downstream gamma entry point.",
    },
    "loggamma": {
        "public_name": "arb_lgamma",
        "complex_public_name": "acb_lgamma",
        "family": "gamma",
        "notes": "Downstream alias that routes to `arb_lgamma` for real inputs and `acb_lgamma` for complex inputs.",
    },
    "incomplete_gamma_lower": {
        "public_name": "incomplete_gamma_lower",
        "family": "gamma",
        "notes": "Supported downstream incomplete-gamma lower kernel.",
    },
    "incomplete_gamma_upper": {
        "public_name": "incomplete_gamma_upper",
        "family": "gamma",
        "notes": "Supported downstream incomplete-gamma upper kernel.",
    },
    "incomplete_bessel_i": {
        "public_name": "incomplete_bessel_i",
        "family": "bessel",
        "notes": "Supported downstream incomplete-Bessel-I kernel.",
    },
    "incomplete_bessel_k": {
        "public_name": "incomplete_bessel_k",
        "family": "bessel",
        "notes": "Supported downstream incomplete-Bessel-K kernel.",
    },
    "provider_incomplete_bessel_i": {
        "public_name": "incomplete_bessel_i",
        "family": "bessel",
        "notes": "Explicit downstream provider surface for incomplete-Bessel-I with diagnostics-aware method routing through the public API.",
    },
    "provider_incomplete_bessel_k": {
        "public_name": "incomplete_bessel_k",
        "family": "bessel",
        "notes": "Explicit downstream provider surface for incomplete-Bessel-K with diagnostics-aware method routing through the public API.",
    },
    "barnesdoublegamma": {
        "public_name": "ifj_barnesdoublegamma",
        "family": "barnes",
        "notes": "Supported downstream Barnes double-gamma kernel through the IFJ-compatible public provider surface.",
    },
    "provider_barnesdoublegamma": {
        "public_name": "ifj_barnesdoublegamma",
        "family": "barnes",
        "notes": "Explicit downstream provider surface for Barnes double-gamma with diagnostics-aware IFJ routing through the public API.",
    },
    "log_barnesdoublegamma": {
        "public_name": "ifj_log_barnesdoublegamma",
        "family": "barnes",
        "notes": "Supported downstream log Barnes double-gamma kernel through the IFJ-compatible public provider surface.",
    },
    "provider_log_barnesdoublegamma": {
        "public_name": "ifj_log_barnesdoublegamma",
        "family": "barnes",
        "notes": "Explicit downstream provider surface for log Barnes double-gamma with diagnostics-aware IFJ routing through the public API.",
    },
    "fragile_regime_promotion_gamma_upper": {
        "public_name": "incomplete_gamma_upper",
        "family": "gamma",
        "notes": "Downstream promotion hook for fragile upper incomplete-gamma regimes; callers should use method='high_precision_refine' or method='auto' and inspect diagnostics.",
    },
    "fragile_regime_promotion_bessel_k": {
        "public_name": "incomplete_bessel_k",
        "family": "bessel",
        "notes": "Downstream promotion hook for fragile incomplete-Bessel-K regimes; callers should use method='high_precision_refine' or method='auto' and inspect diagnostics.",
    },
    "fragile_regime_promotion_bessel_i": {
        "public_name": "incomplete_bessel_i",
        "family": "bessel",
        "notes": "Downstream promotion hook for fragile incomplete-Bessel-I regimes; callers should use method='high_precision_refine' or method='auto' and inspect diagnostics.",
    },
}


def _generated_at() -> str:
    return "2026-03-17T00:00:00Z"


@lru_cache(maxsize=1)
def build_capability_registry() -> dict[str, object]:
    metadata = {entry.name: entry for entry in api.list_public_function_metadata()}
    downstream_aliases: dict[str, list[str]] = {}
    for alias, spec in DOWNSTREAM_KERNELS.items():
        downstream_aliases.setdefault(str(spec["public_name"]), []).append(alias)

    functions: dict[str, dict[str, object]] = {}
    for name, entry in metadata.items():
        engineering = engineering_status_for_public_name(name) or {}
        functions[name] = {
            "name": entry.name,
            "qualified_name": entry.qualified_name,
            "module": entry.module,
            "implementation_name": entry.implementation_name,
            "family": entry.family,
            "stability": entry.stability,
            "point_support": entry.point_support,
            "interval_support": entry.interval_support,
            "interval_modes": list(entry.interval_modes),
            "value_kinds": list(entry.value_kinds),
            "implementation_options": list(entry.implementation_options),
            "implementation_versions": list(entry.implementation_versions),
            "default_implementation": entry.default_implementation,
            "method_tags": list(entry.method_tags),
            "default_method": entry.default_method,
            "method_parameter_names": list(entry.method_parameter_names),
            "execution_strategies": list(entry.execution_strategies),
            "regime_tags": list(entry.regime_tags),
            "derivative_status": entry.derivative_status,
            "notes": entry.notes,
            "engineering": engineering,
            "downstream_supported": name in downstream_aliases,
            "downstream_aliases": sorted(downstream_aliases.get(name, ())),
        }

    downstream = {}
    for alias, spec in DOWNSTREAM_KERNELS.items():
        public_name = str(spec["public_name"])
        row = {
            "alias": alias,
            "public_name": public_name,
            "family": str(spec["family"]),
            "notes": str(spec["notes"]),
        }
        row["capability"] = functions[public_name]
        complex_name = spec.get("complex_public_name")
        if complex_name is not None:
            row["complex_public_name"] = str(complex_name)
            row["complex_capability"] = functions[str(complex_name)]
        downstream[alias] = row
    return {
        "generated_at": _generated_at(),
        "source": "arbplusjax.capability_registry",
        "policy_refs": [
            "docs/standards/jax_surface_policy_standard.md",
            "docs/standards/engineering_standard.md",
        ],
        "downstream_kernels": downstream,
        "functions": functions,
    }


def render_capability_registry_json() -> str:
    return json.dumps(build_capability_registry(), indent=2, sort_keys=True) + "\n"


def lookup_capability(name: str) -> dict[str, object]:
    registry = build_capability_registry()
    functions = dict(registry["functions"])
    downstream = dict(registry["downstream_kernels"])
    if name in downstream:
        return downstream[name]
    if name in functions:
        return functions[name]
    raise KeyError(name)


__all__ = [
    "DOWNSTREAM_KERNELS",
    "build_capability_registry",
    "render_capability_registry_json",
    "lookup_capability",
]
