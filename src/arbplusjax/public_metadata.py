from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Container, Mapping


_ALT_PREFIXES = ("boost_", "cuda_", "cusf_", "mpmath_", "scipy_", "shahen_")
_EXPERIMENTAL_MODULES = {"boost_hypgeom", "cusf_compat", "jrb_mat", "jcb_mat", "shahen_double_gamma"}
_STABLE_INTERVAL_MODES = ("point", "basic", "adaptive", "rigorous")


@dataclass(frozen=True)
class PublicFunctionMetadata:
    name: str
    qualified_name: str
    module: str
    implementation_name: str
    family: str
    stability: str
    point_support: bool
    interval_support: bool
    interval_modes: tuple[str, ...]
    method_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    derivative_status: str
    notes: str


def build_public_metadata_registry(
    public_registry: Mapping[str, Callable],
    point_names: Container[str],
    interval_names: Container[str],
) -> dict[str, PublicFunctionMetadata]:
    metadata: dict[str, PublicFunctionMetadata] = {}
    for public_name, fn in public_registry.items():
        module = fn.__module__.rsplit(".", 1)[-1]
        implementation_name = getattr(fn, "__name__", public_name)
        family = _infer_family(public_name, module)
        point_support = public_name in point_names
        interval_support = public_name in interval_names or _supports_mode_kwargs(fn)
        metadata[public_name] = PublicFunctionMetadata(
            name=public_name,
            qualified_name=f"{module}.{implementation_name}",
            module=module,
            implementation_name=implementation_name,
            family=family,
            stability=_infer_stability(public_name, module),
            point_support=point_support,
            interval_support=interval_support,
            interval_modes=_infer_interval_modes(point_support, interval_support),
            method_tags=_infer_method_tags(family),
            regime_tags=_infer_regime_tags(family),
            derivative_status=_infer_derivative_status(family, public_name, module),
            notes=_infer_notes(family, public_name, module, interval_support),
        )
    return metadata


def _supports_mode_kwargs(fn: Callable) -> bool:
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    return "mode" in code.co_varnames or "impl" in code.co_varnames


def _infer_family(name: str, module: str) -> str:
    lowered = name.lower()
    if "tail_integral" in lowered or "tail_acceleration" in lowered:
        return "integration"
    if "bessel" in lowered or "cubesselk" in lowered:
        return "bessel"
    if "barnes" in lowered or module in {"barnesg", "double_gamma", "shahen_double_gamma"}:
        return "barnes"
    if "gamma" in lowered or "pochhammer" in lowered or "rgamma" in lowered or "lgamma" in lowered:
        return "gamma"
    if "_mat_" in lowered or module in {"arb_mat", "acb_mat", "jrb_mat", "jcb_mat"}:
        return "matrix"
    if "hypgeom" in lowered or "hypergeometric" in lowered:
        return "hypergeometric"
    if "calc_integrate" in lowered or module in {"arb_calc", "acb_calc"}:
        return "integration"
    return "core"


def _infer_stability(name: str, module: str) -> str:
    if name.startswith("incomplete_bessel_"):
        return "experimental"
    if name.startswith(_ALT_PREFIXES):
        return "experimental"
    if "." in name:
        short_name = name.split(".", 1)[1]
        if short_name.startswith("incomplete_bessel_"):
            return "experimental"
        if short_name.startswith(_ALT_PREFIXES):
            return "experimental"
    if module in _EXPERIMENTAL_MODULES:
        return "experimental"
    return "stable"


def _infer_interval_modes(point_support: bool, interval_support: bool) -> tuple[str, ...]:
    if not interval_support:
        return ("point",) if point_support else ()
    return _STABLE_INTERVAL_MODES


def _infer_method_tags(family: str) -> tuple[str, ...]:
    if family == "bessel":
        return ("power_series", "asymptotic", "mode_dispatch")
    if family == "barnes":
        return ("reflection", "series", "mode_dispatch")
    if family == "gamma":
        return ("direct", "reflection", "asymptotic")
    if family == "hypergeometric":
        return ("series", "transform", "asymptotic", "mode_dispatch")
    if family == "matrix":
        return ("dense_kernel", "mode_dispatch")
    if family == "integration":
        return ("quadrature", "mode_dispatch")
    return ("direct", "mode_dispatch")


def _infer_regime_tags(family: str) -> tuple[str, ...]:
    if family == "bessel":
        return ("small_argument", "large_argument", "mixed_order_argument")
    if family == "barnes":
        return ("reflection_zone", "large_argument", "containment_sensitive")
    if family == "gamma":
        return ("reflection_zone", "large_argument", "pole_avoidance")
    if family == "hypergeometric":
        return ("series_zone", "transformed_zone", "asymptotic_zone")
    if family == "matrix":
        return ("small_dense", "batched_dense")
    if family == "integration":
        return ("finite_panel", "tail_control")
    return ("generic",)


def _infer_derivative_status(family: str, name: str, module: str) -> str:
    if family == "matrix" and module in {"jrb_mat", "jcb_mat"}:
        return "explicit_ad_planned"
    if family in {"bessel", "gamma", "hypergeometric"}:
        return "jax_autodiff_audited"
    if name.startswith(_ALT_PREFIXES) or module in _EXPERIMENTAL_MODULES:
        return "jax_autodiff_partial"
    return "jax_autodiff_basic"


def _infer_notes(family: str, name: str, module: str, interval_support: bool) -> str:
    if family == "bessel":
        return "Current scalar Bessel stack is a priority hardening target; incomplete-Bessel specializations remain planned."
    if family == "barnes":
        return "Barnes-family kernels remain on the hardening backlog, especially for tighter enclosure modes."
    if family == "matrix":
        return "Jones-labeled matrix-function layers remain experimental; canonical arb_mat/acb_mat stay primary."
    if family == "hypergeometric":
        return "Hypergeometric families use staged family-specific kernels behind the shared API."
    if family == "integration":
        return "Current integration entry points are precursors to the planned general tail-acceleration engine."
    if not interval_support and module in _EXPERIMENTAL_MODULES:
        return "Point-oriented or helper-only implementation; interval hardening is not part of the stable surface yet."
    return "Metadata is a high-level API contract summary, not an exhaustive algorithm inventory."
