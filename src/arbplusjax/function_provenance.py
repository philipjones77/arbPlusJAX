from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "references" / "inventory" / "function_list.md"

SUFFIXES = (
    "_batch_prec_jit",
    "_batch_jit",
    "_batch_prec",
    "_prec_jit",
    "_point",
    "_mode",
    "_rigorous",
    "_batch",
    "_prec",
    "_jit",
)
ALT_PREFIXES = ("bdg_", "boost_", "cuda_", "cusf_", "jaxsci_", "mpmath_", "scipy_")
NEW_MODULE_HINTS = ("acb_dirichlet", "acb_modular")


@dataclass(frozen=True)
class FunctionEntry:
    public_name: str
    preferred_public_name: str
    base_name: str
    category: str
    lineage: str
    module: str
    four_modes: str
    tightening: str
    references: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class ImplementationEntry:
    base_name: str
    category: str
    public_name: str
    preferred_public_name: str
    lineage: str
    module: str
    four_modes: str
    tightening: str
    why: str


POLICY_TEXT = """Last updated: 2026-03-07T00:00:00Z

# Function Naming Policy

## Model

- `arb_like`: canonical Arb/FLINT-style public mathematical surface. These names remain unprefixed.
- `alternative`: same mathematical target as an Arb-like function, but from a different implementation lineage. These names must use a provenance prefix.
- `new`: mathematical families without an Arb/FLINT-style canonical base name in this repo. These use a descriptive family name and do not need an alternative-prefix migration.

## Rules

- Public naming is based on the mathematical function name, not the Python filename.
- Alternative implementations must use `prefix_<base_name>`.
- The prefix is chosen when the function family is introduced and should reflect provenance, not a vague label like `custom`.
- No prefix implies the canonical Arb-like public function for this repo.
- If a function is intended to become part of the canonical Arb-like public surface, it stays unprefixed and must meet the same four-mode and tightening expectations as the rest of that surface.
- If a function is not canonical and implements the same mathematical target as an Arb-like name, it should not be placed in the canonical namespace.

## Examples

- Canonical: `besselk`
- Alternative: `cuda_besselk`, `boost_besselk`
- New family: `modular_j`

## Current repo intent

- `arb_core` / `acb_core` define the canonical Arb-like public surface for this repo, including approved Arb-like extensions.
- External-lineage implementations such as Boost- or CUDA-lineage code should be prefixed in the public API.
- Python module names may remain implementation-oriented; the policy applies to public function names.
"""


SPECIAL_BASE_NAMES = {
    "cusf_besselj": "besselj",
    "cusf_bessely": "bessely",
    "cusf_besseli": "besseli",
    "cusf_besselk": "besselk",
    "boost_hypergeometric_1f0": "hypergeometric_1f0",
    "boost_hypergeometric_0f1": "hypergeometric_0f1",
    "boost_hypergeometric_1f1": "hypergeometric_1f1",
    "boost_hypergeometric_2f0": "hypergeometric_2f0",
    "boost_hypergeometric_pfq": "hypergeometric_pfq",
    "cuda_besselk": "besselk",
    "cuda_besselk_point": "besselk",
}

SPECIAL_PREFERRED_NAMES = {
    "cusf_hyp1f1": "cusf_hyp1f1",
    "cusf_hyp2f1": "cusf_hyp2f1",
    "cusf_erf": "cusf_erf",
    "cusf_gamma": "cusf_gamma",
    "cusf_digamma": "cusf_digamma",
    "cusf_faddeeva_w": "cusf_faddeeva_w",
    "cusf_tgamma1pmv": "cusf_tgamma1pmv",
    "cusf_chebyshev": "cusf_chebyshev",
    "cusf_polynomial": "cusf_polynomial",
    "cusf_poly_rational": "cusf_poly_rational",
    "cusf_besselj": "cusf_besselj",
    "cusf_bessely": "cusf_bessely",
    "cusf_besseli": "cusf_besseli",
    "cusf_besselk": "cusf_besselk",
    "boost_hypergeometric_1f0": "boost_hypergeometric_1f0",
    "boost_hypergeometric_0f1": "boost_hypergeometric_0f1",
    "boost_hypergeometric_1f1": "boost_hypergeometric_1f1",
    "boost_hypergeometric_2f0": "boost_hypergeometric_2f0",
    "boost_hypergeometric_pfq": "boost_hypergeometric_pfq",
}

SPECIAL_CATEGORY = {
}

SPECIAL_NOTES = {
    "cusf_hyp1f1": "Alternative implementation aligned to the canonical Arb-like `hyp1f1` base name.",
    "cusf_hyp2f1": "Alternative implementation aligned to the canonical Arb-like `hyp2f1` base name.",
    "cusf_erf": "Alternative implementation aligned to the canonical Arb-like `erf` base name.",
    "cusf_besselj": "Alternative implementation aligned to the canonical Arb-like `besselj` base name.",
    "cusf_bessely": "Alternative implementation aligned to the canonical Arb-like `bessely` base name.",
    "cusf_besseli": "Alternative implementation aligned to the canonical Arb-like `besseli` base name.",
    "cusf_besselk": "Alternative implementation aligned to the canonical Arb-like `besselk` base name.",
    "cuda_besselk": "Alternative implementation aligned to the canonical Arb-like `besselk` base name.",
    "cuda_besselk_point": "Alternative implementation aligned to the canonical Arb-like `besselk` base name.",
}

CORE_TIGHTNESS: dict[str, str] = {
    "sin_pi": "specialized rigorous dispatch complete",
    "cos_pi": "specialized rigorous dispatch complete",
    "tan_pi": "specialized rigorous dispatch complete",
    "sinc": "specialized rigorous dispatch complete",
    "sinc_pi": "specialized rigorous dispatch complete",
    "sign": "specialized rigorous dispatch complete",
    "pow_fmpq": "specialized rigorous dispatch complete",
    "root": "specialized rigorous dispatch complete",
    "cbrt": "specialized rigorous dispatch complete",
    "lgamma": "specialized rigorous dispatch complete",
    "rgamma": "specialized rigorous dispatch complete",
    "sinh_cosh": "specialized rigorous dispatch complete",
    "rsqrt": "specialized rigorous dispatch complete",
    "cot": "specialized rigorous dispatch complete",
    "sech": "specialized rigorous dispatch complete",
    "csch": "specialized rigorous dispatch complete",
    "sin_cos_pi": "specialized rigorous dispatch complete",
    "cot_pi": "specialized rigorous dispatch complete",
    "csc_pi": "specialized rigorous dispatch complete",
    "exp_pi_i": "specialized rigorous dispatch complete",
    "exp_invexp": "specialized rigorous dispatch complete",
    "addmul": "specialized rigorous dispatch complete",
    "submul": "specialized rigorous dispatch complete",
    "pow_arb": "specialized rigorous dispatch complete",
    "pow_si": "specialized rigorous dispatch complete",
    "sqr": "specialized rigorous dispatch complete",
    "root_ui": "specialized rigorous dispatch complete",
    "log_sin_pi": "specialized rigorous dispatch complete",
    "digamma": "specialized rigorous dispatch complete",
    "zeta": "specialized rigorous dispatch complete",
    "hurwitz_zeta": "specialized rigorous dispatch complete",
    "polygamma": "specialized rigorous dispatch complete",
    "bernoulli_poly_ui": "specialized rigorous dispatch complete",
    "polylog": "specialized rigorous dispatch complete",
    "polylog_si": "specialized rigorous dispatch complete",
    "agm": "specialized rigorous dispatch complete",
    "agm1": "specialized rigorous dispatch complete",
    "agm1_cpx": "specialized rigorous dispatch complete",
}

MANUAL_IMPLEMENTATIONS: tuple[ImplementationEntry, ...] = (
    ImplementationEntry("besselk", "alternative", "cusf_besselk", "cusf_besselk", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `besselk` base name."),
    ImplementationEntry("besseli", "alternative", "cusf_besseli", "cusf_besseli", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `besseli` base name."),
    ImplementationEntry("besselj", "alternative", "cusf_besselj", "cusf_besselj", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `besselj` base name."),
    ImplementationEntry("bessely", "alternative", "cusf_bessely", "cusf_bessely", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `bessely` base name."),
    ImplementationEntry("erf", "alternative", "cusf_erf", "cusf_erf", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `erf` base name."),
    ImplementationEntry("gamma", "alternative", "cusf_gamma", "cusf_gamma", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation aligned to the canonical Arb-like `gamma` base name."),
    ImplementationEntry("digamma", "alternative", "cusf_digamma", "cusf_digamma", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation aligned to the canonical Arb-like `digamma` base name."),
    ImplementationEntry("hyp1f1", "alternative", "cusf_hyp1f1", "cusf_hyp1f1", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `hyp1f1` base name."),
    ImplementationEntry("hyp2f1", "alternative", "cusf_hyp2f1", "cusf_hyp2f1", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `hyp2f1` base name."),
    ImplementationEntry("faddeeva_w", "alternative", "cusf_faddeeva_w", "cusf_faddeeva_w", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation for the Faddeeva helper family."),
    ImplementationEntry("tgamma1pmv", "alternative", "cusf_tgamma1pmv", "cusf_tgamma1pmv", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation for the tgamma1pmv helper family."),
    ImplementationEntry("chebyshev", "alternative", "cusf_chebyshev", "cusf_chebyshev", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation for the Chebyshev evaluation helper family."),
    ImplementationEntry("polynomial", "alternative", "cusf_polynomial", "cusf_polynomial", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation for the polynomial evaluation helper family."),
    ImplementationEntry("poly_rational", "alternative", "cusf_poly_rational", "cusf_poly_rational", "CuSF/CUDA-style alternative implementation", "src/arbplusjax/cusf_compat.py", "point", "point-only or helper path", "Alternative implementation for the rational polynomial helper family."),
    ImplementationEntry("hypergeometric_1f0", "alternative", "boost_hypergeometric_1f0", "boost_hypergeometric_1f0", "Boost.Math-inspired alternative implementation", "src/arbplusjax/boost_hypgeom.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the hypergeometric 1f0 family."),
    ImplementationEntry("hypergeometric_0f1", "alternative", "boost_hypergeometric_0f1", "boost_hypergeometric_0f1", "Boost.Math-inspired alternative implementation", "src/arbplusjax/boost_hypgeom.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the hypergeometric 0f1 family."),
    ImplementationEntry("hypergeometric_1f1", "alternative", "boost_hypergeometric_1f1", "boost_hypergeometric_1f1", "Boost.Math-inspired alternative implementation", "src/arbplusjax/boost_hypgeom.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the hypergeometric 1f1 family."),
    ImplementationEntry("hypergeometric_2f0", "alternative", "boost_hypergeometric_2f0", "boost_hypergeometric_2f0", "Boost.Math-inspired alternative implementation", "src/arbplusjax/boost_hypgeom.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the hypergeometric 2f0 family."),
    ImplementationEntry("hypergeometric_pfq", "alternative", "boost_hypergeometric_pfq", "boost_hypergeometric_pfq", "Boost.Math-inspired alternative implementation", "src/arbplusjax/boost_hypgeom.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the generic hypergeometric pfq family."),
    ImplementationEntry("besselk", "alternative", "cuda_besselk", "cuda_besselk", "CUDA/CubesselK-inspired alternative Bessel-K implementation", "src/arbplusjax/cubesselk.py", "point|basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Alternative implementation aligned to the canonical Arb-like `besselk` base name."),
)

MANUAL_FUNCTION_ENTRIES: tuple[FunctionEntry, ...] = tuple(
    [
        FunctionEntry(name, name, base, "alternative", "Julia/BarnesDoubleGamma.jl-derived alternative implementation", "double_gamma", modes, tightening, ("docs/implementation/modules/double_gamma.md",), note)
        for name, base, modes, tightening, note in (
            ("bdg_log_barnesdoublegamma", "log_barnesdoublegamma", "point", "point-only or helper path", "Julia-derived alternative implementation of the Barnes double-gamma logarithm."),
            ("bdg_barnesdoublegamma", "barnesdoublegamma", "point", "point-only or helper path", "Julia-derived alternative implementation of the Barnes double-gamma value."),
            ("bdg_log_barnesgamma2", "log_barnesgamma2", "point", "point-only or helper path", "Julia-derived alternative implementation of the log-BarnesGamma2 family."),
            ("bdg_barnesgamma2", "barnesgamma2", "point", "point-only or helper path", "Julia-derived alternative implementation of the BarnesGamma2 family."),
            ("bdg_log_normalizeddoublegamma", "log_normalizeddoublegamma", "point", "point-only or helper path", "Julia-derived alternative implementation of the normalized double-gamma logarithm."),
            ("bdg_normalizeddoublegamma", "normalizeddoublegamma", "point", "point-only or helper path", "Julia-derived alternative implementation of the normalized double-gamma family."),
            ("bdg_double_sine", "double_sine", "point", "point-only or helper path", "Julia-derived alternative implementation of the double-sine family."),
            ("bdg_interval_log_barnesdoublegamma", "log_barnesdoublegamma", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived Barnes double-gamma logarithm."),
            ("bdg_interval_barnesdoublegamma", "barnesdoublegamma", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived Barnes double-gamma value."),
            ("bdg_interval_log_barnesgamma2", "log_barnesgamma2", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived log-BarnesGamma2 family."),
            ("bdg_interval_barnesgamma2", "barnesgamma2", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived BarnesGamma2 family."),
            ("bdg_interval_log_normalizeddoublegamma", "log_normalizeddoublegamma", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived normalized double-gamma logarithm."),
            ("bdg_interval_normalizeddoublegamma", "normalizeddoublegamma", "basic", "implementation-specific mode-aware tightening", "Real interval wrapper for the Julia-derived normalized double-gamma family."),
            ("bdg_complex_log_barnesdoublegamma", "log_barnesdoublegamma", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived Barnes double-gamma logarithm."),
            ("bdg_complex_barnesdoublegamma", "barnesdoublegamma", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived Barnes double-gamma value."),
            ("bdg_complex_log_barnesgamma2", "log_barnesgamma2", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived log-BarnesGamma2 family."),
            ("bdg_complex_barnesgamma2", "barnesgamma2", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived BarnesGamma2 family."),
            ("bdg_complex_log_normalizeddoublegamma", "log_normalizeddoublegamma", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived normalized double-gamma logarithm."),
            ("bdg_complex_normalizeddoublegamma", "normalizeddoublegamma", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived normalized double-gamma family."),
            ("bdg_complex_double_sine", "double_sine", "basic", "implementation-specific mode-aware tightening", "Complex box wrapper for the Julia-derived double-sine family."),
            ("bdg_interval_log_barnesdoublegamma_mode", "log_barnesdoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived Barnes double-gamma logarithm."),
            ("bdg_interval_barnesdoublegamma_mode", "barnesdoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived Barnes double-gamma value."),
            ("bdg_interval_log_barnesgamma2_mode", "log_barnesgamma2", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived log-BarnesGamma2 family."),
            ("bdg_interval_barnesgamma2_mode", "barnesgamma2", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived BarnesGamma2 family."),
            ("bdg_interval_log_normalizeddoublegamma_mode", "log_normalizeddoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived normalized double-gamma logarithm."),
            ("bdg_interval_normalizeddoublegamma_mode", "normalizeddoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched real interval wrapper for the Julia-derived normalized double-gamma family."),
            ("bdg_complex_log_barnesdoublegamma_mode", "log_barnesdoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived Barnes double-gamma logarithm."),
            ("bdg_complex_barnesdoublegamma_mode", "barnesdoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived Barnes double-gamma value."),
            ("bdg_complex_log_barnesgamma2_mode", "log_barnesgamma2", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived log-BarnesGamma2 family."),
            ("bdg_complex_barnesgamma2_mode", "barnesgamma2", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived BarnesGamma2 family."),
            ("bdg_complex_log_normalizeddoublegamma_mode", "log_normalizeddoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived normalized double-gamma logarithm."),
            ("bdg_complex_normalizeddoublegamma_mode", "normalizeddoublegamma", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived normalized double-gamma family."),
            ("bdg_complex_double_sine_mode", "double_sine", "basic|adaptive|rigorous", "implementation-specific mode-aware tightening", "Mode-dispatched complex box wrapper for the Julia-derived double-sine family."),
        )
    ]
)


def _parse_section(header: str) -> list[str]:
    text = INVENTORY_PATH.read_text(encoding="utf-8")
    section = text.split(header, 1)[1]
    items: list[str] = []
    for line in section.splitlines()[1:]:
        if line.startswith("## "):
            break
        if line.startswith("- `") and line.endswith("`"):
            items.append(line[3:-1])
    return items


def point_functions() -> set[str]:
    return set(_parse_section("## Point Functions"))


def interval_functions() -> set[str]:
    return set(_parse_section("## Interval Functions"))


def public_functions() -> list[str]:
    return _parse_section("## Public Functions")


def _leaf_name(public_name: str) -> str:
    return public_name.rsplit(".", 1)[-1]


def _module_name(public_name: str) -> str:
    return public_name.rsplit(".", 1)[0] if "." in public_name else ""


def _normalize_base_name(leaf: str) -> str:
    if leaf in SPECIAL_BASE_NAMES:
        return SPECIAL_BASE_NAMES[leaf]
    core = leaf
    for prefix in ALT_PREFIXES:
        if core.lower().startswith(prefix):
            core = core[len(prefix):]
            break
    for suffix in SUFFIXES:
        if core.endswith(suffix):
            core = core[: -len(suffix)]
            break
    if core.startswith(("arb_", "acb_")):
        core = core[4:]
    return core.lower()


def _preferred_public_name(leaf: str) -> str:
    if leaf in SPECIAL_PREFERRED_NAMES:
        return SPECIAL_PREFERRED_NAMES[leaf]
    return leaf


def _category(leaf: str, module: str, base_name: str) -> str:
    if leaf in SPECIAL_CATEGORY:
        return SPECIAL_CATEGORY[leaf]
    lower_leaf = leaf.lower()
    lower_module = module.lower()
    if lower_module == "double_gamma":
        return "alternative"
    if lower_leaf.startswith(ALT_PREFIXES) or lower_module.endswith("cusf_compat"):
        return "alternative"
    if any(hint in lower_module for hint in NEW_MODULE_HINTS) or base_name in {"modular_j", "dirichlet_zeta", "dirichlet_eta"}:
        return "new"
    return "arb_like"


def _lineage(category: str, leaf: str, module: str) -> str:
    lower_leaf = leaf.lower()
    lower_module = module.lower()
    if category == "alternative":
        if lower_leaf.startswith("bdg_"):
            return "Julia/BarnesDoubleGamma.jl-derived alternative implementation"
        if "double_gamma" in lower_module:
            return "Julia/BarnesDoubleGamma.jl-derived alternative implementation"
        if lower_leaf.startswith("cusf_") or "cusf" in lower_module:
            return "CuSF/CUDA-style alternative implementation"
        if lower_leaf.startswith("boost_"):
            return "Boost.Math-inspired alternative implementation"
        if lower_leaf.startswith("cuda_"):
            return "CUDA-style alternative implementation"
        if lower_leaf.startswith("jaxsci_"):
            return "JAX SciPy-style alternative implementation"
        return "Alternative implementation"
    if category == "new":
        if "modular" in lower_module:
            return "Repo-original modular function family"
        if "dirichlet" in lower_module:
            return "Repo-original Dirichlet helper family"
        if "double_gamma" in lower_module:
            return "Repo-original Barnes/double-gamma family"
        if "cubesselk" in lower_module or "cubesselk" in leaf.lower():
            return "Repo-original CubesselK lineage"
        return "Repo-original mathematical family"
    return "Arb/FLINT-style canonical or Arb-like extension surface"


def _mode_profile(public_name: str, leaf: str, base_name: str, points: set[str], intervals: set[str]) -> str:
    if leaf.startswith("bdg_complex_") or leaf.startswith("bdg_interval_"):
        return "basic|adaptive|rigorous" if leaf.endswith("_mode") else "basic"
    if leaf.startswith("bdg_"):
        return "point"
    if leaf.endswith("_point"):
        return "point"
    if leaf.endswith("_mode"):
        return "basic|adaptive|rigorous"
    if leaf.endswith("_rigorous"):
        return "rigorous"
    if base_name in points and base_name in intervals:
        return "point|basic|adaptive|rigorous"
    if base_name in points:
        return "point"
    if base_name in intervals:
        return "basic|adaptive|rigorous"
    if leaf.endswith("_prec"):
        return "basic"
    if "_batch_prec" in leaf:
        return "basic(batch)"
    if "_batch" in leaf or leaf.endswith("_jit"):
        return "batch/helper"
    return "surface/helper"


def _tightening_level(category: str, base_name: str, mode_profile: str) -> str:
    if base_name in CORE_TIGHTNESS:
        return CORE_TIGHTNESS[base_name]
    if category == "alternative":
        if "rigorous" in mode_profile or "adaptive" in mode_profile:
            return "implementation-specific mode-aware tightening"
        return "point-only or helper path"
    if category == "new":
        if "rigorous" in mode_profile or "adaptive" in mode_profile or "basic" in mode_profile:
            return "module-specific tightening or precision path"
        return "point-only or helper path"
    if "rigorous" in mode_profile or "adaptive" in mode_profile or "basic" in mode_profile:
        return "canonical interval/precision path"
    return "helper/no standalone tightening"


def _notes(category: str, leaf: str, module: str) -> str:
    if leaf in SPECIAL_NOTES:
        return SPECIAL_NOTES[leaf]
    if category == "alternative":
        if leaf.lower().startswith("bdg_"):
            return "Alternative implementation of the mathematical family sourced from BarnesDoubleGamma.jl; the `bdg_` prefix identifies lineage."
        if module == "double_gamma":
            return "Alternative implementation of the mathematical family sourced from BarnesDoubleGamma.jl; use the `bdg_`-prefixed public name."
        return "Alternative implementation of the canonical mathematical function; provenance prefix identifies lineage."
    if category == "new":
        return "New mathematical family without an Arb-like base-name collision in this repo."
    if module:
        return "Canonical Arb-like public symbol in this repo."
    return "Canonical top-level public export in this repo."


def _references(category: str, module: str, base_name: str) -> tuple[str, ...]:
    refs = ["docs/references/inventory/function_list.md"]
    if base_name in CORE_TIGHTNESS:
        refs.append("docs/reports/custom_core_status.md")
    elif category == "arb_like" and module in {"arb_core", "acb_core"}:
        refs.append("docs/reports/core_function_status.md")
    elif category == "alternative" and "boost" in module:
        refs.append("docs/implementation/external/boost_hypgeom.md")
    elif category == "alternative" and "cusf" in module.lower():
        refs.append("docs/implementation/external/cusf_compat.md")
    elif category == "new" and module:
        refs.append(f"docs/implementation/modules/{module}.md")
    return tuple(refs)


def build_entries() -> list[FunctionEntry]:
    points = point_functions()
    intervals = interval_functions()
    rows: list[FunctionEntry] = []
    for public_name in public_functions():
        leaf = _leaf_name(public_name)
        module = _module_name(public_name)
        base_name = _normalize_base_name(leaf)
        category = _category(leaf, module, base_name)
        modes = _mode_profile(public_name, leaf, base_name, points, intervals)
        rows.append(
            FunctionEntry(
                public_name=public_name,
                preferred_public_name=_preferred_public_name(leaf),
                base_name=base_name,
                category=category,
                lineage=_lineage(category, leaf, module),
                module=module or "top-level runtime export",
                four_modes=modes,
                tightening=_tightening_level(category, base_name, modes),
                references=_references(category, module, base_name),
                notes=_notes(category, leaf, module),
            )
        )
    seen = {row.public_name for row in rows}
    for entry in MANUAL_FUNCTION_ENTRIES:
        if entry.public_name not in seen:
            rows.append(entry)
    return rows


def _summarize_modes(values: set[str]) -> str:
    if "point|basic|adaptive|rigorous" in values:
        return "point|basic|adaptive|rigorous"
    ordered = []
    for token in ("point", "basic", "adaptive", "rigorous"):
        if any(token in value for value in values):
            ordered.append(token)
    if ordered:
        return "|".join(ordered)
    return "; ".join(sorted(values))


def _summarize_tightening(values: set[str]) -> str:
    priority = (
        "specialized rigorous dispatch complete",
        "implementation-specific mode-aware tightening",
        "module-specific tightening or precision path",
        "canonical interval/precision path",
        "point-only or helper path",
        "helper/no standalone tightening",
    )
    for item in priority:
        if item in values:
            return item
    return "; ".join(sorted(values))


def build_implementation_entries() -> list[ImplementationEntry]:
    points = point_functions()
    intervals = interval_functions()
    grouped: dict[tuple[str, str, str], list[FunctionEntry]] = {}
    for entry in build_entries():
        key = (entry.base_name, entry.category, entry.preferred_public_name)
        grouped.setdefault(key, []).append(entry)

    out: list[ImplementationEntry] = []
    for (base_name, category, preferred_name), rows in grouped.items():
        public_name = rows[0].public_name.rsplit(".", 1)[-1]
        modules = sorted({row.module for row in rows})
        lineages = sorted({row.lineage for row in rows})
        mode_values = {row.four_modes for row in rows}
        tightening_values = {row.tightening for row in rows}
        notes = sorted({row.notes for row in rows})
        out.append(
            ImplementationEntry(
                base_name=base_name,
                category=category,
                public_name=public_name,
                preferred_public_name=preferred_name,
                lineage="; ".join(lineages),
                module=", ".join(modules),
                four_modes=_summarize_modes(mode_values),
                tightening=_summarize_tightening(tightening_values),
                why="; ".join(notes[:2]) if len(notes) > 1 else notes[0],
            )
        )
    for base_name in sorted(points | intervals):
        key = (base_name, "arb_like", base_name)
        if key in grouped:
            continue
        modes = []
        if base_name in points:
            modes.append("point")
        if base_name in intervals:
            modes.extend(["basic", "adaptive", "rigorous"])
        mode_str = "|".join(dict.fromkeys(modes))
        out.append(
            ImplementationEntry(
                base_name=base_name,
                category="arb_like",
                public_name=base_name,
                preferred_public_name=base_name,
                lineage="Arb/FLINT-style canonical mathematical family",
                module="inventory-derived canonical surface",
                four_modes=mode_str,
                tightening="canonical interval/precision path" if base_name in intervals else "point-only canonical path",
                why="Synthetic canonical implementation row derived from the public point/interval inventory.",
            )
        )
    existing = {(row.base_name, row.category, row.preferred_public_name) for row in out}
    for row in MANUAL_IMPLEMENTATIONS:
        key = (row.base_name, row.category, row.preferred_public_name)
        if key not in existing:
            out.append(row)
    return sorted(out, key=lambda row: (row.base_name, row.category, row.preferred_public_name))


def render_policy() -> str:
    return POLICY_TEXT


def render_report(category: str, title: str) -> str:
    rows = [entry for entry in build_entries() if entry.category == category]
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        f"# {title}",
        "",
        "Generated from `arbplusjax.function_provenance`.",
        "",
        f"Summary: `entries={len(rows)}`.",
        "",
        "| public_name | preferred_public_name | base_name | lineage | module | four_modes | tightening | references | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        refs = ", ".join(f"`{ref}`" for ref in row.references)
        lines.append(
            f"| {row.public_name} | {row.preferred_public_name} | {row.base_name} | "
            f"{row.lineage} | `{row.module}` | {row.four_modes} | {row.tightening} | {refs} | {row.notes} |"
        )
    return "\n".join(lines) + "\n"


def render_registry_summary() -> str:
    entries = build_entries()
    total = len(entries)
    arb_like = sum(1 for row in entries if row.category == "arb_like")
    alternative = sum(1 for row in entries if row.category == "alternative")
    new = sum(1 for row in entries if row.category == "new")
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Function Provenance Registry",
        "",
        "Generated from `arbplusjax.function_provenance`.",
        "",
        f"Summary: `entries={total}`, `arb_like={arb_like}`, `alternative={alternative}`, `new={new}`.",
        "",
        "See the split reports for the actual registry tables:",
        "- `docs/reports/arb_like_functions.md`",
        "- `docs/reports/alternative_functions.md`",
        "- `docs/reports/new_functions.md`",
        "- `docs/reports/function_implementation_index.md`",
        "",
        "See `docs/function_naming.md` for the naming and provenance policy.",
        "",
    ]
    return "\n".join(lines)


def render_implementation_index() -> str:
    rows = build_implementation_entries()
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Function Implementation Index",
        "",
        "Generated from `arbplusjax.function_provenance`.",
        "",
        f"Summary: `implementations={len(rows)}`.",
        "",
        "| base_name | category | public_name | preferred_public_name | four_modes | tightening | why | lineage | module |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.base_name} | {row.category} | {row.public_name} | {row.preferred_public_name} | "
            f"{row.four_modes} | {row.tightening} | {row.why} | {row.lineage} | `{row.module}` |"
        )
    return "\n".join(lines) + "\n"


def render_lookup(base_name: str) -> str:
    key = base_name.strip().lower()
    matches = [row for row in build_implementation_entries() if row.base_name == key]
    lines = [f"Lookup: {base_name}", ""]
    if not matches:
        lines.append("No registry entries found.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| category | public_name | preferred_public_name | modes | tightening | why | lineage | module |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in matches:
        lines.append(
            f"| {row.category} | {row.public_name} | {row.preferred_public_name} | "
            f"{row.four_modes} | {row.tightening} | {row.why} | {row.lineage} | `{row.module}` |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "FunctionEntry",
    "ImplementationEntry",
    "POLICY_TEXT",
    "build_entries",
    "build_implementation_entries",
    "public_functions",
    "point_functions",
    "interval_functions",
    "render_policy",
    "render_report",
    "render_registry_summary",
    "render_implementation_index",
    "render_lookup",
]
