from __future__ import annotations

"""Alexanian--Kuznetsov paper-lineage Barnes and double-gamma families.

This module exposes the Barnes/double-gamma family under the `shahen_` prefix,
using the Alexanian--Kuznetsov paper lineage as the provenance label.

Current implementation note:
- the mathematical intent is the same as the existing `bdg_*` family
- the current `shahen_*` surface delegates to the same JAX kernels and mode
  wrappers already used by `bdg_*`
- this keeps provenance explicit without duplicating numerically identical code

Provenance:
- classification: alternative
- module lineage: Alexanian--Kuznetsov paper-derived implementation family
- naming policy: see docs/function_naming.md
- registry report: see docs/reports/function_implementation_index.md
"""

from . import double_gamma as _bdg

PROVENANCE = {
    "classification": "alternative",
    "base_names": ("barnesdoublegamma", "barnesgamma2", "normalizeddoublegamma", "double_sine"),
    "preferred_prefix": "shahen",
    "module_lineage": "Alexanian--Kuznetsov paper-derived implementation family; current implementation delegates to bdg_* because the mathematical construction is the same in intent",
    "naming_policy": "docs/function_naming.md",
    "registry_report": "docs/reports/function_implementation_index.md",
}

shahen_log_barnesdoublegamma = _bdg.bdg_log_barnesdoublegamma
shahen_barnesdoublegamma = _bdg.bdg_barnesdoublegamma
shahen_log_barnesgamma2 = _bdg.bdg_log_barnesgamma2
shahen_barnesgamma2 = _bdg.bdg_barnesgamma2
shahen_log_normalizeddoublegamma = _bdg.bdg_log_normalizeddoublegamma
shahen_normalizeddoublegamma = _bdg.bdg_normalizeddoublegamma
shahen_double_sine = _bdg.bdg_double_sine

shahen_interval_log_barnesdoublegamma = _bdg.bdg_interval_log_barnesdoublegamma
shahen_interval_barnesdoublegamma = _bdg.bdg_interval_barnesdoublegamma
shahen_interval_log_barnesgamma2 = _bdg.bdg_interval_log_barnesgamma2
shahen_interval_barnesgamma2 = _bdg.bdg_interval_barnesgamma2
shahen_interval_log_normalizeddoublegamma = _bdg.bdg_interval_log_normalizeddoublegamma
shahen_interval_normalizeddoublegamma = _bdg.bdg_interval_normalizeddoublegamma

shahen_complex_log_barnesdoublegamma = _bdg.bdg_complex_log_barnesdoublegamma
shahen_complex_barnesdoublegamma = _bdg.bdg_complex_barnesdoublegamma
shahen_complex_log_barnesgamma2 = _bdg.bdg_complex_log_barnesgamma2
shahen_complex_barnesgamma2 = _bdg.bdg_complex_barnesgamma2
shahen_complex_log_normalizeddoublegamma = _bdg.bdg_complex_log_normalizeddoublegamma
shahen_complex_normalizeddoublegamma = _bdg.bdg_complex_normalizeddoublegamma
shahen_complex_double_sine = _bdg.bdg_complex_double_sine

shahen_interval_log_barnesdoublegamma_mode = _bdg.bdg_interval_log_barnesdoublegamma_mode
shahen_interval_barnesdoublegamma_mode = _bdg.bdg_interval_barnesdoublegamma_mode
shahen_interval_log_barnesgamma2_mode = _bdg.bdg_interval_log_barnesgamma2_mode
shahen_interval_barnesgamma2_mode = _bdg.bdg_interval_barnesgamma2_mode
shahen_interval_log_normalizeddoublegamma_mode = _bdg.bdg_interval_log_normalizeddoublegamma_mode
shahen_interval_normalizeddoublegamma_mode = _bdg.bdg_interval_normalizeddoublegamma_mode

shahen_complex_log_barnesdoublegamma_mode = _bdg.bdg_complex_log_barnesdoublegamma_mode
shahen_complex_barnesdoublegamma_mode = _bdg.bdg_complex_barnesdoublegamma_mode
shahen_complex_log_barnesgamma2_mode = _bdg.bdg_complex_log_barnesgamma2_mode
shahen_complex_barnesgamma2_mode = _bdg.bdg_complex_barnesgamma2_mode
shahen_complex_log_normalizeddoublegamma_mode = _bdg.bdg_complex_log_normalizeddoublegamma_mode
shahen_complex_normalizeddoublegamma_mode = _bdg.bdg_complex_normalizeddoublegamma_mode
shahen_complex_double_sine_mode = _bdg.bdg_complex_double_sine_mode

__all__ = [
    "PROVENANCE",
    "shahen_log_barnesdoublegamma",
    "shahen_barnesdoublegamma",
    "shahen_log_barnesgamma2",
    "shahen_barnesgamma2",
    "shahen_log_normalizeddoublegamma",
    "shahen_normalizeddoublegamma",
    "shahen_double_sine",
    "shahen_interval_log_barnesdoublegamma",
    "shahen_interval_barnesdoublegamma",
    "shahen_interval_log_barnesgamma2",
    "shahen_interval_barnesgamma2",
    "shahen_interval_log_normalizeddoublegamma",
    "shahen_interval_normalizeddoublegamma",
    "shahen_complex_log_barnesdoublegamma",
    "shahen_complex_barnesdoublegamma",
    "shahen_complex_log_barnesgamma2",
    "shahen_complex_barnesgamma2",
    "shahen_complex_log_normalizeddoublegamma",
    "shahen_complex_normalizeddoublegamma",
    "shahen_complex_double_sine",
    "shahen_interval_log_barnesdoublegamma_mode",
    "shahen_interval_barnesdoublegamma_mode",
    "shahen_interval_log_barnesgamma2_mode",
    "shahen_interval_barnesgamma2_mode",
    "shahen_interval_log_normalizeddoublegamma_mode",
    "shahen_interval_normalizeddoublegamma_mode",
    "shahen_complex_log_barnesdoublegamma_mode",
    "shahen_complex_barnesdoublegamma_mode",
    "shahen_complex_log_barnesgamma2_mode",
    "shahen_complex_barnesgamma2_mode",
    "shahen_complex_log_normalizeddoublegamma_mode",
    "shahen_complex_normalizeddoublegamma_mode",
    "shahen_complex_double_sine_mode",
]
