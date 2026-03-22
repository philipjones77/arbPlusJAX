from __future__ import annotations

import importlib


_EXPORTS = {
    "IFJBarnesDoubleGammaDiagnostics": ".barnes_double_gamma_ifj",
    "barnesdoublegamma_ifj": ".barnes_double_gamma_ifj",
    "barnesdoublegamma_ifj_diagnostics": ".barnes_double_gamma_ifj",
    "log_barnesdoublegamma_ifj": ".barnes_double_gamma_ifj",
    "incomplete_gamma_lower_argument_derivative": ".derivatives",
    "incomplete_gamma_lower_derivative": ".derivatives",
    "incomplete_gamma_lower_parameter_derivative": ".derivatives",
    "incomplete_gamma_upper_argument_derivative": ".derivatives",
    "incomplete_gamma_upper_derivative": ".derivatives",
    "incomplete_gamma_upper_parameter_derivative": ".derivatives",
    "incomplete_gamma_lower": ".incomplete_gamma",
    "incomplete_gamma_lower_batch": ".incomplete_gamma",
    "incomplete_gamma_lower_point": ".incomplete_gamma",
    "incomplete_gamma_upper": ".incomplete_gamma",
    "incomplete_gamma_upper_batch": ".incomplete_gamma",
    "incomplete_gamma_upper_point": ".incomplete_gamma",
    "incomplete_gamma_upper_switched_point_with_fingerprint": ".incomplete_gamma_ad",
    "incomplete_gamma_upper_switched_z_jvp": ".incomplete_gamma_ad",
    "incomplete_gamma_upper_switched_z_vjp": ".incomplete_gamma_ad",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
