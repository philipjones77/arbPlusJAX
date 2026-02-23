# arb_core

Real interval core utilities. These are the primary interval arithmetic kernels used throughout the library.

## Representation

An `arb` value is stored as:
```
[lo, hi]
```
with outward rounding provided by `double_interval`.

## Core formulas

- Addition/subtraction/multiplication/division are interval operations.
- `log`, `sqrt`, `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` have analytic interval bounds in rigorous mode (monotonicity and critical point checks).

## Precision semantics

`*_prec` kernels compute the midpoint and then round outward to `prec_bits`.
Rigorous and adaptive modes are dispatched by `core_wrappers`.
