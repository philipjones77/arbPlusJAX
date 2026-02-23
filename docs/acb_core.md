# acb_core

Complex ball core utilities built on real interval arithmetic. This is the foundation for complex operations across the library.

## Representation

An `acb` value is stored as a box:
```
[re_lo, re_hi, im_lo, im_hi]
```
where each component is an outward‑rounded interval from `double_interval`.

## Core formulas

- Addition and subtraction are intervalwise per component.
- Multiplication uses interval arithmetic:
\[
(a+ib)(c+id) = (ac-bd) + i(ad+bc)
\]
- Division uses:
\[
\frac{a+ib}{c+id} = \frac{(a+ib)(c-id)}{c^2+d^2}
\]
with interval propagation for numerator and denominator.
- `log`, `sqrt`, `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` include analytic interval formulas in rigorous mode via `core_wrappers`.

## Precision semantics

`*_prec` kernels compute midpoint results and then apply outward rounding. Rigorous and adaptive bounds are provided by `core_wrappers` using analytic interval propagation and branch‑cut widening for `log` and polar `sqrt`.
