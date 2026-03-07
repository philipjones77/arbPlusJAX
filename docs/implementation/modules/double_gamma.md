Last updated: 2026-02-25T03:51:38Z

# Barnes Double Gamma and Double Sine

This module ports the BarnesDoubleGamma.jl algorithms into the JAX stack, following the existing basic / rigorous / adaptive conventions.

## Functions

- `bdg_log_barnesdoublegamma(z, τ)` and `bdg_barnesdoublegamma(z, τ)` implement the logarithm and value of the Barnes double gamma / Barnes G with parameter `τ`.
- `bdg_log_barnesgamma2(w, β)` and `bdg_barnesgamma2(w, β)` implement the `Γ_2(w, β)` function.
- `bdg_log_normalizeddoublegamma(w, β)` and `bdg_normalizeddoublegamma(w, β)` implement the normalized double gamma `Γ_β(w)`.
- `bdg_double_sine(z, b)` implements the double sine `S_b(z)` (DSine in the Julia reference).

Real interval versions:

- `bdg_interval_log_barnesdoublegamma`, `bdg_interval_barnesdoublegamma`
- `bdg_interval_log_barnesgamma2`, `bdg_interval_barnesgamma2`
- `bdg_interval_log_normalizeddoublegamma`, `bdg_interval_normalizeddoublegamma`

Complex box versions:

- `bdg_complex_log_barnesdoublegamma`, `bdg_complex_barnesdoublegamma`
- `bdg_complex_log_barnesgamma2`, `bdg_complex_barnesgamma2`
- `bdg_complex_log_normalizeddoublegamma`, `bdg_complex_normalizeddoublegamma`
- `bdg_complex_double_sine`

Mode-dispatched wrappers (basic / rigorous / adaptive):

- `bdg_interval_*_mode` variants for the interval-valued real wrappers
- `bdg_complex_*_mode` variants for the box-valued complex wrappers

## Core formulas

We follow the product-formula approach used in BarnesDoubleGamma.jl (see references below). The main components are:

- Modular coefficients `a(τ)`, `b(τ)` computed from integrals `C(τ)` and `D(τ)`.
- A finite log-gamma summation with precomputed `loggamma`, `digamma`, `trigamma` terms.
- A remainder correction using polynomial `P_n` terms and a residual `R_{M,N}` series.

Definitions (schematic):

- `log G(z; τ) ≈ log_Barnes_GN(z, τ) + z^3 * R_{M,N}(z, τ)`
- `log Γ_2(w, β) = w * c1 + (w/2 * (w + c2) + 1) * log β - log G(w/β; τ)`
- `log Γ_β(w) = log Γ_2(w, β) - log Γ_2((β + 1/β)/2, β)`

## JAX implementation notes

- basic functions use midpoint evaluation.
- Rigorous and adaptive versions are implemented in `ball_wrappers.py` using Lipschitz/gradient bounding consistent with the rest of the system.
- We fix the truncation lengths based on `prec_bits` and use deterministic trapezoidal integration for `C(τ)` / `D(τ)`.

## Code references

- `src/arbplusjax/double_gamma.py`
- `src/arbplusjax/ball_wrappers.py`
- `src/arbplusjax/barnesg.py` (reused complex log-gamma)

## References

- BarnesDoubleGamma.jl (local reference): `C:\Users\phili\OneDrive\Documents\GitHub\BarnesDoubleGamma.jl`
- The product-formula implementation notes in `src/double_gamma_product_formula.jl` (within the above repository).
- arXiv:2208.13876 (as cited in the Julia reference source header).
