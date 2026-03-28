Last updated: 2026-03-27T00:00:00Z

# Notation

Status: authoritative
Version: v1.0

## 0. Preamble

### 0.1 Scope

This document is the authoritative notation bridge for arbPlusJAX.

It defines the mathematical symbols and code-facing names that recur across:

- interval and box arithmetic
- mode semantics
- dense, sparse, block-sparse, variable-block sparse, and matrix-free operators
- transform and special-function methodology
- production calling controls such as `prec_bits`, `dps`, `pad_to`, and `chunk_size`

This file does not replace detailed theory or implementation notes. It gives
the common symbol table those documents should reuse.

### 0.2 Authority Rule

Symbols defined here should not be silently redefined elsewhere in the docs
tree. Other documents may introduce temporary local names only when they are
explicitly file-local and non-authoritative.

### 0.3 Code-Name Rule

When a canonical ASCII name is useful, it follows the repo style:

- keep the existing symbol capitalization where practical
- TeX-like subscripts map to `_`
- superscripts map to `__`
- Greek names are spelled out in ASCII

Examples:

- $d_{\mathcal{M}}$ -> `d_M`
- $\sigma^2$ -> `sigma__2`
- $\tau$ -> `tau`

## 1. Domains, Values, and Enclosures

- `symbol`: $\mathcal{D}$
  - `ascii_name`: `D`
  - `meaning`: generic domain of evaluation
  - `notes`: used when the specific real, complex, matrix, or operator domain is not the main point

- `symbol`: $x, y, z$
  - `ascii_name`: `x`, `y`, `z`
  - `meaning`: generic scalar inputs
  - `notes`: may be real or complex depending on the surface

- `symbol`: $A, B$
  - `ascii_name`: `A`, `B`
  - `meaning`: generic matrix inputs
  - `notes`: used for dense, sparse, block-sparse, and operator-adapted matrices

- `symbol`: $v, w$
  - `ascii_name`: `v`, `w`
  - `meaning`: generic vectors or right-hand sides

- `symbol`: $[l, u]$
  - `ascii_name`: `interval(lo, hi)`
  - `meaning`: real interval with lower bound $l$ and upper bound $u$
  - `notes`: corresponds to repo interval arrays represented through `double_interval`

- `symbol`: $a + ib$
  - `ascii_name`: `complex midpoint`
  - `meaning`: ordinary complex midpoint value

- `symbol`: $( [l_r, u_r], [l_i, u_i] )$
  - `ascii_name`: `acb_box`
  - `meaning`: complex box with separate real and imaginary interval parts
  - `notes`: this is the repo’s standard complex enclosure object

## 2. Mode and Precision Semantics

- `symbol`: `point`
  - `ascii_name`: `mode="point"`
  - `meaning`: midpoint-style or helper-path evaluation without interval tightening

- `symbol`: `basic`
  - `ascii_name`: `mode="basic"`
  - `meaning`: baseline enclosure/tightening mode

- `symbol`: $\mathcal{S}_{\mathrm{point}}(f)$
  - `ascii_name`: `point surface of f`
  - `meaning`: the public point-mode evaluation surface exposed for a function or family

- `symbol`: $\mathcal{S}_{\mathrm{basic}}(f)$
  - `ascii_name`: `basic surface of f`
  - `meaning`: the public baseline enclosure/tightening surface exposed for a function or family

- `symbol`: $\mathcal{D}_f$
  - `ascii_name`: `diagnostics payload`
  - `meaning`: optional diagnostics object or metadata-bearing helper surface attached to a public function family
  - `notes`: used in reports, notebooks, and theory notes when a public family exposes `*_with_diagnostics` or named diagnostics records

- `symbol`: $\partial_x f,\ \partial_\theta f$
  - `ascii_name`: `argument-direction AD`, `parameter-direction AD`
  - `meaning`: differentiation through the main evaluation variable and through a continuous family parameter, respectively
  - `notes`: both directions are tracked separately in the repo’s AD audits, benchmarks, and notebooks

- `symbol`: `adaptive`
  - `ascii_name`: `mode="adaptive"`
  - `meaning`: mode that may increase work or inflation policy based on the local regime

- `symbol`: `rigorous`
  - `ascii_name`: `mode="rigorous"`
  - `meaning`: strongest currently exposed enclosure/tightening path for the family

- `symbol`: $p_{\mathrm{bits}}$
  - `ascii_name`: `prec_bits`
  - `meaning`: precision-like outward-rounding control used by many interval/box surfaces
  - `notes`: in arbPlusJAX this is not true arbitrary-precision arithmetic; it governs widening/tightening policy in float-based kernels

- `symbol`: $p_{\mathrm{dps}}$
  - `ascii_name`: `dps`
  - `meaning`: decimal-precision style control used on selected special-function and mpmath-influenced paths

- `symbol`: $\varepsilon_{\mathrm{tol}}$
  - `ascii_name`: `tol`
  - `meaning`: numerical tolerance for iterative or adaptive procedures

## 3. Service and Batch Controls

- `symbol`: $n$
  - `ascii_name`: `n`
  - `meaning`: primary size parameter
  - `notes`: matrix dimension, vector length, or number of modes depending on context

- `symbol`: $m$
  - `ascii_name`: `m`
  - `meaning`: secondary size parameter
  - `notes`: often used for numbers of samples, quadrature panels, or truncation lengths

- `symbol`: $B$
  - `ascii_name`: `batch_size`
  - `meaning`: number of requests in a batched call

- `symbol`: $B_{\mathrm{pad}}$
  - `ascii_name`: `pad_to`
  - `meaning`: padded batch multiple or stable padded batch size
  - `notes`: used to reduce recompiles from varying request shapes

- `symbol`: $B_{\mathrm{chunk}}$
  - `ascii_name`: `chunk_size`
  - `meaning`: chunk size used to split service traffic while keeping a stable public calling pattern

- `symbol`: $\tau_{\mathrm{cold}}, \tau_{\mathrm{warm}}, \tau_{\mathrm{recompile}}$
  - `ascii_name`: `cold_time_s`, `warm_time_s`, `recompile_time_s`
  - `meaning`: benchmark timings for first call, steady-state execution, and changed-shape/static-control recompilation

- `symbol`: $\mathcal{V}_{\mathrm{pb}}$
  - `ascii_name`: `point/basic verification ledger`
  - `meaning`: the checked-in family-level report that records point/basic surface counts and attached tests, benchmarks, notebooks, and diagnostics evidence
  - `notes`: in the current repo this is [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)

## 4. Matrix and Operator Notation

- `symbol`: $A v$
  - `ascii_name`: `matvec`
  - `meaning`: forward matrix-vector application

- `symbol`: $A^\top v$
  - `ascii_name`: `rmatvec`
  - `meaning`: transpose/reverse matrix-vector application
  - `notes`: for complex families this may correspond to transpose or adjoint depending on the family surface

- `symbol`: $\mathcal{P}_A$
  - `ascii_name`: `cached_prepare(A)`
  - `meaning`: cached plan prepared from matrix/operator data

- `symbol`: $\mathcal{P}_A(v)$
  - `ascii_name`: `cached_apply(plan, v)`
  - `meaning`: repeated application through a cached plan

- `symbol`: $\mathcal{O}_A$
  - `ascii_name`: `operator_plan`
  - `meaning`: matrix-free/operator-plan representation of a dense, sparse, block-sparse, or variable-block matrix

- `symbol`: $\mathcal{M}_{\mathrm{dense}}$
  - `ascii_name`: `dense matrix surface`
  - `meaning`: explicitly materialized dense matrix path

- `symbol`: $\mathcal{M}_{\mathrm{sparse}}$
  - `ascii_name`: `sparse matrix surface`
  - `meaning`: sparse storage path such as COO/CSR/BCOO

- `symbol`: $\mathcal{M}_{\mathrm{block}}$
  - `ascii_name`: `block-sparse matrix surface`
  - `meaning`: fixed block-structured sparse matrix path

- `symbol`: $\mathcal{M}_{\mathrm{vblock}}$
  - `ascii_name`: `vblock matrix surface`
  - `meaning`: variable-block sparse matrix path

- `symbol`: $\mathcal{M}_{\mathrm{free}}$
  - `ascii_name`: `matrix-free surface`
  - `meaning`: operator-plan / callback / Krylov-oriented execution path

## 5. Special-Function and Transform Controls

- `symbol`: $\nu$
  - `ascii_name`: `nu`
  - `meaning`: order parameter for Bessel-family functions

- `symbol`: $s$
  - `ascii_name`: `s`
  - `meaning`: gamma-family or Mellin-type parameter

- `symbol`: $\tau$
  - `ascii_name`: `tau`
  - `meaning`: Barnes/double-gamma period-like parameter

- `symbol`: $\omega$
  - `ascii_name`: `mode index`
  - `meaning`: transform/spectral mode or harmonic index when a more specific symbol is not introduced

- `symbol`: $\mathcal{Q}$
  - `ascii_name`: `quadrature rule`
  - `meaning`: quadrature configuration or integration policy

- `symbol`: $\mathcal{F}$
  - `ascii_name`: `transform`
  - `meaning`: Fourier/DFT/NUFFT-style transform map

## 6. Runtime and Portability Controls

- `symbol`: $\mathbb{T}_{32}, \mathbb{T}_{64}$
  - `ascii_name`: `float32`, `float64`
  - `meaning`: active real dtype policy

- `symbol`: $\mathbb{B}_{\mathrm{cpu}}, \mathbb{B}_{\mathrm{gpu}}$
  - `ascii_name`: `JAX_MODE=cpu|gpu`
  - `meaning`: requested backend class for execution
  - `notes`: current repo validation is CPU-first, but contracts and examples are written to remain GPU-portable

- `symbol`: $\mathcal{D}_{\mathrm{meta}}$
  - `ascii_name`: `metadata`
  - `meaning`: public metadata describing execution strategies, method choices, derivative status, and current hardening level

- `symbol`: $\mathcal{D}_{\mathrm{diag}}$
  - `ascii_name`: `diagnostics`
  - `meaning`: optional runtime diagnostics payload returned by public surfaces

## 7. Interpretation Rule

This notation file should be read together with:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [engineering_standard.md](/docs/standards/engineering_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)

The standards define how the symbols are exposed in production APIs.
The theory and implementation notes define the family-specific mathematics and
algorithms that instantiate the symbols.
