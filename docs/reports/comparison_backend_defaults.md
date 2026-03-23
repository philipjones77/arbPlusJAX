Last updated: 2026-03-23T00:00:00Z

# Comparison Backend Defaults

This report records the default comparison stack for each benchmark-harness function.

Policy:
- use `c_arb` as the default interval/enclosure reference whenever a C Arb/FLINT adapter exists
- use `mpmath` as the default high-precision point reference whenever available
- use `scipy` as the default float64 engineering parity reference whenever available
- keep `jax_point` as an internal point-mode comparison path rather than the primary external truth source

| function | interval default | high-precision default | float default | comparison order |
|---|---|---|---|---|
| `exp` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `log` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `sqrt` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `sin` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `cos` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `tan` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `sinh` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `cosh` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `tanh` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `gamma` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `erf` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `erfc` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `barnesg` | `none` | `mpmath` | `none` | `mpmath` |
| `besselj` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `bessely` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `besseli` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `besselk` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
| `CubesselK` | `c_arb` | `mpmath` | `scipy` | `c_arb`, `mpmath`, `scipy`, `jax_point` |
