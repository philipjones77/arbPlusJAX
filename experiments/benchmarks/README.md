# Benchmark Experiments

This folder holds notebook-driven benchmark experiments that are broader than the
lightweight CLI benchmark smoke checks in `benchmarks/`.

Current notebook:

- `elementary_core_backend_sweep.ipynb`

It compares:

- arbPlusJAX core interval kernels
- arbPlusJAX elementary helpers
- optional external references when available:
  - FLINT source install discovery
  - Arb/FLINT C refs
  - mpmath
  - SciPy
  - JAX / JAX SciPy
  - Mathematica via `wolframscript`

Reference backends are auto-discovered from the standard local prefix:

- `~/.local/opt/arbplusjax_refs/flint/current`
- `~/.local/opt/arbplusjax_refs/boost/current`

The optional shell bootstrap is:

```bash
source tools/source_reference_env.sh
```

Optional JAX diagnostics are opt-in through
`arbplusjax.jax_diagnostics.JaxDiagnosticsConfig` or environment variables:

- `ARBPLUSJAX_JAX_DIAGNOSTICS_ENABLED=1`
- `ARBPLUSJAX_JAX_DIAGNOSTICS_JAXPR=1`
- `ARBPLUSJAX_JAX_DIAGNOSTICS_HLO=1`
- `ARBPLUSJAX_JAX_DIAGNOSTICS_TRACE=1`
- `ARBPLUSJAX_JAX_DIAGNOSTICS_TRACE_DIR=/path/to/traces`
