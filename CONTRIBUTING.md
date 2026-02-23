# Contributing

## Development setup

```powershell
python -m pip install -e .
```

## Tests

```powershell
python -m pytest tests -q -m "not parity"
```

Parity tests (requires Arb C libs from the Arb workspace):

```powershell
$env:arbplusjax_RUN_PARITY = "1"
python -m pytest tests -q -m parity
```

## Guidelines

- Keep JAX as the primary implementation. Use the Arb workspace only for parity/benchmarks.
- Preserve interval/box ordering and outward rounding semantics.
- Add tests for any new kernel: jit, batch/vectorization, and grad on smooth domains.
