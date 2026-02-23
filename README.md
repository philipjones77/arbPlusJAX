# arbPlusJAX

Implementation of arb for JAX.

## Layout

- `src/arbplusjax/`: JAX implementation (primary runtime).
- `tests/`: chassis + parity tests.
- `tools/`: parity and benchmark scripts.
- `docs/`: theory and implementation notes.
- `examples/`, `notebooks/`: usage and experiments.
- `results/`: benchmark outputs (what ran and when).

## Install (editable)

```powershell
python -m pip install -e .
```

## Tests

Chassis tests:

```powershell
python -m pytest tests -q -m "not parity"
```

Parity tests (requires Arb C libs):

```powershell
$env:ARBPLUSJAX_RUN_PARITY = "1"
python -m pytest tests -q -m parity
```

## Parity tools

The scripts in `tools/` load Arb reference libraries built in the Arb workspace or archived under `stuff/`.

## Notes

- JAX is the primary implementation. The Arb workspace is used for parity and benchmarking.
- Some advanced helper routines are approximations; see `docs/` for details.
