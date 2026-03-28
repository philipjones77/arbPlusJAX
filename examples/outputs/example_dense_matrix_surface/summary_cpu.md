# Example Dense Matrix Surface Summary (cpu)

- backend: `cpu`
- benchmark_rows: `43`

## Comparison Slice

- `arb` / `dense_plan_prepare`: warm=3.67805e-05s
- `acb` / `dense_plan_prepare`: warm=5.1322e-05s
- `acb` / `direct_solve`: warm=0.000139848s
- `acb` / `cached_matvec`: warm=0.000202701s
- `arb` / `direct_solve`: warm=0.000266315s
- `arb` / `cached_matvec`: warm=0.000558475s
