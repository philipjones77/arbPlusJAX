# Example Dense Matrix Surface Summary (cpu)

- backend: `cpu`
- benchmark_rows: `43`

## Comparison Slice

- `acb` / `dense_plan_prepare`: warm=2.30975e-05s
- `arb` / `dense_plan_prepare`: warm=3.73915e-05s
- `arb` / `direct_solve`: warm=5.8315e-05s
- `acb` / `cached_matvec`: warm=0.000101041s
- `acb` / `direct_solve`: warm=0.000215798s
- `arb` / `cached_matvec`: warm=0.000234222s
