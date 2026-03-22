Last updated: 2026-03-22T00:00:00Z

# Benchmark Results

Canonical benchmark run artifacts live under `benchmarks/results/`.

Structure rules:

- each run gets its own named subfolder
- ad hoc sweeps should use `run_<timestamp>/`
- named benchmark/profile/example runs may use a stable human-readable run name
- benchmark outputs must not overwrite prior runs in place

Typical contents:

- `run_<timestamp>/`
- `example_<name>_<cpu|gpu>/`
- `profile_<timestamp>/`
- compile-probe or backend-comparison subfolders

Derived diagnostics and curated summaries may still be written under:

- `experiments/benchmarks/outputs/`

Top-level `results/` should not exist.
