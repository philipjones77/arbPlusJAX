Last updated: 2026-03-14T12:00:00Z

# Experiments

This directory is reserved for larger-scale exploratory work and experimental
implementations, usually grown out of `example_*.ipynb` notebooks.

Policy:

- [experiment_layout_standard.md](/home/phili/projects/arbplusJAX/docs/standards/experiment_layout_standard.md)

Rules:

- each subfolder represents one experiment
- experiment subfolders should use human-readable names
- notebooks and scripts live at the root of that experiment subfolder
- `inputs/` and transient artifact subfolders should not be under source control
- experiment-local `outputs/` are scratch or pre-promotion material
- semi-permanent retained artifacts belong under the repo-root `outputs/` tree

Typical shape:

```text
experiments/<human_name>/
  README.md
  notebook.ipynb
  run.py
  inputs/
  outputs/
```

Benchmark code still belongs in `benchmarks/`.
Experiments may call benchmarks and summarize their outputs, but benchmark CLI
entrypoints do not move here.

Experiments may read benchmark run artifacts from `benchmarks/results/`.

Experiment-local outputs may stay inside the experiment tree while work is in
progress, but semi-permanent retained artifacts should be promoted into a named
subfolder under the repo-root `outputs/` tree.

Current retained experiment root:

- `experiments/benchmarks/`
  - canonical retained benchmark experiment area
  - keeps benchmark notebooks, support code, and experiment-local working outputs
