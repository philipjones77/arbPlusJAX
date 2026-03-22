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
- retained `outputs/` subfolders are for important artifacts and should be kept under source control via the repo's large-artifact storage path

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
