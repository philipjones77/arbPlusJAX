Last updated: 2026-03-26T00:00:00Z

# Startup Compile Repo Template

Use this as the implementation template when carrying the startup-compile
policy into another JAX repo.

Primary policy:

- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_compile_playbook_standard.md](/docs/standards/startup_compile_playbook_standard.md)

## Required Deliverables

Every adopting repo should land these artifacts:

1. one startup-compile policy document
2. one rollout/runbook document
3. one cache/bootstrap helper or canonical environment path
4. one representative startup-probe set with retained artifacts
5. one CI check that enforces startup budgets

## Required Runtime Decisions

For each hot family, record:

- canonical public entrypoint
- stable-shape contract:
  - fixed batch
  - `pad_to`
  - shape bucket
  - prepare/apply
- JIT owner layer
- compile-relevant static controls
- warmup eligibility
- expected process model:
  - long-lived service
  - notebook
  - CLI
  - batch job

## Required Environment Policy

Canonical environment requirements:

```bash
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR=/path/to/shared/cache
```

Recommended repo helper responsibilities:

- choose a default cache path
- respect user override
- expose a warmup toggle
- expose cold-versus-warm benchmark mode

## Required Probe Matrix

At minimum, add probes for:

- one point-family hot path
- one matrix or sparse cached-apply hot path
- one provider or alternative backend family when applicable

Each probe should report:

- import time
- backend init time
- first compile plus first call
- steady call
- prepare cost when applicable

## Required CI Policy

CI should fail when:

- a representative probe no longer compiles
- cold compile latency exceeds budget
- warm latency exceeds budget
- recompile behavior regresses on changed shape or static controls

## Migration Sequence

1. audit current hot entrypoints
2. define stable-shape contracts
3. centralize JIT ownership
4. enable persistent cache in canonical launchers
5. add warmup for top hot kernels
6. add startup probes and budgets
7. update examples so stable-shape calling is the default teaching path

## arbPlusJAX Example Mapping

This repo already demonstrates the template with:

- stable-shape point and matrix paths using `pad_to`
- centralized binder-style compiled entrypoints in the public API
- startup probes under [benchmarks](/benchmarks)
- import and first-use inventories under [docs/reports](/docs/reports)
- import-tier and first-use budgets in [import_tiers.py](/src/arbplusjax/import_tiers.py)
