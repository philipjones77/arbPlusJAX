# Architecture

arbPlusJAX is a JAX-first implementation of key Arb functionality with three execution modes:

- Baseline: fast midpoint-based kernels (with optional interval wrappers).
- Adaptive: sampling/Jacobian-based inflation to widen bounds without recompilation.
- Rigorous: ball/interval wrappers that enforce outward rounding and explicit remainder bounds.

## Core Layout

- `src/arbplusjax/`: primary implementation modules and wrappers.
- `src/arbplusjax/hypgeom.py`: special functions, series helpers, and interval logic.
- `src/arbplusjax/*_wrappers.py`: mode dispatch (`baseline|adaptive|rigorous`) for kernels.
- `src/arbplusjax/ball_wrappers.py`: rigorous ball semantics using Arb-style outward rounding.
- `src/arbplusjax/double_interval.py`: interval arithmetic utilities.
- `src/arbplusjax/acb_*` and `src/arbplusjax/arb_*`: complex/real module families.

## Execution Modes

Mode dispatch is centralized in `wrappers_common.py` and used by `*_wrappers.py`.

- `baseline`: direct kernel result (intervals where the kernel already produces them).
- `adaptive`: expands bounds using sampling/Jacobian-based estimates.
- `rigorous`: uses ball wrappers or series tail bounds to ensure containment.

## Testing

- `tests/*_chassis.py`: shape, vectorization, and AD-path smoke checks.
- `tests/*_parity.py`: compare against Arb C reference libraries.
- `tests/test_hypgeom_completeness.py`: coverage and helper consistency.

## Results and Benchmarking

- `results/`: benchmark logs and test runs, including timestamps.
- `tools/`: scripts for comparisons and audits.

## Archived Migration

- `stuff/migration/`: prior migration workspace and C reference builds for parity.
