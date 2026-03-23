Last updated: 2026-03-16T00:00:00Z

# Optional Soft Ops

## Purpose

`soft_ops` and `soft_types` provide a narrow JAX-native subsystem for differentiable relaxations of a few discontinuous operations.

This subsystem is intentionally optional.

That means:

- it is not part of the default Arb-like runtime surface,
- it is not mixed into `point`, `basic`, `adaptive`, or `rigorous` interval modes,
- it is not required by any existing matrix, sparse, or special-function path,
- users opt into it explicitly by importing the modules directly.

## Modules

- [soft_ops.py](/src/arbplusjax/soft_ops.py)
- [soft_types.py](/src/arbplusjax/soft_types.py)

Current functions:

- `grad_replace`
- `st`
- `soft_sign`
- `soft_heaviside`
- `soft_clip`
- `soft_where`
- `soft_argmax`
- `soft_take_along_axis`
- `soft_top_k`

Current types:

- `SoftBool`
- `SoftIndex`

## Design boundary

These are AD-oriented surrogate objects, not enclosure objects.

So they should be thought of as:

- JAX differentiation helpers,
- optimization utilities,
- soft-selection primitives,

and not as part of the rigorous arithmetic story.

## Import policy

Use them explicitly:

```python
from arbplusjax.soft_ops import soft_argmax, soft_where
from arbplusjax.soft_types import SoftIndex
```

Do not treat them as canonical top-level runtime exports.

## Current intent

This subsystem exists to support:

- straight-through estimators,
- gradient replacement patterns,
- differentiable selection in optimization workflows,

without forcing those semantics onto the rest of the repository.
