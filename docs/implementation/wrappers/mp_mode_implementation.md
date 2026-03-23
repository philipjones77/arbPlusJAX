Last updated: 2026-02-25T03:51:38Z

# mp_mode

mpmath‑style precision wrapper for basic kernels.

## Behavior

`mp_mode` allows passing `dps` and automatically inflates bounds after midpoint evaluation. This is used for basic `*_prec` kernels to emulate mpmath’s `mp.dps` interface while preserving JAX JIT compatibility.
