# mp_mode

mpmath‑style precision wrapper for baseline kernels.

## Behavior

`mp_mode` allows passing `dps` and automatically inflates bounds after midpoint evaluation. This is used for baseline `*_prec` kernels to emulate mpmath’s `mp.dps` interface while preserving JAX JIT compatibility.
