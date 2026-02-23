# double_interval_wrappers

Mode dispatch for `double_interval` kernels.

## Fast interval kernels

Provides `fast_*_mode` wrappers that select between baseline, rigorous, and adaptive behavior for interval arithmetic where it is useful (e.g., add, mul, div, sqrt, log).
