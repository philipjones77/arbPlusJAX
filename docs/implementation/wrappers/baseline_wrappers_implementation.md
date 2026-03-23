Last updated: 2026-02-25T03:51:38Z

# baseline_wrappers

basic wrappers for interval kernels. These expose `*_prec` functions with outward rounding and optional precision selection.

## Modes

Official modes are `point`, `basic`, `adaptive`, `rigorous`. The `baseline_wrappers` module implements the `basic` path and is used by the `*_mp` dispatchers.

## Purpose

Provide a consistent API for midpoint evaluation followed by outward rounding. No analytic bounds or sampling are applied here.
