Last updated: 2026-03-22T00:00:00Z

# Repo Organization By Coverage Categories

## Purpose

This report defines the default way to view and organize the repo at the
top level.

The canonical top-level organization for this repo is the category structure in
[test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).

That matrix is not only a testing breakdown. It is the default repo
functionality map.

## Default Repo Categories

Use these as the primary top-level categories across docs, reports, examples,
benchmarks, and implementation planning:

1. Core Numeric Scalars
2. Interval / Box / Precision Modes
3. Dense Matrix Functionality
4. Sparse / Block-Sparse / VBlock Functionality
5. Matrix-Free / Operator Functionality
6. Special Functions

Helper and infrastructure categories should also follow the supporting sections
of the same matrix rather than inventing unrelated top-level taxonomies.

## Rule

When a repo-wide artifact needs a primary grouping, it should default to these
categories unless there is a strong reason not to.

Examples:

- `docs/status/todo.md` should track tranche work in this order
- `docs/reports/*.md` should summarize progress and gaps against these
  categories
- `examples/` notebook coverage should be described in terms of these
  categories
- `benchmarks/` inventories should map benchmark groups back to these
  categories
- production-readiness reviews should name the category they are closing

## Why This Is The Default Lens

This repo is broad enough that module-by-module or provenance-by-provenance
views are too low level to serve as the main organizing structure.

The coverage categories work better because they align:

- implementation surfaces
- public APIs
- tests
- benchmarks
- examples
- production hardening work

That makes them the cleanest common language for repo planning and review.

## Current Immediate Priority

The current immediate focus remains:

- `1. Core Numeric Scalars`

That category should therefore be treated as the active tranche for CPU
completion and notebook/example hardening until it is explicitly marked done in
status tracking.

## Reference

Canonical source:

- [test_coverage_matrix.md](/docs/status/test_coverage_matrix.md)
