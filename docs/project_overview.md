Last updated: 2026-03-23T00:00:00Z

# Project Overview

arbPlusJAX is the active hardened JAX numerical-kernel workspace. The repository separates runtime code, conformance tests, benchmarks, examples, tooling, contracts, and documentation into stable top-level folders.

## Repo Root

- `src/`
- `tests/`
- `benchmarks/`
- `tools/`
- `docs/`
- `contracts/`
- `examples/`
- `experiments/`
- `outputs/`

## Docs Map

- governance: [docs/governance/README.md](/docs/governance/README.md)
- standards: [docs/standards/README.md](/docs/standards/README.md)
- notation: [docs/notation/README.md](/docs/notation/README.md)
- specs: [docs/specs/README.md](/docs/specs/README.md)
- objects: [docs/objects/README.md](/docs/objects/README.md)
- reports: [docs/reports/README.md](/docs/reports/README.md)
- status: [docs/status/README.md](/docs/status/README.md)
- theory: [docs/theory/README.md](/docs/theory/README.md)
- implementation: [docs/implementation/README.md](/docs/implementation/README.md)
- practical: [docs/practical/README.md](/docs/practical/README.md)

## Generation Rule

Docs landing pages, report indexes, status indexes, and current repo mapping are generated and should be refreshed through `python tools/check_generated_reports.py` before commit/push.
