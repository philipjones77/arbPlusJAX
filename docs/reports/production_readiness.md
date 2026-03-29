Last updated: 2026-03-29T00:00:00Z

# Production Readiness

This generated report summarizes the repo's production-readiness governance layer.

It is the current-state companion to:
- [production_readiness_standard.md](/docs/standards/production_readiness_standard.md)
- [release_packaging_standard.md](/docs/standards/release_packaging_standard.md)
- [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md)
- [release_governance_standard.md](/docs/standards/release_governance_standard.md)
- [security_supply_chain_standard.md](/docs/standards/security_supply_chain_standard.md)
- [operational_support_standard.md](/docs/standards/operational_support_standard.md)
- [capability_maturity_standard.md](/docs/standards/capability_maturity_standard.md)

Interpretation:
- `present`: the required Markdown and automation surfaces exist
- `partial`: some, but not all, required surfaces exist
- `missing`: the governed surfaces are not yet in place

| area | standards | status lane | automation / repo surface | readiness |
|---|---|---|---|---|
| `release and packaging` | [release_packaging_standard.md](/docs/standards/release_packaging_standard.md), [production_readiness_standard.md](/docs/standards/production_readiness_standard.md) | [release_packaging_todo.md](/docs/status/release_packaging_todo.md) | [build-dist.yml](/.github/workflows/build-dist.yml), [publish-release.yml](/.github/workflows/publish-release.yml), [pyproject.toml](/pyproject.toml) | `present` |
| `docs publishing` | [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md), [production_readiness_standard.md](/docs/standards/production_readiness_standard.md) | [docs_publishing_todo.md](/docs/status/docs_publishing_todo.md) | [docs-publish.yml](/.github/workflows/docs-publish.yml), [pyproject.toml](/pyproject.toml) | `present` |
| `release governance` | [release_governance_standard.md](/docs/standards/release_governance_standard.md) | [production_readiness_todo.md](/docs/status/production_readiness_todo.md) | [CHANGELOG.md](/CHANGELOG.md) | `present` |
| `security and supply chain` | [security_supply_chain_standard.md](/docs/standards/security_supply_chain_standard.md) | [security_supply_chain_todo.md](/docs/status/security_supply_chain_todo.md) | [SECURITY.md](/SECURITY.md), [dependency-audit.yml](/.github/workflows/dependency-audit.yml) | `present` |
| `operational support` | [operational_support_standard.md](/docs/standards/operational_support_standard.md) | [operational_support_todo.md](/docs/status/operational_support_todo.md) | [CONTRIBUTING.md](/CONTRIBUTING.md), [SUPPORT.md](/SUPPORT.md) | `present` |
| `capability and maturity reporting` | [capability_maturity_standard.md](/docs/standards/capability_maturity_standard.md) | [capability_maturity_todo.md](/docs/status/capability_maturity_todo.md) | `planned` | `present` |

## Packaging Extras

Current expected optional dependency groups from `pyproject.toml`:
- `compare`
- `docs`
- `dev`
- `bench`
- `release`
- `colab`

## Main Function-Category Closeout Snapshot

| category | current status |
|---|---|
| `1. Core Numeric Scalars` | `done` |
| `2. Interval / Box / Precision Modes` | `in_progress` |
| `3. Dense Matrix Functionality` | `done` |
| `4. Sparse / Block-Sparse / VBlock Functionality` | `in_progress` |
| `5. Matrix-Free / Operator Functionality` | `in_progress` |
| `6. Special Functions` | `done` |
| `7. Analytic / Algebraic / Domain Functionality` | `in_progress` |
| `8. API / Runtime / Metadata / Validation` | `in_progress` |

## Current Reading

- This report measures structure and governance presence, not full implementation quality.
- Category statuses above come from [todo.md](/docs/status/todo.md) and remain the canonical implementation-state signal.
- Production claims should rely on this report together with the category-specific reports and standards verification surfaces.
