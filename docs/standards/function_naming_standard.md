Last updated: 2026-03-13T00:00:00Z

# Function Naming Policy

## Model

- `arb_like`: canonical Arb/FLINT-style public mathematical surface. These names remain unprefixed.
- `alternative`: same mathematical target as an Arb-like function, but from a different implementation lineage. These should be separable by implementation metadata, module lineage, registry entries, or explicit dispatch controls, not by changing the mathematical public name.
- `new`: mathematical families without an Arb/FLINT-style canonical base name in this repo. These use a descriptive family name and do not need an alternative-prefix migration.

## Rules

- Public naming is based on the mathematical function name, not the Python filename.
- The canonical public function name should stay tied to the mathematical target whenever the repo intends a single user-facing function family.
- Alternative implementations should be separated by one or more of:
  - implementation/module lineage
  - explicit runtime `impl=` or backend selection
  - provenance registry and reports
  - internal/helper names that are not the primary public math surface
- Provenance prefixes are still acceptable for clearly secondary helper surfaces or compatibility layers, but they should not be the default way users discover the main mathematical function.
- No prefix implies the canonical Arb-like public function for this repo.
- If a function is intended to become part of the canonical Arb-like public surface, it stays unprefixed and must meet the same four-mode and tightening expectations as the rest of that surface.
- If multiple implementations exist for the same mathematical target, prefer one public name plus implementation selection over multiple provenance-prefixed public names.

## Examples

- Canonical: `besselk`
- Canonical name with implementation selection: `besselk(..., impl="cuda")`, `besselk(..., impl="boost")`
- Secondary compatibility helper: `boost_hypergeometric_1f1` may still exist internally or as a compatibility layer, but it should not define the long-term main public naming direction for the mathematical family
- New family: `modular_j`

## Current repo intent

- `arb_core` / `acb_core` define the canonical Arb-like public surface for this repo, including approved Arb-like extensions.
- External-lineage implementations such as Boost- or CUDA-lineage code should remain visible in provenance records, module layout, and explicit implementation selection.
- Python module names may remain implementation-oriented; the policy applies to user-facing public math names.
- Long-term preferred direction:
  - one public mathematical name
  - multiple selectable implementations underneath it
  - provenance retained in docs, reports, and registry metadata
