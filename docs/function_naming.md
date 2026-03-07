Last updated: 2026-03-07T00:00:00Z

# Function Naming Policy

## Model

- `arb_like`: canonical Arb/FLINT-style public mathematical surface. These names remain unprefixed.
- `alternative`: same mathematical target as an Arb-like function, but from a different implementation lineage. These names must use a provenance prefix.
- `new`: mathematical families without an Arb/FLINT-style canonical base name in this repo. These use a descriptive family name and do not need an alternative-prefix migration.

## Rules

- Public naming is based on the mathematical function name, not the Python filename.
- Alternative implementations must use `prefix_<base_name>`.
- The prefix is chosen when the function family is introduced and should reflect provenance, not a vague label like `custom`.
- No prefix implies the canonical Arb-like public function for this repo.
- If a function is intended to become part of the canonical Arb-like public surface, it stays unprefixed and must meet the same four-mode and tightening expectations as the rest of that surface.
- If a function is not canonical and implements the same mathematical target as an Arb-like name, it should not be placed in the canonical namespace.

## Examples

- Canonical: `besselk`
- Alternative: `cuda_besselk`, `boost_besselk`
- New family: `modular_j`

## Current repo intent

- `arb_core` / `acb_core` define the canonical Arb-like public surface for this repo, including approved Arb-like extensions.
- External-lineage implementations such as Boost- or CUDA-lineage code should be prefixed in the public API.
- Python module names may remain implementation-oriented; the policy applies to public function names.
