Last updated: 2026-03-07T00:00:00Z

# shahen_double_gamma

## Role

`shahen_double_gamma` is the Alexanian--Kuznetsov paper-lineage alternative implementation family for:
- Barnes double gamma
- BarnesGamma2
- normalized double gamma
- double sine

It is an alternative family, not a new mathematical family. The base names stay the same and the `shahen_` prefix carries the provenance.

## Current intent relationship to `bdg_*`

The current `shahen_*` family has the same mathematical intent as the existing `bdg_*` family.

Current implementation state:
- `shahen_*` delegates to the existing `bdg_*` kernels and mode wrappers
- this makes the family immediately testable against the rest of the repo
- provenance remains distinct even though the current numeric implementation path is shared

So:
- mathematically intended target: same as `bdg_*`
- current runtime implementation: same code path as `bdg_*`
- provenance label: different

## Functions

Point:
- `shahen_log_barnesdoublegamma`
- `shahen_barnesdoublegamma`
- `shahen_log_barnesgamma2`
- `shahen_barnesgamma2`
- `shahen_log_normalizeddoublegamma`
- `shahen_normalizeddoublegamma`
- `shahen_double_sine`

Real interval/basic:
- `shahen_interval_log_barnesdoublegamma`
- `shahen_interval_barnesdoublegamma`
- `shahen_interval_log_barnesgamma2`
- `shahen_interval_barnesgamma2`
- `shahen_interval_log_normalizeddoublegamma`
- `shahen_interval_normalizeddoublegamma`

Complex box/basic:
- `shahen_complex_log_barnesdoublegamma`
- `shahen_complex_barnesdoublegamma`
- `shahen_complex_log_barnesgamma2`
- `shahen_complex_barnesgamma2`
- `shahen_complex_log_normalizeddoublegamma`
- `shahen_complex_normalizeddoublegamma`
- `shahen_complex_double_sine`

Mode-dispatched:
- `shahen_interval_*_mode`
- `shahen_complex_*_mode`

## Notes

- Because this family currently delegates to `bdg_*`, it inherits the same current engineering strengths and limitations.
- In particular, it is available in all the same modes now, but it is not yet a second independent numerical kernel.
- If later paper-specific differences are implemented, this family is the correct place to put them.
