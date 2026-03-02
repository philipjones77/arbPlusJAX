Last updated: 2026-03-01T00:00:00Z

# Boost.Math Hypergeometric Inventory

Source inspected:

- `C:\Users\phili\OneDrive\Documents\GitHub\math`
- Header root: `include/boost/math/special_functions`

## Public hypergeometric APIs currently available

From `math_fwd.hpp` and top-level hypergeometric headers:

1. `boost::math::hypergeometric_1F0(a, z)`
2. `boost::math::hypergeometric_0F1(b, z)`
3. `boost::math::hypergeometric_2F0(a1, a2, z)`
4. `boost::math::hypergeometric_1F1(a, b, z)`
5. `boost::math::hypergeometric_pFq(aj, bj, z, ...)`
6. `boost::math::hypergeometric_pFq_precision(aj, bj, z, digits10, ...)`

Top-level headers present:

- `hypergeometric_1F0.hpp`
- `hypergeometric_0F1.hpp`
- `hypergeometric_2F0.hpp`
- `hypergeometric_1F1.hpp`
- `hypergeometric_pFq.hpp`

## Detail/internal hypergeometric implementations present

Boost includes internal implementations for additional forms in `detail/*` that are not exposed as standalone public top-level API headers:

- `hypergeometric_2F1_*` (generic series / CF / Pade / rational helpers)
- `hypergeometric_1F2_*` (generic series / CF helpers)
- Multiple `1F1` region-specific algorithms:
  - asymptotic expansions
  - recurrence-on-`a`/`b`/`z`
  - Kummer reflection logic
  - ratio-based and scaled series variants

## Notes for arbPlusJAX planning

- Treat the public set above as the first Boost-derived hypergeometric implementable list.
- `2F1` exists in Boost internals, but does not appear as a dedicated top-level public header in this checkout; if needed, map via `pFq` or mirror Boost detail strategy.
