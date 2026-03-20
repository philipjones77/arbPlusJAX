Last updated: 2026-03-18T00:00:00Z

# Bessel Family Methodology

This note records the formulas currently used for the higher-level Hankel and spherical Bessel surfaces in `arbplusjax.special.bessel`.

## 1. Hankel functions

The cylindrical Hankel functions are defined by

- `H_nu^(1)(z) = J_nu(z) + i Y_nu(z)`
- `H_nu^(2)(z) = J_nu(z) - i Y_nu(z)`

The current implementation uses these identities directly on top of the existing complex cylindrical `J` and `Y` kernels.

### Asymptotic form

For large `|z|`, the leading oscillatory approximation is

- `H_nu^(1)(z) ~ sqrt(2 / (pi z)) * exp(i (z - nu pi / 2 - pi / 4))`
- `H_nu^(2)(z) ~ sqrt(2 / (pi z)) * exp(-i (z - nu pi / 2 - pi / 4))`

This is used as the explicit large-argument path in `hankel_asymptotics.py`.

### Derivative identity

The code uses the standard adjacent-order relation

- `d/dz H_nu^(1)(z) = 0.5 * (H_(nu-1)^(1)(z) - H_(nu+1)^(1)(z))`
- `d/dz H_nu^(2)(z) = 0.5 * (H_(nu-1)^(2)(z) - H_(nu+1)^(2)(z))`

### Scaling

The scaled surfaces are defined as

- `scaled_hankel1(nu, z) = exp(-i z) H_nu^(1)(z)`
- `scaled_hankel2(nu, z) = exp(+i z) H_nu^(2)(z)`

These strip the dominant oscillatory factor from the large-`|z|` asymptotic.

## 2. Spherical Bessel families

For integer `n >= 0`, the spherical families satisfy

- `j_n(z) = sqrt(pi / (2 z)) J_(n+1/2)(z)`
- `y_n(z) = sqrt(pi / (2 z)) Y_(n+1/2)(z)`
- `i_n(z) = sqrt(pi / (2 z)) I_(n+1/2)(z)`
- `k_n(z) = sqrt(pi / (2 z)) K_(n+1/2)(z)`

The implementation does not expose these half-integer identities as the primary computation path. Instead, it uses family-specific series, seeds, recurrences, and asymptotics.

### Small-argument series

For `j_n` and `i_n`, the code uses the series

- `j_n(z) = (sqrt(pi)/2) * sum_{k>=0} (-1)^k (z/2)^(2k+n) / (k! Gamma(k + n + 3/2))`
- `i_n(z) = (sqrt(pi)/2) * sum_{k>=0} (z/2)^(2k+n) / (k! Gamma(k + n + 3/2))`

This avoids the `sin(z)/z` or `sinh(z)/z` seed formulas in the fragile small-`|z|` regime.

### Seed formulas

The recurrence path starts from:

- `j_0(z) = sin(z) / z`
- `j_1(z) = sin(z) / z^2 - cos(z) / z`
- `y_0(z) = -cos(z) / z`
- `y_1(z) = -cos(z) / z^2 - sin(z) / z`
- `i_0(z) = sinh(z) / z`
- `i_1(z) = cosh(z) / z - sinh(z) / z^2`
- `k_0(z) = (pi / 2) exp(-z) / z`
- `k_1(z) = (pi / 2) exp(-z) (1 + 1/z) / z`

### Upward recurrences

For `j_n` and `y_n`,

- `f_(n+1)(z) = ((2n + 1) / z) f_n(z) - f_(n-1)(z)`

For `i_n`,

- `i_(n+1)(z) = i_(n-1)(z) - ((2n + 1) / z) i_n(z)`

For `k_n`,

- `k_(n+1)(z) = k_(n-1)(z) + ((2n + 1) / z) k_n(z)`

### Derivative identities

For `j_n`, `y_n`, and `i_n`,

- `f_n'(z) = f_(n-1)(z) - ((n+1)/z) f_n(z)`

For `k_n`,

- `k_n'(z) = -k_(n-1)(z) - ((n+1)/z) k_n(z)`

### Leading asymptotics

The current leading asymptotic paths are

- `j_n(z) ~ sin(z - n pi / 2) / z`
- `y_n(z) ~ -cos(z - n pi / 2) / z`
- `i_n(z) ~ exp(z) / (2 z)`
- `k_n(z) ~ (pi / 2) exp(-z) / z`

These are the large-`|z|` branches in `spherical_asymptotics.py`.

## 3. Regime split

The current regime selection is intentionally simple:

- small-`|z|` series for `j_n` and `i_n`,
- recurrence for the mid-range,
- leading asymptotic for large `|z|`.

This keeps the initial implementation concrete and repo-owned while leaving room for later:

- downward recurrence with normalization,
- uniform asymptotics in large-order regimes,
- explicit complex-sector region maps,
- interval and enclosure-specific machinery.
