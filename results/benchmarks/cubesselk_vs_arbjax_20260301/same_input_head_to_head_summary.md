# CubesselK vs Arb JAX BesselK (same inputs)

Date: 2026-03-01
Inputs: n=20000, nu~U(0.1,5.0), z~U(0.1,20.0), interval radius=1e-6, prec_bits=80

## Summary
- Point outputs are numerically identical (`mean/p99/max abs diff = 0`).
- Basic midpoints are numerically identical; Arb basic intervals fully contain CubesselK basic intervals.
- Rigorous and adaptive outputs are numerically identical (equal widths, mutual containment = 1.0).
- Timing (median ms):
  - point: CubesselK 10.073, Arb JAX 9.126
  - basic: CubesselK 8.705, Arb JAX 13.558
  - rigorous: CubesselK 186.411, Arb JAX 162.187
  - adaptive: CubesselK 1836.050, Arb JAX 1606.596
