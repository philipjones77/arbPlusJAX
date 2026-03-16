Last updated: 2026-03-13T00:00:00Z

# Incomplete Bessel Benchmark

Benchmark summary for the current incomplete-Bessel package and tail-engine-backed method surface.

## Environment

- OS: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Backend: cpu
- Devices: cpu
- Default dtype: float64 / <class 'jax.numpy.complex128'>

| family | regime | time_ms | sample_output |
|---|---|---:|---:|
| incomplete_bessel_k | quadrature_benign | 94.5220 | 0.165541 |
| incomplete_bessel_k | recurrence_large_lower | 63.6655 | 7.61129e-18 |
| incomplete_bessel_k | asymptotic_large_decay | 0.2257 | 5.11908e-11 |
| incomplete_bessel_k | high_precision_refine_fragile | 242.3750 | 1.59892e+16 |
| incomplete_bessel_i | angular_point | 75.2294 | 1.8198 |
| incomplete_bessel_i | angular_basic | 76.2571 | 1.81519 |
| incomplete_bessel_i | angular_batch_point | 98.3517 | 3.97746 |
