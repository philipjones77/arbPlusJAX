Last updated: 2026-03-17T00:00:00Z

# Practical Docs

This section is for operational and numerically informed guidance: how to run things, how to validate them, how to benchmark them, and what tends to work in practice.

Use [theory](/home/phili/projects/arbplusJAX/docs/theory) for mathematics and derivations.
Use [implementation](/home/phili/projects/arbplusJAX/docs/implementation) for code structure, module layout, wrappers, and implementation details.

## Practical entry points

- [running.md](/home/phili/projects/arbplusJAX/docs/practical/running.md): day-to-day run workflows across local, CI-style, and Colab environments
- [benchmarking.md](/home/phili/projects/arbplusJAX/docs/practical/benchmarking.md): benchmark workflow, harness roles, and comparison policy
- [numerical_guidance.md](/home/phili/projects/arbplusJAX/docs/practical/numerical_guidance.md): numerically informed operating guidance and guardrails
- [dense_matrices.md](/home/phili/projects/arbplusJAX/docs/practical/dense_matrices.md): practical dense runtime guidance for direct dense, cached `matvec`, structured solve reuse, and padded batch tradeoffs
- [matrix_free_adjoints.md](/home/phili/projects/arbplusJAX/docs/practical/matrix_free_adjoints.md): practical use of the matrix-free custom-adjoint Lanczos, Arnoldi, quadrature, and trace-estimator surfaces

## Related implementation references

- [build.md](/home/phili/projects/arbplusJAX/docs/implementation/build.md)
- [run_platform.md](/home/phili/projects/arbplusJAX/docs/implementation/run_platform.md)
- [linux_gpu_colab.md](/home/phili/projects/arbplusJAX/docs/implementation/linux_gpu_colab.md)
- [benchmarks.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmarks.md)
- [benchmark_process.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmark_process.md)
- [testing_harness.md](/home/phili/projects/arbplusJAX/docs/implementation/testing_harness.md)
- [precision_guardrails_gpu.md](/home/phili/projects/arbplusJAX/docs/implementation/precision_guardrails_gpu.md)
- [matrix_logdet_landscape.md](/home/phili/projects/arbplusJAX/docs/implementation/matrix_logdet_landscape.md)
- [soft_ops_optional.md](/home/phili/projects/arbplusJAX/docs/implementation/soft_ops_optional.md)

These remain implementation documents. The `docs/practical/` pages are a separate operational layer that points to them where useful.
