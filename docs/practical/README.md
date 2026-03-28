Last updated: 2026-03-17T00:00:00Z

# Practical Docs

This section is for operational and numerically informed guidance: how to run things, how to validate them, how to benchmark them, and what tends to work in practice.

Use [theory](/docs/theory) for mathematics and derivations.
Use [implementation](/docs/implementation) for code structure, module layout, wrappers, and implementation details.

## Practical entry points

- [running.md](/docs/practical/running.md): day-to-day run workflows across local, CI-style, and Colab environments
- [benchmarking.md](/docs/practical/benchmarking.md): benchmark workflow, harness roles, and comparison policy
- [backend_realized_performance_usage.md](/docs/practical/backend_realized_performance_usage.md): when GPU actually helps, what a raw or lightly wrapped compiled path means, and how to call the API for repeated CPU/GPU execution
- [numerical_guidance.md](/docs/practical/numerical_guidance.md): numerically informed operating guidance and guardrails
- [dense_matrices.md](/docs/practical/dense_matrices.md): practical dense runtime guidance for direct dense, cached `matvec`, structured solve reuse, and padded batch tradeoffs
- [matrix_free_adjoints.md](/docs/practical/matrix_free_adjoints.md): practical use of the matrix-free custom-adjoint Lanczos, Arnoldi, quadrature, and trace-estimator surfaces

## Related implementation references

- [build_implementation.md](/docs/implementation/build_implementation.md)
- [run_platform_implementation.md](/docs/implementation/run_platform_implementation.md)
- [linux_gpu_colab_implementation.md](/docs/implementation/linux_gpu_colab_implementation.md)
- [benchmarks_implementation.md](/docs/implementation/benchmarks_implementation.md)
- [benchmark_process_implementation.md](/docs/implementation/benchmark_process_implementation.md)
- [testing_harness_implementation.md](/docs/implementation/testing_harness_implementation.md)
- [precision_guardrails_gpu_implementation.md](/docs/implementation/precision_guardrails_gpu_implementation.md)
- [matrix_logdet_landscape_implementation.md](/docs/implementation/matrix_logdet_landscape_implementation.md)
- [soft_ops_optional_implementation.md](/docs/implementation/soft_ops_optional_implementation.md)

These remain implementation documents. The `docs/practical/` pages are a separate operational layer that points to them where useful.
