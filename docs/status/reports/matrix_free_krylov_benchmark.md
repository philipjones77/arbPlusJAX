Last updated: 2026-03-14T00:00:00Z

# Matrix-Free Krylov Benchmark

Benchmark runner:
- [benchmark_matrix_free_krylov.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_matrix_free_krylov.py)

Environment notes:
- backend used for this run: CPU
- runtime note observed during the run: NVIDIA hardware may be present, but CUDA-enabled `jaxlib` was not installed, so JAX fell back to CPU

Current coverage:
- real Lanczos action timing
- real Lanczos restarted-action timing
- real Lanczos gradient timing
- real SLQ/logdet timing
- real SLQ/logdet gradient timing
- complex Arnoldi action timing
- complex Arnoldi restarted-action timing
- complex Arnoldi gradient timing
- complex SLQ/logdet timing
- complex SLQ/logdet gradient timing

Latest sampled timings:
- `real_action_s`: `1.740898`
- `real_restarted_action_s`: `0.388969`
- `real_grad_s`: `0.434975`
- `real_logdet_s`: `0.382321`
- `real_logdet_grad_s`: `0.597902`
- `complex_action_s`: `0.545070`
- `complex_restarted_action_s`: `0.437593`
- `complex_grad_s`: `0.592529`
- `complex_logdet_s`: `0.646839`
- `complex_logdet_grad_s`: `0.954977`

Interpretation:
- the backward matrix-free path is now exercised directly, not only the forward action path
- the complex backward path is materially heavier than the real symmetric path, which matches the added Arnoldi and adjoint-operator cost
- this is a focused chassis benchmark, not a full RF77-scale workload characterization
- structured diagnostics wrappers now exist for the same Jones action and estimator families, so timing and state metadata no longer live only in ad hoc benchmark notes

Pipeline check:
- direct source scan found no SciPy calls, host callbacks, or callback-based JAX escapes in `jrb_mat.py` or `jcb_mat.py`
- direct JIT/JAXPR check on the real Lanczos action path found no `host_callback`, `pure_callback`, or `io_callback` tokens
- compiled end-to-end checks succeeded for:
  - real Lanczos action
  - real SLQ/logdet
  - real action gradient
  - complex Arnoldi action
  - complex SLQ/logdet
  - complex action gradient

Linked correctness coverage:
- [test_jrb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jrb_mat_chassis.py)
- [test_jcb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_jcb_mat_chassis.py)
