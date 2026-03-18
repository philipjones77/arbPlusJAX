Last updated: 2026-03-17T18:45:00Z

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
- sparse selected-inverse local inverse-diagonal timing
- sparse selected-inverse corrected inverse-diagonal timing
- complex Arnoldi action timing
- complex Arnoldi restarted-action timing
- complex Arnoldi gradient timing
- complex SLQ/logdet timing
- complex SLQ/logdet gradient timing

Latest sampled timings:
- `real_action_s`: `0.865659`
- `real_restarted_action_s`: `0.356963`
- `real_grad_s`: `0.506229`
- `real_logdet_s`: `0.587504`
- `real_logdet_grad_s`: `0.760225`
- `sparse_real_apply_s`: `0.139512`
- `sparse_real_logdet_s`: `0.593752`
- `sparse_real_logdet_leja_hutchpp_s`: `1.787462`
- `sparse_real_logdet_leja_hutchpp_auto_s`: `2.445490`
- `sparse_real_logdet_grad_s`: `0.649159`
- `sparse_real_inverse_diag_local_s`: `0.333264`
- `sparse_real_inverse_diag_corrected_s`: `2.912418`
- `complex_action_s`: `0.664971`
- `complex_restarted_action_s`: `0.764930`
- `complex_grad_s`: `0.944080`
- `complex_logdet_s`: `0.765486`
- `complex_logdet_grad_s`: `1.561081`

Interpretation:
- the backward matrix-free path is now exercised directly, not only the forward action path
- the complex backward path is materially heavier than the real symmetric path, which matches the added Arnoldi and adjoint-operator cost
- the new sparse selected-inverse local path is materially cheaper than the corrected path, which is expected because the corrected estimator pays for multiple preconditioned full-operator solves
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
