# Benchmarks

`benchmarks/` is the canonical home for benchmark and cross-backend comparison scripts.

Primary entry points:
- `bench_harness.py`: sweep-based accuracy/speed comparison for JAX interval/point modes.
- `compare_*.py`: focused parity/accuracy checks.
- `benchmark_*.py`: focused throughput/timing runs.

Recommended invocations:
- Quick sweep: `python tools/run_benchmarks.py --profile quick`
- Full sweep: `python tools/run_benchmarks.py --profile full`
- Markdown report from latest run: `python tools/bench_report.py`

Optional Boost baseline:
- Pass `--boost-ref-cmd "<command>"` to `tools/run_benchmarks.py` or set `BOOST_REF_CMD`.
- The command must follow the stdin/stdout JSON contract documented in `boost_ref_adapter_template.py`.

