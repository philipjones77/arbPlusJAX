# Example Core Scalar Surface Summary (gpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `gpu`
- devices: `['cuda:0']`
- benchmark_rows: `42`
- profile_rows: `288`
- comparison_rows: `2`
- batch_padding_rows: `37`

## Benchmark Operations

- `acf_mul`
- `arb_fpwrap_double_exp`
- `arf_add`
- `fmpr_mul`
- `fmpzi_add`

## Backend Summary

- `scipy`: mean_time_ms=0.0238096, mean_containment=1
- `c_arb`: mean_time_ms=0.256024, mean_containment=nan
- `mpmath`: mean_time_ms=2.01479, mean_containment=1
- `jax_rigorous`: mean_time_ms=3.41409, mean_containment=0
- `jax_adaptive`: mean_time_ms=3.47385, mean_containment=0
- `jax_point`: mean_time_ms=3.55741, mean_containment=1
- `boost`: mean_time_ms=11.9648, mean_containment=1
- `jax_basic`: mean_time_ms=75.5145, mean_containment=0

## Batch Padding Speed

- `arf_add` / `api_batch_unpadded`: warm=0.00100746s, recompile=0.0504288s
- `arf_add` / `api_batch_padded`: warm=0.000949721s, recompile=0.048379s
- `acf_mul` / `api_batch_unpadded`: warm=0.000608395s, recompile=0.0391907s
- `acf_mul` / `api_batch_padded`: warm=0.000200842s, recompile=0.0547454s
- `fmpr_mul` / `api_batch_unpadded`: warm=0.00169711s, recompile=0.0595623s
- `fmpr_mul` / `api_batch_padded`: warm=0.000292131s, recompile=0.0624073s
- `fmpzi_add` / `api_batch_unpadded`: warm=0.00207537s, recompile=0.0638766s
- `fmpzi_add` / `api_batch_padded`: warm=0.00126869s, recompile=0.0817591s
- `arb_fpwrap_double_exp` / `api_batch_unpadded`: warm=0.000549827s, recompile=0.0475731s
- `arb_fpwrap_double_exp` / `api_batch_padded`: warm=0.00122783s, recompile=0.0431799s
- `arf_add` / `service_api_unpadded`: warm=0.00182927s, recompile=0.0431128s
- `arf_add` / `service_api_padded`: warm=0.00315839s, recompile=0.12532s
- `arf_add` / `service_api_bucketed`: warm=0.00441164s, recompile=0.16288s
- `fmpr_mul` / `service_api_unpadded`: warm=0.0012725s, recompile=0.0446216s
- `fmpr_mul` / `service_api_padded`: warm=0.00433701s, recompile=0.00277277s
- `fmpr_mul` / `service_api_bucketed`: warm=0.00176563s, recompile=0.0351524s
- `arb_fpwrap_double_exp` / `service_api_unpadded`: warm=0.00133081s, recompile=0.0480978s
- `arb_fpwrap_double_exp` / `service_api_padded`: warm=0.0025724s, recompile=0.00354797s
- `arb_fpwrap_double_exp` / `service_api_bucketed`: warm=0.00358415s, recompile=0.0484303s
- `acf_mul` / `service_api_unpadded`: warm=0.000592119s, recompile=0.0459221s
- `acf_mul` / `service_api_padded`: warm=0.00331329s, recompile=0.138663s
- `acf_mul` / `service_api_bucketed`: warm=0.00649297s, recompile=0.178382s
- `arf_add` / `service_api_unpadded`: warm=0.0024624s, recompile=0.0561496s
- `arf_add` / `service_api_padded`: warm=0.00539455s, recompile=0.167635s
- `arf_add` / `service_api_bucketed`: warm=0.00427158s, recompile=0.169s
- `fmpr_mul` / `service_api_unpadded`: warm=0.00111062s, recompile=0.0407829s
- `fmpr_mul` / `service_api_padded`: warm=0.00283872s, recompile=0.00233152s
- `fmpr_mul` / `service_api_bucketed`: warm=0.00174669s, recompile=0.0537809s
- `arb_fpwrap_double_exp` / `service_api_unpadded`: warm=0.00092117s, recompile=0.0432004s
- `arb_fpwrap_double_exp` / `service_api_padded`: warm=0.00354436s, recompile=0.00285296s
- `arb_fpwrap_double_exp` / `service_api_bucketed`: warm=0.00189059s, recompile=0.0495781s
- `acf_mul` / `service_api_unpadded`: warm=0.000782206s, recompile=0.0489506s
- `acf_mul` / `service_api_padded`: warm=0.00302885s, recompile=0.136732s
- `acf_mul` / `service_api_bucketed`: warm=0.00307354s, recompile=0.179058s
- `fmpzi_add` / `service_api_unpadded`: warm=0.00108111s, recompile=0.0833238s
- `fmpzi_add` / `service_api_padded`: warm=0.0044707s, recompile=0.181112s
- `fmpzi_add` / `service_api_bucketed`: warm=0.00526512s, recompile=0.260694s
