Last updated: 2026-03-23T00:00:00Z

# Sparse Block VBlock Methodology

## Purpose

This note records the current mathematical and production interpretation of the
sparse, block-sparse, and variable-block sparse matrix surfaces in arbPlusJAX.

## Scope

This note covers:

- COO and CSR sparse matrix surfaces
- fixed-block sparse surfaces
- variable-block sparse surfaces
- cached `matvec` and `rmatvec` reuse
- the relation between sparse storage structure and the public API contract

## Sparse Interpretation

The sparse point surfaces represent linear maps whose structural zeros are part
of the contract, not merely a storage optimization.

The public sparse contract should be read as:

- sparse storage is authoritative for repeated apply paths
- dense conversion is an explicit fallback or comparison path
- cached sparse plans exist to move setup and canonicalization work out of the
  repeated numeric loop

## Block-Sparse Interpretation

Block-sparse matrices preserve a second level of structure:

- the matrix is sparse at the block level
- each stored nonzero carries a dense block payload

This structure matters because it changes both:

- the execution strategy
- the preferred API surface for repeated right-hand sides and transpose/adjoint
  application

Block-sparse should therefore be treated as its own production family rather
than as a cosmetic variant of scalar-entry sparse storage.

## Variable-Block Interpretation

Variable-block sparse matrices preserve irregular partition structure. The
public contract is:

- partitions are part of the semantic input
- block boundaries should not be erased unless the caller explicitly converts to
  a denser representation
- cached apply paths should preserve the chosen partitioning so recompiles and
  setup work do not drift across repeated calls

## Cached Apply Contract

For sparse, block-sparse, and variable-block sparse workloads, production use
should prefer:

1. explicit preparation for a fixed structure
2. repeated `matvec` or `rmatvec` application through the cached plan
3. stable dtype and batch-shape policy across repeated calls

This is both a performance rule and an API-discipline rule, because structure
changes are often compile-relevant.

## Interval And Box Interpretation

The current sparse family is strongest in point mode. Interval and box lifting
should be interpreted as a higher semantic layer, not as proof that every
sparse factorization path is already a first-class rigorous sparse kernel.

Current production interpretation:

- point sparse execution is the primary substrate
- interval/box sparse support exists in selected places, including block/vblock
  `basic` determinant/inverse/square paths, main sparse direct `basic`
  determinant/inverse/square and core factor/solve entrypoints, and the main
  sparse `basic` LU / SPD / HPD plan-prepare lift into dense interval/box solve
  plans
- status and benchmarks should distinguish structural sparse execution from
  enclosure-quality claims

## Diagnostics Interpretation

Sparse diagnostics should expose enough metadata for production routing:

- storage kind
- cached versus uncached path
- transpose/adjoint usage
- method choice for solve-oriented paths when present
- current hardening level for interval/box semantics

Diagnostics and metadata should stay outside the mandatory sparse numeric hot
path when disabled.

## Production Calling Contract

Canonical sparse examples and benchmarks should teach:

- fixed dtype policy
- stable structural inputs
- cached prepare/apply reuse
- explicit choice between sparse, block-sparse, variable-block, and matrix-free
  surfaces
- visual comparison against dense baselines only as an explicit contrast path

## Current Limitations

- interval/box sparse semantics still need further hardening beyond the point
  substrate
- variable-block factorization assumptions still need to be tightened in some
  paths
- sparse benchmark normalization now separates storage preparation and cached
  plan preparation, plus main sparse LU and SPD/HPD factor-plan preparation, on
  the main sparse and block/vblock benchmark surfaces, but broader sparse
  benchmark families still need the same split
- GPU validation is deferred to a later tranche
