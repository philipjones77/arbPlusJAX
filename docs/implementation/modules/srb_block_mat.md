Last updated: 2026-03-15T00:00:00Z

# srb_block_mat

## Role

`srb_block_mat` is the real block-sparse JAX matrix layer.

It is separate from:

- [srb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/srb_mat.py): scalar sparse matrices
- [arb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/arb_mat.py): dense interval matrices

## Current Scope

Implemented fixed-block BSR-like structures:

- `BlockCOO`
- `BlockCSR`

Implemented kernels:

- shape / block-shape / `nnzb`
- dense-to-block-sparse conversion
- `BlockCOO <-> BlockCSR`
- block-sparse to dense
- transpose
- block `matvec`
- cached block `matvec` prepare/apply
- fixed/padded batch block `matvec`
- block times dense-RHS `matmul`
- block triangular solve
- iterative block solve (`cg` / `gmres` / `bicgstab`)

## Current Constraint

This is a point-value fixed-block sparse layer.

Not implemented yet:

- variable block-size storage
- block direct factorizations
- interval/rigorous block-sparse modes
