Last updated: 2026-03-15T00:00:00Z

# scb_block_mat

## Role

`scb_block_mat` is the complex block-sparse JAX matrix layer.

It is separate from:

- [scb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/scb_mat.py): scalar sparse matrices
- [acb_mat](/home/phili/projects/arbplusJAX/src/arbplusjax/acb_mat.py): dense complex box matrices

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
- box/interval block-sparse modes
