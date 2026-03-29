Last updated: 2026-03-29T00:00:00Z

# Sparse Block VBlock Functionality Spec

## Purpose

This document defines the semantic scope of the sparse matrix category,
including fixed-block and variable-block sparse forms.

## Category

Coverage category:

4. Sparse / Block-Sparse / VBlock Functionality

## Scope

This category covers:

- sparse explicit matrices
- block-sparse matrices
- variable-block sparse matrices

Representative families:

- `srb_mat`
- `scb_mat`
- `srb_block_mat`
- `scb_block_mat`
- `srb_vblock_mat`
- `scb_vblock_mat`

## Semantic Rule

Sparse, block-sparse, and variable-block sparse are explicit sparse-storage
families, not dense matrices with an implementation detail.

They should remain semantically distinct from:

- dense matrices
- matrix-free/operator surfaces

## Required Axes

Sparse APIs should remain legible across:

- matrix kind:
  sparse / block-sparse / variable-block sparse
- storage kind:
  COO / CSR / BCOO / block-structured variants
- structure subtype:
  symmetric / Hermitian / SPD / HPD / triangular / similar
- execution route:
  direct sparse apply / cached sparse apply / sparse plan route /
  diagnostics-bearing route / compiled batch binder
- mode:
  point and any sparse-native enclosure mode that the family supports

## No-Dense-Fallback Rule

Sparse operational paths should be explicit about whether they remain sparse or
lift to dense.

At the spec level, the semantic distinction is:

- sparse-native path
- dense-lifted sparse path

The API and notebooks should not blur these two.

## Relationship To Other Categories

- dense is a separate category
- block-sparse/vblock remain inside the sparse category, not a fifth matrix kind
- matrix-free/operator remains a separate top-level category

## Notebook Requirement

Sparse notebooks should show:

- sparse kind choice
- structure flags where relevant
- sparse-native cached or prepared repeated-call route
- block/vblock contrast
- honest CPU/GPU guidance

