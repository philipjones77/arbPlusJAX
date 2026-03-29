Last updated: 2026-03-29T00:00:00Z

# Special Functions Spec

## Purpose

This document defines the semantic scope of the special-functions category.

## Category

Coverage category:

6. Special Functions

## Scope

This category covers special-function families evaluated on scalar, matrix, or
operator-adjacent surfaces where the family identity is the primary semantic
organizer.

Representative family groups include:

- gamma and related functions
- Barnes and multiple-gamma style functions
- hypergeometric families
- Dirichlet / zeta / modular / elliptic families
- Bessel and incomplete-tail families

## Semantic Rule

A special-function family is defined by its mathematical identity first, not by
one implementation backend.

The category therefore spans:

- point evaluation
- interval/basic evaluation
- parameterized family behavior
- AD over arguments and continuous parameters
- family-specific diagnostics and asymptotic/regime switching

## Required Axes

Special-function APIs should remain legible across:

- family identity
- argument kind:
  real / complex / parameterized
- mode:
  point / basic / adaptive / rigorous where supported
- execution route:
  direct / repeated-call binder / diagnostics-bearing
- regime or method selection when mathematically relevant

## Relationship To Other Categories

- scalar and mode infrastructure underpin this category
- matrix/operator categories may expose special-function matrix functionals, but
  those remain separate top-level categories unless the family identity is the
  main organizing principle

## Notebook Requirement

Special-function notebooks should show:

- family-specific direct usage
- repeated-call production usage
- argument-direction and parameter-direction AD
- diagnostics/regime information
- practical backend guidance where relevant

