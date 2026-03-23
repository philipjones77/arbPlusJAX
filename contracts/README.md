Last updated: 2026-03-17T00:00:00Z

# Contracts

This directory is reserved for binding runtime and API guarantees.

Operational obligations should go here rather than under `docs/`.

## Active contract set

- [public_api_mode_contract.md](/contracts/public_api_mode_contract.md)
- [public_api_selection_contract.md](/contracts/public_api_selection_contract.md)
- [dtype_and_precision_contract.md](/contracts/dtype_and_precision_contract.md)
- [stable_kernel_subset_contract.md](/contracts/stable_kernel_subset_contract.md)
- [matrix_surface_contract.md](/contracts/matrix_surface_contract.md)
- [sparse_layout_operator_contract.md](/contracts/sparse_layout_operator_contract.md)
- [capability_registry_contract.md](/contracts/capability_registry_contract.md)

## Scope

These contracts apply to the currently stabilized repo surface.

- They do not promise that every experimental module is frozen.
- They do define the current guarantees we expect downstream code to rely on.
