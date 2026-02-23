# modular_elliptic_wrappers

Mode dispatch for `acb_modular` and `acb_elliptic`.

## Rigorous handling

Rigorous mode uses explicit interval arithmetic for available formulas (e.g., `acb_modular_j_rigorous`, `acb_elliptic_k_rigorous`, `acb_elliptic_e_rigorous`). Other functions use Jacobian or adaptive sampling bounds.
