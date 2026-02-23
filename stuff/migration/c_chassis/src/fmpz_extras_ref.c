#include "fmpz_extras_ref.h"

fmpz_extras_ref_t fmpz_extras_add_ref(fmpz_extras_ref_t a, fmpz_extras_ref_t b)
{
    fmpz_extras_ref_t out;
    out.v = a.v + b.v;
    return out;
}

fmpz_extras_ref_t fmpz_extras_mul_ref(fmpz_extras_ref_t a, fmpz_extras_ref_t b)
{
    fmpz_extras_ref_t out;
    out.v = a.v * b.v;
    return out;
}

void fmpz_extras_add_batch_ref(
    const fmpz_extras_ref_t *a, const fmpz_extras_ref_t *b, fmpz_extras_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpz_extras_add_ref(a[i], b[i]);
}

void fmpz_extras_mul_batch_ref(
    const fmpz_extras_ref_t *a, const fmpz_extras_ref_t *b, fmpz_extras_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpz_extras_mul_ref(a[i], b[i]);
}
