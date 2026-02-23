#include "fmpr_ref.h"

fmpr_ref_t fmpr_add_ref(fmpr_ref_t a, fmpr_ref_t b)
{
    fmpr_ref_t out;
    out.v = a.v + b.v;
    return out;
}

fmpr_ref_t fmpr_mul_ref(fmpr_ref_t a, fmpr_ref_t b)
{
    fmpr_ref_t out;
    out.v = a.v * b.v;
    return out;
}

void fmpr_add_batch_ref(const fmpr_ref_t *a, const fmpr_ref_t *b, fmpr_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpr_add_ref(a[i], b[i]);
}

void fmpr_mul_batch_ref(const fmpr_ref_t *a, const fmpr_ref_t *b, fmpr_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpr_mul_ref(a[i], b[i]);
}
