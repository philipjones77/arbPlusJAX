#include "arf_ref.h"

arf_ref_t arf_add_ref(arf_ref_t a, arf_ref_t b)
{
    arf_ref_t out;
    out.v = a.v + b.v;
    return out;
}

arf_ref_t arf_mul_ref(arf_ref_t a, arf_ref_t b)
{
    arf_ref_t out;
    out.v = a.v * b.v;
    return out;
}

void arf_add_batch_ref(const arf_ref_t *a, const arf_ref_t *b, arf_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arf_add_ref(a[i], b[i]);
}

void arf_mul_batch_ref(const arf_ref_t *a, const arf_ref_t *b, arf_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arf_mul_ref(a[i], b[i]);
}
