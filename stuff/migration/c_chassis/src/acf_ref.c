#include "acf_ref.h"

acf_ref_t acf_add_ref(acf_ref_t a, acf_ref_t b)
{
    acf_ref_t out;
    out.re = a.re + b.re;
    out.im = a.im + b.im;
    return out;
}

acf_ref_t acf_mul_ref(acf_ref_t a, acf_ref_t b)
{
    acf_ref_t out;
    out.re = a.re * b.re - a.im * b.im;
    out.im = a.re * b.im + a.im * b.re;
    return out;
}

void acf_add_batch_ref(const acf_ref_t *a, const acf_ref_t *b, acf_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acf_add_ref(a[i], b[i]);
}

void acf_mul_batch_ref(const acf_ref_t *a, const acf_ref_t *b, acf_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acf_mul_ref(a[i], b[i]);
}
