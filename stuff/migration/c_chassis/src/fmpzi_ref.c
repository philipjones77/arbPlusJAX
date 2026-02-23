#include "fmpzi_ref.h"

static void fmpzi_order(int64_t *lo, int64_t *hi)
{
    if (*lo > *hi)
    {
        int64_t t = *lo;
        *lo = *hi;
        *hi = t;
    }
}

fmpzi_ref_t fmpzi_interval_ref(int64_t lo, int64_t hi)
{
    fmpzi_ref_t out;
    fmpzi_order(&lo, &hi);
    out.lo = lo;
    out.hi = hi;
    return out;
}

fmpzi_ref_t fmpzi_add_ref(fmpzi_ref_t a, fmpzi_ref_t b)
{
    fmpzi_ref_t out;
    out.lo = a.lo + b.lo;
    out.hi = a.hi + b.hi;
    return out;
}

fmpzi_ref_t fmpzi_sub_ref(fmpzi_ref_t a, fmpzi_ref_t b)
{
    fmpzi_ref_t out;
    out.lo = a.lo - b.hi;
    out.hi = a.hi - b.lo;
    return out;
}

void fmpzi_add_batch_ref(const fmpzi_ref_t *a, const fmpzi_ref_t *b, fmpzi_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpzi_add_ref(a[i], b[i]);
}

void fmpzi_sub_batch_ref(const fmpzi_ref_t *a, const fmpzi_ref_t *b, fmpzi_ref_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = fmpzi_sub_ref(a[i], b[i]);
}
