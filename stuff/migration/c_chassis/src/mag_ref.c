#include "mag_ref.h"

double mag_add_ref(double a, double b)
{
    return a + b;
}

double mag_mul_ref(double a, double b)
{
    return a * b;
}

void mag_add_batch_ref(const double *a, const double *b, double *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = mag_add_ref(a[i], b[i]);
}

void mag_mul_batch_ref(const double *a, const double *b, double *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = mag_mul_ref(a[i], b[i]);
}
