#include "dlog_ref.h"

#include <math.h>

double dlog_log1p_ref(double x)
{
    return log1p(x);
}

void dlog_log1p_batch_ref(const double *x, double *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = dlog_log1p_ref(x[i]);
}
