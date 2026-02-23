#include "arb_mat_ref.h"

#include <math.h>

#define ARB_MAT_ULP_FACTOR 4.440892098500626e-16
#define ARB_MAT_HUGE 1e300
#define ARB_MAT_TINY 1e-300

static double arb_mat_below(double x)
{
    double t;

    if (x <= ARB_MAT_HUGE)
    {
        t = fabs(x) + ARB_MAT_TINY;
        return x - t * ARB_MAT_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ARB_MAT_HUGE;
    }
}

static double arb_mat_above(double x)
{
    double t;

    if (x >= -ARB_MAT_HUGE)
    {
        t = fabs(x) + ARB_MAT_TINY;
        return x + t * ARB_MAT_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ARB_MAT_HUGE;
    }
}

di_t arb_mat_2x2_det_ref(const di_t *a)
{
    double a00 = di_midpoint(a[0]);
    double a01 = di_midpoint(a[1]);
    double a10 = di_midpoint(a[2]);
    double a11 = di_midpoint(a[3]);
    double v = a00 * a11 - a01 * a10;
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    return di_interval(arb_mat_below(v), arb_mat_above(v));
}

di_t arb_mat_2x2_trace_ref(const di_t *a)
{
    double a00 = di_midpoint(a[0]);
    double a11 = di_midpoint(a[3]);
    double v = a00 + a11;
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    return di_interval(arb_mat_below(v), arb_mat_above(v));
}

void arb_mat_2x2_det_batch_ref(const di_t *a, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_mat_2x2_det_ref(a + 4 * i);
}

void arb_mat_2x2_trace_batch_ref(const di_t *a, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_mat_2x2_trace_ref(a + 4 * i);
}
