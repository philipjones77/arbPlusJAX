#include "arb_poly_ref.h"

#include <math.h>

#define ARB_POLY_ULP_FACTOR 4.440892098500626e-16
#define ARB_POLY_HUGE 1e300
#define ARB_POLY_TINY 1e-300

static double arb_poly_below(double x)
{
    double t;

    if (x <= ARB_POLY_HUGE)
    {
        t = fabs(x) + ARB_POLY_TINY;
        return x - t * ARB_POLY_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ARB_POLY_HUGE;
    }
}

static double arb_poly_above(double x)
{
    double t;

    if (x >= -ARB_POLY_HUGE)
    {
        t = fabs(x) + ARB_POLY_TINY;
        return x + t * ARB_POLY_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ARB_POLY_HUGE;
    }
}

di_t arb_poly_eval_cubic_ref(const di_t *coeffs, di_t x)
{
    double c0 = di_midpoint(coeffs[0]);
    double c1 = di_midpoint(coeffs[1]);
    double c2 = di_midpoint(coeffs[2]);
    double c3 = di_midpoint(coeffs[3]);
    double xm = di_midpoint(x);
    double v = ((c3 * xm + c2) * xm + c1) * xm + c0;
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    return di_interval(arb_poly_below(v), arb_poly_above(v));
}

void arb_poly_eval_cubic_batch_ref(
    const di_t *coeffs, const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_poly_eval_cubic_ref(coeffs + 4 * i, x[i]);
}
