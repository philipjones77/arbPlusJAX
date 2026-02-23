#include "arb_calc_ref.h"

#include <math.h>

#define ARB_CALC_ULP_FACTOR 4.440892098500626e-16
#define ARB_CALC_HUGE 1e300
#define ARB_CALC_TINY 1e-300

static double arb_calc_below(double x)
{
    double t;

    if (x <= ARB_CALC_HUGE)
    {
        t = fabs(x) + ARB_CALC_TINY;
        return x - t * ARB_CALC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ARB_CALC_HUGE;
    }
}

static double arb_calc_above(double x)
{
    double t;

    if (x >= -ARB_CALC_HUGE)
    {
        t = fabs(x) + ARB_CALC_TINY;
        return x + t * ARB_CALC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ARB_CALC_HUGE;
    }
}

static double arb_calc_eval(double x, int integrand_id, int *ok)
{
    *ok = 1;
    switch (integrand_id)
    {
        case ARB_CALC_REF_INTEGRAND_EXP:
            return exp(x);
        case ARB_CALC_REF_INTEGRAND_SIN:
            return sin(x);
        case ARB_CALC_REF_INTEGRAND_COS:
            return cos(x);
        default:
            *ok = 0;
            return 0.0;
    }
}

di_t arb_calc_integrate_line_ref(di_t a, di_t b, int integrand_id, int n_steps)
{
    int ok;
    int k;
    double am = di_midpoint(a);
    double bm = di_midpoint(b);
    double delta = bm - am;
    double inv_n;
    double sum = 0.0;

    if (n_steps <= 0)
        n_steps = 1;
    inv_n = 1.0 / (double) n_steps;

    for (k = 0; k < n_steps; k++)
    {
        double t = (k + 0.5) * inv_n;
        double x = am + delta * t;
        double fx = arb_calc_eval(x, integrand_id, &ok);
        if (!ok)
            return di_interval(-INFINITY, INFINITY);
        sum += fx;
    }

    sum *= delta * inv_n;
    if (!isfinite(sum))
        return di_interval(-INFINITY, INFINITY);
    return di_interval(arb_calc_below(sum), arb_calc_above(sum));
}

void arb_calc_integrate_line_batch_ref(
    const di_t *a, const di_t *b, di_t *out, size_t count, int integrand_id, int n_steps)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_calc_integrate_line_ref(a[i], b[i], integrand_id, n_steps);
}
