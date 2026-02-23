#include "arb_core_ref.h"

#include <math.h>

#define ARB_CORE_ULP_FACTOR 4.440892098500626e-16
#define ARB_CORE_HUGE 1e300
#define ARB_CORE_TINY 1e-300
#define ARB_CORE_PI 3.14159265358979323846
#define ARB_CORE_HALF_PI 1.57079632679489661923
#define ARB_CORE_TWO_PI 6.28318530717958647692

static double arb_core_below(double x)
{
    double t;

    if (x <= ARB_CORE_HUGE)
    {
        t = fabs(x) + ARB_CORE_TINY;
        return x - t * ARB_CORE_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ARB_CORE_HUGE;
    }
}

static double arb_core_above(double x)
{
    double t;

    if (x >= -ARB_CORE_HUGE)
    {
        t = fabs(x) + ARB_CORE_TINY;
        return x + t * ARB_CORE_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ARB_CORE_HUGE;
    }
}

static double arb_core_min2(double a, double b)
{
    return a < b ? a : b;
}

static double arb_core_max2(double a, double b)
{
    return a > b ? a : b;
}

static int arb_core_contains_max_sin(double a, double b)
{
    double kmin = ceil((a - 0.5 * ARB_CORE_PI) / ARB_CORE_TWO_PI);
    double kmax = floor((b - 0.5 * ARB_CORE_PI) / ARB_CORE_TWO_PI);
    return kmin <= kmax;
}

static int arb_core_contains_min_sin(double a, double b)
{
    double kmin = ceil((a - 1.5 * ARB_CORE_PI) / ARB_CORE_TWO_PI);
    double kmax = floor((b - 1.5 * ARB_CORE_PI) / ARB_CORE_TWO_PI);
    return kmin <= kmax;
}

static int arb_core_contains_max_cos(double a, double b)
{
    double kmin = ceil(a / ARB_CORE_TWO_PI);
    double kmax = floor(b / ARB_CORE_TWO_PI);
    return kmin <= kmax;
}

static int arb_core_contains_min_cos(double a, double b)
{
    double kmin = ceil((a - ARB_CORE_PI) / ARB_CORE_TWO_PI);
    double kmax = floor((b - ARB_CORE_PI) / ARB_CORE_TWO_PI);
    return kmin <= kmax;
}

static int arb_core_contains_tan_pole(double a, double b)
{
    double kmin = ceil((a - ARB_CORE_HALF_PI) / ARB_CORE_PI);
    double kmax = floor((b - ARB_CORE_HALF_PI) / ARB_CORE_PI);
    return kmin <= kmax;
}

di_t arb_exp_ref(di_t x)
{
    if (!isfinite(x.a) || !isfinite(x.b))
        return di_interval(-INFINITY, INFINITY);

    return di_interval(arb_core_below(exp(x.a)), arb_core_above(exp(x.b)));
}

di_t arb_log_ref(di_t x)
{
    if (!(x.a > 0.0) || !isfinite(x.a) || !isfinite(x.b))
        return di_interval(-INFINITY, INFINITY);

    return di_interval(arb_core_below(log(x.a)), arb_core_above(log(x.b)));
}

di_t arb_sqrt_ref(di_t x)
{
    double a;
    double b;

    if (!(x.b >= 0.0) || !isfinite(x.a) || !isfinite(x.b))
        return di_interval(-INFINITY, INFINITY);

    a = x.a < 0.0 ? 0.0 : x.a;
    b = x.b < 0.0 ? 0.0 : x.b;
    return di_interval(arb_core_below(sqrt(a)), arb_core_above(sqrt(b)));
}

di_t arb_sin_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double sa, sb, lo, hi;

    if (!isfinite(a) || !isfinite(b))
        return di_interval(-INFINITY, INFINITY);

    if (b - a >= ARB_CORE_TWO_PI)
        return di_interval(arb_core_below(-1.0), arb_core_above(1.0));

    sa = sin(a);
    sb = sin(b);
    lo = arb_core_min2(sa, sb);
    hi = arb_core_max2(sa, sb);

    if (arb_core_contains_max_sin(a, b))
        hi = 1.0;
    if (arb_core_contains_min_sin(a, b))
        lo = -1.0;

    return di_interval(arb_core_below(lo), arb_core_above(hi));
}

di_t arb_cos_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double ca, cb, lo, hi;

    if (!isfinite(a) || !isfinite(b))
        return di_interval(-INFINITY, INFINITY);

    if (b - a >= ARB_CORE_TWO_PI)
        return di_interval(arb_core_below(-1.0), arb_core_above(1.0));

    ca = cos(a);
    cb = cos(b);
    lo = arb_core_min2(ca, cb);
    hi = arb_core_max2(ca, cb);

    if (arb_core_contains_max_cos(a, b))
        hi = 1.0;
    if (arb_core_contains_min_cos(a, b))
        lo = -1.0;

    return di_interval(arb_core_below(lo), arb_core_above(hi));
}

di_t arb_tan_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double ta, tb, lo, hi;

    if (!isfinite(a) || !isfinite(b))
        return di_interval(-INFINITY, INFINITY);

    if (arb_core_contains_tan_pole(a, b))
        return di_interval(-INFINITY, INFINITY);

    ta = tan(a);
    tb = tan(b);
    lo = arb_core_min2(ta, tb);
    hi = arb_core_max2(ta, tb);

    return di_interval(arb_core_below(lo), arb_core_above(hi));
}

di_t arb_sinh_ref(di_t x)
{
    if (!isfinite(x.a) || !isfinite(x.b))
        return di_interval(-INFINITY, INFINITY);

    return di_interval(arb_core_below(sinh(x.a)), arb_core_above(sinh(x.b)));
}

di_t arb_cosh_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double ca, cb, lo, hi;

    if (!isfinite(a) || !isfinite(b))
        return di_interval(-INFINITY, INFINITY);

    ca = cosh(a);
    cb = cosh(b);
    lo = arb_core_min2(ca, cb);
    if (a <= 0.0 && b >= 0.0)
        lo = 1.0;
    hi = arb_core_max2(ca, cb);

    return di_interval(arb_core_below(lo), arb_core_above(hi));
}

di_t arb_tanh_ref(di_t x)
{
    if (!isfinite(x.a) || !isfinite(x.b))
        return di_interval(-INFINITY, INFINITY);

    return di_interval(arb_core_below(tanh(x.a)), arb_core_above(tanh(x.b)));
}

void arb_exp_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_exp_ref(x[i]);
}

void arb_log_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_log_ref(x[i]);
}

void arb_sqrt_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_sqrt_ref(x[i]);
}

void arb_sin_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_sin_ref(x[i]);
}

void arb_cos_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_cos_ref(x[i]);
}

void arb_tan_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_tan_ref(x[i]);
}

void arb_sinh_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_sinh_ref(x[i]);
}

void arb_cosh_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_cosh_ref(x[i]);
}

void arb_tanh_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_tanh_ref(x[i]);
}
