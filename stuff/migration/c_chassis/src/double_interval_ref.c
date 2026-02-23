#include "double_interval_ref.h"

#include <math.h>

#define DI_ULP_FACTOR 4.440892098500626e-16
#define DI_HUGE 1e300
#define DI_TINY 1e-300

static double di_min2(double a, double b)
{
    return a < b ? a : b;
}

static double di_max2(double a, double b)
{
    return a > b ? a : b;
}

static double di_below(double x)
{
    double t;

    if (x <= DI_HUGE)
    {
        t = fabs(x) + DI_TINY;
        return x - t * DI_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;

        return DI_HUGE;
    }
}

static double di_above(double x)
{
    double t;

    if (x >= -DI_HUGE)
    {
        t = fabs(x) + DI_TINY;
        return x + t * DI_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;

        return -DI_HUGE;
    }
}

di_t di_interval(double a, double b)
{
    di_t res;

    if (!(a <= b))
    {
        double t = a;
        a = b;
        b = t;
    }

    res.a = a;
    res.b = b;
    return res;
}

di_t di_neg(di_t x)
{
    di_t res;
    res.a = -x.b;
    res.b = -x.a;
    return res;
}

di_t di_fast_add(di_t x, di_t y)
{
    di_t res;
    res.a = di_below(x.a + y.a);
    res.b = di_above(x.b + y.b);
    return res;
}

di_t di_fast_sub(di_t x, di_t y)
{
    di_t res;
    res.a = di_below(x.a - y.b);
    res.b = di_above(x.b - y.a);
    return res;
}

di_t di_fast_mul(di_t x, di_t y)
{
    di_t res;

    if (x.a > 0.0 && y.a > 0.0)
    {
        res.a = x.a * y.a;
        res.b = x.b * y.b;
    }
    else if (x.a > 0.0 && y.b < 0.0)
    {
        res.a = x.b * y.a;
        res.b = x.a * y.b;
    }
    else if (x.b < 0.0 && y.a > 0.0)
    {
        res.a = x.a * y.b;
        res.b = x.b * y.a;
    }
    else if (x.b < 0.0 && y.b < 0.0)
    {
        res.a = x.b * y.b;
        res.b = x.a * y.a;
    }
    else
    {
        double a = x.a * y.a;
        double b = x.a * y.b;
        double c = x.b * y.a;
        double d = x.b * y.b;

        if (isnan(a) || isnan(b) || isnan(c) || isnan(d))
        {
            res.a = -INFINITY;
            res.b = INFINITY;
        }
        else
        {
            res.a = di_min2(di_min2(a, b), di_min2(c, d));
            res.b = di_max2(di_max2(a, b), di_max2(c, d));
        }
    }

    res.a = di_below(res.a);
    res.b = di_above(res.b);
    return res;
}

di_t di_fast_div(di_t x, di_t y)
{
    di_t res;

    if (y.a > 0.0)
    {
        if (x.a >= 0.0)
        {
            res.a = x.a / y.b;
            res.b = x.b / y.a;
        }
        else if (x.b <= 0.0)
        {
            res.a = x.a / y.a;
            res.b = x.b / y.b;
        }
        else
        {
            res.a = x.a / y.a;
            res.b = x.b / y.a;
        }
    }
    else if (y.b < 0.0)
    {
        if (x.a >= 0.0)
        {
            res.a = x.b / y.b;
            res.b = x.a / y.a;
        }
        else if (x.b <= 0.0)
        {
            res.a = x.b / y.a;
            res.b = x.a / y.b;
        }
        else
        {
            res.a = x.b / y.b;
            res.b = x.a / y.b;
        }
    }
    else
    {
        res.a = -INFINITY;
        res.b = INFINITY;
    }

    res.a = di_below(res.a);
    res.b = di_above(res.b);
    return res;
}

di_t di_fast_sqr(di_t x)
{
    di_t res;

    if (x.a >= 0.0)
    {
        res.a = x.a * x.a;
        res.b = x.b * x.b;
    }
    else if (x.b <= 0.0)
    {
        res.a = x.b * x.b;
        res.b = x.a * x.a;
    }
    else
    {
        res.a = 0.0;
        res.b = di_max2(x.a * x.a, x.b * x.b);
    }

    if (res.a != 0.0)
        res.a = di_below(res.a);

    res.b = di_above(res.b);
    return res;
}

di_t di_fast_log_nonnegative(di_t x)
{
    di_t res;

    if (x.a <= 0.0)
        res.a = -INFINITY;
    else
        res.a = di_below(log(x.a));

    res.b = di_above(log(x.b));
    return res;
}

double di_fast_ubound_radius(di_t x)
{
    return di_above((x.b - x.a) * 0.5);
}

double di_midpoint(di_t x)
{
    if (isinf(x.a) || isinf(x.b))
        return NAN;

    return 0.5 * (x.a + x.b);
}

void di_fast_add_batch(const di_t *x, const di_t *y, di_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
        out[i] = di_fast_add(x[i], y[i]);
}

void di_fast_sub_batch(const di_t *x, const di_t *y, di_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
        out[i] = di_fast_sub(x[i], y[i]);
}

void di_fast_mul_batch(const di_t *x, const di_t *y, di_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
        out[i] = di_fast_mul(x[i], y[i]);
}

void di_fast_div_batch(const di_t *x, const di_t *y, di_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
        out[i] = di_fast_div(x[i], y[i]);
}

void di_fast_sqr_batch(const di_t *x, di_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
        out[i] = di_fast_sqr(x[i]);
}
