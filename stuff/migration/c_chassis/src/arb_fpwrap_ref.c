#include "arb_fpwrap_ref.h"

#include <math.h>

int arb_fpwrap_double_exp_ref(double *res, double x)
{
    double v = exp(x);
    if (!isfinite(v))
        return 1;
    *res = v;
    return 0;
}

int arb_fpwrap_double_log_ref(double *res, double x)
{
    double v = log(x);
    if (!isfinite(v))
        return 1;
    *res = v;
    return 0;
}

int arb_fpwrap_cdouble_exp_ref(arb_fpwrap_cdouble_t *res, arb_fpwrap_cdouble_t x)
{
    double e = exp(x.real);
    double re = e * cos(x.imag);
    double im = e * sin(x.imag);
    if (!isfinite(re) || !isfinite(im))
        return 1;
    res->real = re;
    res->imag = im;
    return 0;
}

int arb_fpwrap_cdouble_log_ref(arb_fpwrap_cdouble_t *res, arb_fpwrap_cdouble_t x)
{
    double re = log(hypot(x.real, x.imag));
    double im = atan2(x.imag, x.real);
    if (!isfinite(re) || !isfinite(im))
        return 1;
    res->real = re;
    res->imag = im;
    return 0;
}
