#include "acb_modular_ref.h"

#include <math.h>

#define ACB_MODULAR_ULP_FACTOR 4.440892098500626e-16
#define ACB_MODULAR_HUGE 1e300
#define ACB_MODULAR_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_modular_below(double x)
{
    double t;

    if (x <= ACB_MODULAR_HUGE)
    {
        t = fabs(x) + ACB_MODULAR_TINY;
        return x - t * ACB_MODULAR_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_MODULAR_HUGE;
    }
}

static double acb_modular_above(double x)
{
    double t;

    if (x >= -ACB_MODULAR_HUGE)
    {
        t = fabs(x) + ACB_MODULAR_TINY;
        return x + t * ACB_MODULAR_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_MODULAR_HUGE;
    }
}

static acb_cplx_t acb_cplx(double re, double im)
{
    acb_cplx_t out;
    out.re = re;
    out.im = im;
    return out;
}

static acb_cplx_t acb_add(acb_cplx_t a, acb_cplx_t b)
{
    return acb_cplx(a.re + b.re, a.im + b.im);
}

static acb_cplx_t acb_mul(acb_cplx_t a, acb_cplx_t b)
{
    return acb_cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

static acb_cplx_t acb_div(acb_cplx_t a, acb_cplx_t b)
{
    double d = b.re * b.re + b.im * b.im;
    return acb_cplx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

static acb_cplx_t acb_exp(acb_cplx_t z)
{
    double e = exp(z.re);
    return acb_cplx(e * cos(z.im), e * sin(z.im));
}

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_modular_below(z.re), acb_modular_above(z.re));
    out.imag = di_interval(acb_modular_below(z.im), acb_modular_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

acb_box_t acb_modular_j_ref(acb_box_t tau)
{
    acb_cplx_t t = acb_midpoint(tau);
    acb_cplx_t i2pi = acb_cplx(0.0, 6.28318530717958647692);
    acb_cplx_t q = acb_exp(acb_mul(i2pi, t));

    double qnorm = q.re * q.re + q.im * q.im;
    if (!isfinite(q.re) || !isfinite(q.im) || qnorm == 0.0)
        return acb_full();

    acb_cplx_t qinv = acb_div(acb_cplx(1.0, 0.0), q);
    acb_cplx_t q2 = acb_mul(q, q);

    acb_cplx_t v = qinv;
    v = acb_add(v, acb_cplx(744.0, 0.0));
    v = acb_add(v, acb_mul(acb_cplx(196884.0, 0.0), q));
    v = acb_add(v, acb_mul(acb_cplx(21493760.0, 0.0), q2));

    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

void acb_modular_j_batch_ref(const acb_box_t *tau, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_modular_j_ref(tau[i]);
}
