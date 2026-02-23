#include "acb_calc_ref.h"

#include <math.h>

#define ACB_CALC_ULP_FACTOR 4.440892098500626e-16
#define ACB_CALC_HUGE 1e300
#define ACB_CALC_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_calc_below(double x)
{
    double t;

    if (x <= ACB_CALC_HUGE)
    {
        t = fabs(x) + ACB_CALC_TINY;
        return x - t * ACB_CALC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_CALC_HUGE;
    }
}

static double acb_calc_above(double x)
{
    double t;

    if (x >= -ACB_CALC_HUGE)
    {
        t = fabs(x) + ACB_CALC_TINY;
        return x + t * ACB_CALC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_CALC_HUGE;
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

static acb_cplx_t acb_sub(acb_cplx_t a, acb_cplx_t b)
{
    return acb_cplx(a.re - b.re, a.im - b.im);
}

static acb_cplx_t acb_mul(acb_cplx_t a, acb_cplx_t b)
{
    return acb_cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

static acb_cplx_t acb_scale(acb_cplx_t a, double s)
{
    return acb_cplx(a.re * s, a.im * s);
}

static acb_cplx_t acb_exp(acb_cplx_t z)
{
    double e = exp(z.re);
    return acb_cplx(e * cos(z.im), e * sin(z.im));
}

static acb_cplx_t acb_sin(acb_cplx_t z)
{
    return acb_cplx(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

static acb_cplx_t acb_cos(acb_cplx_t z)
{
    return acb_cplx(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_calc_below(z.re), acb_calc_above(z.re));
    out.imag = di_interval(acb_calc_below(z.im), acb_calc_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

static acb_cplx_t acb_eval_integrand(acb_cplx_t z, int integrand_id, int *ok)
{
    *ok = 1;
    switch (integrand_id)
    {
        case ACB_CALC_REF_INTEGRAND_EXP:
            return acb_exp(z);
        case ACB_CALC_REF_INTEGRAND_SIN:
            return acb_sin(z);
        case ACB_CALC_REF_INTEGRAND_COS:
            return acb_cos(z);
        default:
            *ok = 0;
            return acb_cplx(0.0, 0.0);
    }
}

acb_box_t acb_calc_integrate_line_ref(acb_box_t a, acb_box_t b, int integrand_id, int n_steps)
{
    int ok;
    int k;
    double inv_n;
    acb_cplx_t z0 = acb_midpoint(a);
    acb_cplx_t z1 = acb_midpoint(b);
    acb_cplx_t dz;
    acb_cplx_t sum = acb_cplx(0.0, 0.0);
    acb_cplx_t delta = acb_sub(z1, z0);

    if (n_steps <= 0)
        n_steps = 1;
    inv_n = 1.0 / (double) n_steps;
    dz = acb_scale(delta, inv_n);

    for (k = 0; k < n_steps; k++)
    {
        double t = (k + 0.5) * inv_n;
        acb_cplx_t z = acb_add(z0, acb_scale(delta, t));
        acb_cplx_t fz = acb_eval_integrand(z, integrand_id, &ok);
        if (!ok)
            return acb_full();
        sum = acb_add(sum, acb_mul(fz, dz));
    }

    if (!isfinite(sum.re) || !isfinite(sum.im))
        return acb_full();
    return acb_from_complex(sum);
}

void acb_calc_integrate_line_batch_ref(
    const acb_box_t *a, const acb_box_t *b, acb_box_t *out,
    size_t count, int integrand_id, int n_steps)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_calc_integrate_line_ref(a[i], b[i], integrand_id, n_steps);
}
