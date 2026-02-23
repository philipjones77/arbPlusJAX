#include "acb_poly_ref.h"

#include <math.h>

#define ACB_POLY_ULP_FACTOR 4.440892098500626e-16
#define ACB_POLY_HUGE 1e300
#define ACB_POLY_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_poly_below(double x)
{
    double t;

    if (x <= ACB_POLY_HUGE)
    {
        t = fabs(x) + ACB_POLY_TINY;
        return x - t * ACB_POLY_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_POLY_HUGE;
    }
}

static double acb_poly_above(double x)
{
    double t;

    if (x >= -ACB_POLY_HUGE)
    {
        t = fabs(x) + ACB_POLY_TINY;
        return x + t * ACB_POLY_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_POLY_HUGE;
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

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_poly_below(z.re), acb_poly_above(z.re));
    out.imag = di_interval(acb_poly_below(z.im), acb_poly_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

acb_box_t acb_poly_eval_cubic_ref(const acb_box_t *coeffs, acb_box_t z)
{
    acb_cplx_t c0 = acb_midpoint(coeffs[0]);
    acb_cplx_t c1 = acb_midpoint(coeffs[1]);
    acb_cplx_t c2 = acb_midpoint(coeffs[2]);
    acb_cplx_t c3 = acb_midpoint(coeffs[3]);
    acb_cplx_t zz = acb_midpoint(z);

    acb_cplx_t v = c3;
    v = acb_add(acb_mul(v, zz), c2);
    v = acb_add(acb_mul(v, zz), c1);
    v = acb_add(acb_mul(v, zz), c0);

    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

void acb_poly_eval_cubic_batch_ref(
    const acb_box_t *coeffs, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_poly_eval_cubic_ref(coeffs + 4 * i, z[i]);
}
