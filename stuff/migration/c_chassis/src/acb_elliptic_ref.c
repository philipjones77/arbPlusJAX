#include "acb_elliptic_ref.h"

#include <math.h>

#define ACB_ELLIPTIC_ULP_FACTOR 4.440892098500626e-16
#define ACB_ELLIPTIC_HUGE 1e300
#define ACB_ELLIPTIC_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_elliptic_below(double x)
{
    double t;

    if (x <= ACB_ELLIPTIC_HUGE)
    {
        t = fabs(x) + ACB_ELLIPTIC_TINY;
        return x - t * ACB_ELLIPTIC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_ELLIPTIC_HUGE;
    }
}

static double acb_elliptic_above(double x)
{
    double t;

    if (x >= -ACB_ELLIPTIC_HUGE)
    {
        t = fabs(x) + ACB_ELLIPTIC_TINY;
        return x + t * ACB_ELLIPTIC_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_ELLIPTIC_HUGE;
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

static acb_cplx_t acb_scale(acb_cplx_t a, double s)
{
    return acb_cplx(a.re * s, a.im * s);
}

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_elliptic_below(z.re), acb_elliptic_above(z.re));
    out.imag = di_interval(acb_elliptic_below(z.im), acb_elliptic_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

static acb_cplx_t acb_sqrt(acb_cplx_t z)
{
    double r = hypot(z.re, z.im);
    double u = sqrt(fmax(0.0, 0.5 * (r + z.re)));
    double v = sqrt(fmax(0.0, 0.5 * (r - z.re)));
    if (z.im < 0.0)
        v = -v;
    return acb_cplx(u, v);
}

static acb_cplx_t acb_exp(acb_cplx_t z)
{
    double e = exp(z.re);
    return acb_cplx(e * cos(z.im), e * sin(z.im));
}

static acb_cplx_t acb_log(acb_cplx_t z)
{
    return acb_cplx(log(hypot(z.re, z.im)), atan2(z.im, z.re));
}

static acb_cplx_t acb_pow_real(double x, acb_cplx_t s)
{
    acb_cplx_t l = acb_cplx(log(x), 0.0);
    acb_cplx_t t = acb_mul(s, l);
    return acb_exp(t);
}

static acb_cplx_t acb_elliptic_agm(acb_cplx_t a, acb_cplx_t b, int max_iter)
{
    int k;
    for (k = 0; k < max_iter; k++)
    {
        acb_cplx_t a_next = acb_scale(acb_add(a, b), 0.5);
        acb_cplx_t b_next = acb_sqrt(acb_mul(a, b));
        a = a_next;
        b = b_next;
    }
    return a;
}

acb_box_t acb_elliptic_k_ref(acb_box_t m)
{
    acb_cplx_t mm = acb_midpoint(m);
    acb_cplx_t one = acb_cplx(1.0, 0.0);
    acb_cplx_t diff = acb_cplx(one.re - mm.re, -mm.im);
    acb_cplx_t k = acb_sqrt(diff);
    acb_cplx_t agm = acb_elliptic_agm(one, k, 8);
    acb_cplx_t pi = acb_cplx(3.14159265358979323846, 0.0);
    acb_cplx_t v = acb_scale(acb_div(pi, agm), 0.5);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_elliptic_e_ref(acb_box_t m)
{
    acb_cplx_t mm = acb_midpoint(m);
    acb_cplx_t one = acb_cplx(1.0, 0.0);
    acb_cplx_t diff = acb_cplx(one.re - mm.re, -mm.im);
    acb_cplx_t k = acb_sqrt(diff);
    acb_cplx_t agm = acb_elliptic_agm(one, k, 8);
    acb_cplx_t v = acb_scale(agm, 1.57079632679489661923);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

void acb_elliptic_k_batch_ref(const acb_box_t *m, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_elliptic_k_ref(m[i]);
}

void acb_elliptic_e_batch_ref(const acb_box_t *m, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_elliptic_e_ref(m[i]);
}
