#include "acb_dirichlet_ref.h"

#include <math.h>

#define ACB_DIRICHLET_ULP_FACTOR 4.440892098500626e-16
#define ACB_DIRICHLET_HUGE 1e300
#define ACB_DIRICHLET_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_dirichlet_below(double x)
{
    double t;

    if (x <= ACB_DIRICHLET_HUGE)
    {
        t = fabs(x) + ACB_DIRICHLET_TINY;
        return x - t * ACB_DIRICHLET_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_DIRICHLET_HUGE;
    }
}

static double acb_dirichlet_above(double x)
{
    double t;

    if (x >= -ACB_DIRICHLET_HUGE)
    {
        t = fabs(x) + ACB_DIRICHLET_TINY;
        return x + t * ACB_DIRICHLET_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_DIRICHLET_HUGE;
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

static acb_cplx_t acb_scale(acb_cplx_t a, double s)
{
    return acb_cplx(a.re * s, a.im * s);
}

static acb_cplx_t acb_exp(acb_cplx_t z)
{
    double e = exp(z.re);
    return acb_cplx(e * cos(z.im), e * sin(z.im));
}

static acb_cplx_t acb_log_real(double x)
{
    return acb_cplx(log(x), 0.0);
}

static acb_cplx_t acb_pow_real(double x, acb_cplx_t s)
{
    acb_cplx_t l = acb_log_real(x);
    acb_cplx_t t = acb_mul(acb_cplx(-s.re, -s.im), l);
    return acb_exp(t);
}

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_dirichlet_below(z.re), acb_dirichlet_above(z.re));
    out.imag = di_interval(acb_dirichlet_below(z.im), acb_dirichlet_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

static acb_cplx_t acb_zeta_series(acb_cplx_t s, int n_terms)
{
    acb_cplx_t sum = acb_cplx(0.0, 0.0);
    int k;

    if (n_terms <= 0)
        n_terms = 1;
    for (k = 1; k <= n_terms; k++)
    {
        acb_cplx_t term = acb_pow_real((double) k, s);
        sum = acb_add(sum, term);
    }
    return sum;
}

acb_box_t acb_dirichlet_zeta_ref(acb_box_t s, int n_terms)
{
    acb_cplx_t sm = acb_midpoint(s);
    acb_cplx_t v = acb_zeta_series(sm, n_terms);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_dirichlet_eta_ref(acb_box_t s, int n_terms)
{
    acb_cplx_t sm = acb_midpoint(s);
    acb_cplx_t zeta = acb_zeta_series(sm, n_terms);
    acb_cplx_t pow2 = acb_pow_real(2.0, acb_cplx(1.0 - sm.re, -sm.im));
    acb_cplx_t factor = acb_cplx(1.0 - pow2.re, -pow2.im);
    acb_cplx_t v = acb_mul(factor, zeta);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

void acb_dirichlet_zeta_batch_ref(const acb_box_t *s, acb_box_t *out, size_t count, int n_terms)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_dirichlet_zeta_ref(s[i], n_terms);
}

void acb_dirichlet_eta_batch_ref(const acb_box_t *s, acb_box_t *out, size_t count, int n_terms)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_dirichlet_eta_ref(s[i], n_terms);
}
