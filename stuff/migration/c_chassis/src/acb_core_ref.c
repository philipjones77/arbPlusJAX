#include "acb_core_ref.h"

#include <math.h>

#define ACB_CORE_ULP_FACTOR 4.440892098500626e-16
#define ACB_CORE_HUGE 1e300
#define ACB_CORE_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_core_below(double x)
{
    double t;

    if (x <= ACB_CORE_HUGE)
    {
        t = fabs(x) + ACB_CORE_TINY;
        return x - t * ACB_CORE_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_CORE_HUGE;
    }
}

static double acb_core_above(double x)
{
    double t;

    if (x >= -ACB_CORE_HUGE)
    {
        t = fabs(x) + ACB_CORE_TINY;
        return x + t * ACB_CORE_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_CORE_HUGE;
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

static acb_cplx_t acb_div(acb_cplx_t a, acb_cplx_t b)
{
    double d = b.re * b.re + b.im * b.im;
    return acb_cplx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

static acb_cplx_t acb_log(acb_cplx_t z)
{
    return acb_cplx(log(hypot(z.re, z.im)), atan2(z.im, z.re));
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

static acb_cplx_t acb_sqrt(acb_cplx_t z)
{
    double r = hypot(z.re, z.im);
    double u = sqrt(fmax(0.0, 0.5 * (r + z.re)));
    double v = sqrt(fmax(0.0, 0.5 * (r - z.re)));
    if (z.im < 0.0)
        v = -v;
    return acb_cplx(u, v);
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_core_below(z.re), acb_core_above(z.re));
    out.imag = di_interval(acb_core_below(z.im), acb_core_above(z.im));
    return out;
}

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

acb_box_t acb_exp_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_exp(z);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_log_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_log(z);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_sqrt_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_sqrt(z);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_sin_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_sin(z);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_cos_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_cos(z);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_tan_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t s = acb_sin(z);
    acb_cplx_t c = acb_cos(z);
    acb_cplx_t v = acb_div(s, c);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_sinh_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_cplx(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im));
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_cosh_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t v = acb_cplx(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im));
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

acb_box_t acb_tanh_ref(acb_box_t x)
{
    acb_cplx_t z = acb_midpoint(x);
    acb_cplx_t s = acb_cplx(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im));
    acb_cplx_t c = acb_cplx(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im));
    acb_cplx_t v = acb_div(s, c);
    if (!isfinite(v.re) || !isfinite(v.im))
        return acb_full();
    return acb_from_complex(v);
}

void acb_exp_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_exp_ref(x[i]);
}

void acb_log_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_log_ref(x[i]);
}

void acb_sqrt_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_sqrt_ref(x[i]);
}

void acb_sin_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_sin_ref(x[i]);
}

void acb_cos_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_cos_ref(x[i]);
}

void acb_tan_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_tan_ref(x[i]);
}

void acb_sinh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_sinh_ref(x[i]);
}

void acb_cosh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_cosh_ref(x[i]);
}

void acb_tanh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_tanh_ref(x[i]);
}
