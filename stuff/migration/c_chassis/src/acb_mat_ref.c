#include "acb_mat_ref.h"

#include <math.h>

#define ACB_MAT_ULP_FACTOR 4.440892098500626e-16
#define ACB_MAT_HUGE 1e300
#define ACB_MAT_TINY 1e-300

typedef struct
{
    double re;
    double im;
} acb_cplx_t;

static double acb_mat_below(double x)
{
    double t;

    if (x <= ACB_MAT_HUGE)
    {
        t = fabs(x) + ACB_MAT_TINY;
        return x - t * ACB_MAT_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return ACB_MAT_HUGE;
    }
}

static double acb_mat_above(double x)
{
    double t;

    if (x >= -ACB_MAT_HUGE)
    {
        t = fabs(x) + ACB_MAT_TINY;
        return x + t * ACB_MAT_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -ACB_MAT_HUGE;
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

static acb_cplx_t acb_midpoint(acb_box_t x)
{
    return acb_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static acb_box_t acb_from_complex(acb_cplx_t z)
{
    acb_box_t out;
    out.real = di_interval(acb_mat_below(z.re), acb_mat_above(z.re));
    out.imag = di_interval(acb_mat_below(z.im), acb_mat_above(z.im));
    return out;
}

static acb_box_t acb_full(void)
{
    return (acb_box_t) { di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY) };
}

acb_box_t acb_mat_2x2_det_ref(const acb_box_t *a)
{
    acb_cplx_t a00 = acb_midpoint(a[0]);
    acb_cplx_t a01 = acb_midpoint(a[1]);
    acb_cplx_t a10 = acb_midpoint(a[2]);
    acb_cplx_t a11 = acb_midpoint(a[3]);

    acb_cplx_t det = acb_sub(acb_mul(a00, a11), acb_mul(a01, a10));
    if (!isfinite(det.re) || !isfinite(det.im))
        return acb_full();
    return acb_from_complex(det);
}

acb_box_t acb_mat_2x2_trace_ref(const acb_box_t *a)
{
    acb_cplx_t a00 = acb_midpoint(a[0]);
    acb_cplx_t a11 = acb_midpoint(a[3]);
    acb_cplx_t tr = acb_add(a00, a11);
    if (!isfinite(tr.re) || !isfinite(tr.im))
        return acb_full();
    return acb_from_complex(tr);
}

void acb_mat_2x2_det_batch_ref(const acb_box_t *a, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_mat_2x2_det_ref(a + 4 * i);
}

void acb_mat_2x2_trace_batch_ref(const acb_box_t *a, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_mat_2x2_trace_ref(a + 4 * i);
}
