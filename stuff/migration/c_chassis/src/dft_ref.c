#include "dft_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DFT_REF_PI 3.14159265358979323846

static cplx_t cplx_add(cplx_t a, cplx_t b)
{
    cplx_t z;
    z.re = a.re + b.re;
    z.im = a.im + b.im;
    return z;
}

static cplx_t cplx_sub(cplx_t a, cplx_t b)
{
    cplx_t z;
    z.re = a.re - b.re;
    z.im = a.im - b.im;
    return z;
}

static cplx_t cplx_mul(cplx_t a, cplx_t b)
{
    cplx_t z;
    z.re = a.re * b.re - a.im * b.im;
    z.im = a.re * b.im + a.im * b.re;
    return z;
}

static cplx_t cplx_scale(cplx_t a, double s)
{
    cplx_t z;
    z.re = a.re * s;
    z.im = a.im * s;
    return z;
}

static int is_pow2(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

static size_t bit_reverse(size_t x, unsigned int bits)
{
    size_t y = 0;
    unsigned int i;
    for (i = 0; i < bits; i++)
    {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

static void dft_naive_impl(const cplx_t *x, cplx_t *out, size_t n, int inverse)
{
    size_t k, t;
    double sign = inverse ? 1.0 : -1.0;
    for (k = 0; k < n; k++)
    {
        cplx_t acc = cplx_make(0.0, 0.0);
        for (t = 0; t < n; t++)
        {
            double ang = sign * 2.0 * DFT_REF_PI * (double) (k * t) / (double) n;
            cplx_t w = cplx_make(cos(ang), sin(ang));
            acc = cplx_add(acc, cplx_mul(x[t], w));
        }
        if (inverse)
            out[k] = cplx_scale(acc, 1.0 / (double) n);
        else
            out[k] = acc;
    }
}

static void dft_rad2_impl(const cplx_t *x, cplx_t *out, size_t n, int inverse)
{
    size_t i, len, half, j, k;
    unsigned int bits = 0;
    cplx_t *a;
    double sign = inverse ? 1.0 : -1.0;

    if (!is_pow2(n))
    {
        dft_naive_impl(x, out, n, inverse);
        return;
    }

    while (((size_t) 1 << bits) < n)
        bits++;

    a = (cplx_t *) malloc(sizeof(cplx_t) * n);
    if (a == NULL)
        return;

    for (i = 0; i < n; i++)
    {
        size_t r = bit_reverse(i, bits);
        a[r] = x[i];
    }

    for (len = 2; len <= n; len <<= 1)
    {
        half = len >> 1;
        for (i = 0; i < n; i += len)
        {
            for (j = 0; j < half; j++)
            {
                double ang = sign * 2.0 * DFT_REF_PI * (double) j / (double) len;
                cplx_t w = cplx_make(cos(ang), sin(ang));
                cplx_t u = a[i + j];
                cplx_t v = cplx_mul(a[i + j + half], w);
                a[i + j] = cplx_add(u, v);
                a[i + j + half] = cplx_sub(u, v);
            }
        }
    }

    if (inverse)
    {
        for (k = 0; k < n; k++)
            out[k] = cplx_scale(a[k], 1.0 / (double) n);
    }
    else
    {
        for (k = 0; k < n; k++)
            out[k] = a[k];
    }

    free(a);
}

cplx_t cplx_make(double re, double im)
{
    cplx_t z;
    z.re = re;
    z.im = im;
    return z;
}

dft_acb_box_t dft_acb_box(di_t real, di_t imag)
{
    dft_acb_box_t z;
    z.real = real;
    z.imag = imag;
    return z;
}

static dft_acb_box_t acb_add(dft_acb_box_t a, dft_acb_box_t b)
{
    return dft_acb_box(di_fast_add(a.real, b.real), di_fast_add(a.imag, b.imag));
}

static dft_acb_box_t acb_mul(dft_acb_box_t a, dft_acb_box_t b)
{
    di_t ac = di_fast_mul(a.real, b.real);
    di_t bd = di_fast_mul(a.imag, b.imag);
    di_t ad = di_fast_mul(a.real, b.imag);
    di_t bc = di_fast_mul(a.imag, b.real);
    return dft_acb_box(di_fast_sub(ac, bd), di_fast_add(ad, bc));
}

static dft_acb_box_t acb_scale_real(dft_acb_box_t a, double s)
{
    di_t si = di_interval(s, s);
    return dft_acb_box(di_fast_mul(a.real, si), di_fast_mul(a.imag, si));
}

static dft_acb_box_t acb_twiddle(size_t k, size_t t, size_t n, int inverse)
{
    double sign = inverse ? 1.0 : -1.0;
    double ang = sign * 2.0 * DFT_REF_PI * (double) (k * t) / (double) n;
    return dft_acb_box(di_interval(cos(ang), cos(ang)), di_interval(sin(ang), sin(ang)));
}

static void acb_dft_naive_impl(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n, int inverse)
{
    size_t k, t;
    for (k = 0; k < n; k++)
    {
        dft_acb_box_t acc = dft_acb_box(di_interval(0.0, 0.0), di_interval(0.0, 0.0));
        for (t = 0; t < n; t++)
        {
            dft_acb_box_t w = acb_twiddle(k, t, n, inverse);
            acc = acb_add(acc, acb_mul(x[t], w));
        }
        if (inverse)
            out[k] = acb_scale_real(acc, 1.0 / (double) n);
        else
            out[k] = acc;
    }
}

void cplx_dft_naive_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    dft_naive_impl(x, out, n, 0);
}

void cplx_idft_naive_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    dft_naive_impl(x, out, n, 1);
}

void cplx_dft_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    if (is_pow2(n))
        dft_rad2_impl(x, out, n, 0);
    else
        dft_naive_impl(x, out, n, 0);
}

void cplx_idft_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    if (is_pow2(n))
        dft_rad2_impl(x, out, n, 1);
    else
        dft_naive_impl(x, out, n, 1);
}

void cplx_dft_prod_ref(const cplx_t *x, cplx_t *out, const size_t *cyc, size_t num)
{
    size_t i, n = 1;
    for (i = 0; i < num; i++)
        n *= cyc[i];
    cplx_dft_ref(x, out, n);
}

void cplx_convol_circular_naive_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n)
{
    size_t x, y;
    for (x = 0; x < n; x++)
    {
        cplx_t acc = cplx_make(0.0, 0.0);
        for (y = 0; y < n; y++)
        {
            size_t idx = (x + n - y) % n;
            acc = cplx_add(acc, cplx_mul(f[idx], g[y]));
        }
        out[x] = acc;
    }
}

void cplx_convol_circular_dft_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n)
{
    size_t i;
    cplx_t *fd = (cplx_t *) malloc(sizeof(cplx_t) * n);
    cplx_t *gd = (cplx_t *) malloc(sizeof(cplx_t) * n);

    if (fd == NULL || gd == NULL)
    {
        free(fd);
        free(gd);
        return;
    }

    cplx_dft_ref(f, fd, n);
    cplx_dft_ref(g, gd, n);
    for (i = 0; i < n; i++)
        gd[i] = cplx_mul(fd[i], gd[i]);
    cplx_idft_ref(gd, out, n);

    free(fd);
    free(gd);
}

void cplx_convol_circular_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n)
{
    if (is_pow2(n))
        cplx_convol_circular_rad2_ref(f, g, out, n);
    else
        cplx_convol_circular_dft_ref(f, g, out, n);
}

void cplx_dft_rad2_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    dft_rad2_impl(x, out, n, 0);
}

void cplx_idft_rad2_ref(const cplx_t *x, cplx_t *out, size_t n)
{
    dft_rad2_impl(x, out, n, 1);
}

void cplx_convol_circular_rad2_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n)
{
    size_t i;
    cplx_t *fd = (cplx_t *) malloc(sizeof(cplx_t) * n);
    cplx_t *gd = (cplx_t *) malloc(sizeof(cplx_t) * n);
    if (fd == NULL || gd == NULL)
    {
        free(fd);
        free(gd);
        return;
    }

    cplx_dft_rad2_ref(f, fd, n);
    cplx_dft_rad2_ref(g, gd, n);
    for (i = 0; i < n; i++)
        gd[i] = cplx_mul(fd[i], gd[i]);
    cplx_idft_rad2_ref(gd, out, n);

    free(fd);
    free(gd);
}

void acb_dft_naive_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 0);
}

void acb_idft_naive_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 1);
}

void acb_dft_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 0);
}

void acb_idft_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 1);
}

void acb_dft_prod_ref(const dft_acb_box_t *x, dft_acb_box_t *out, const size_t *cyc, size_t num)
{
    size_t i, n = 1;
    for (i = 0; i < num; i++)
        n *= cyc[i];
    acb_dft_ref(x, out, n);
}

void acb_convol_circular_naive_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n)
{
    size_t x, y;
    for (x = 0; x < n; x++)
    {
        dft_acb_box_t acc = dft_acb_box(di_interval(0.0, 0.0), di_interval(0.0, 0.0));
        for (y = 0; y < n; y++)
        {
            size_t idx = (x + n - y) % n;
            acc = acb_add(acc, acb_mul(f[idx], g[y]));
        }
        out[x] = acc;
    }
}

void acb_convol_circular_dft_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n)
{
    size_t i;
    dft_acb_box_t *fd = (dft_acb_box_t *) malloc(sizeof(dft_acb_box_t) * n);
    dft_acb_box_t *gd = (dft_acb_box_t *) malloc(sizeof(dft_acb_box_t) * n);

    if (fd == NULL || gd == NULL)
    {
        free(fd);
        free(gd);
        return;
    }

    acb_dft_ref(f, fd, n);
    acb_dft_ref(g, gd, n);
    for (i = 0; i < n; i++)
        gd[i] = acb_mul(fd[i], gd[i]);
    acb_idft_ref(gd, out, n);

    free(fd);
    free(gd);
}

void acb_convol_circular_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n)
{
    if (is_pow2(n))
        acb_convol_circular_rad2_ref(f, g, out, n);
    else
        acb_convol_circular_dft_ref(f, g, out, n);
}

void acb_dft_rad2_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 0);
}

void acb_idft_rad2_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n)
{
    acb_dft_naive_impl(x, out, n, 1);
}

void acb_convol_circular_rad2_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n)
{
    size_t i;
    dft_acb_box_t *fd = (dft_acb_box_t *) malloc(sizeof(dft_acb_box_t) * n);
    dft_acb_box_t *gd = (dft_acb_box_t *) malloc(sizeof(dft_acb_box_t) * n);

    if (fd == NULL || gd == NULL)
    {
        free(fd);
        free(gd);
        return;
    }

    acb_dft_rad2_ref(f, fd, n);
    acb_dft_rad2_ref(g, gd, n);
    for (i = 0; i < n; i++)
        gd[i] = acb_mul(fd[i], gd[i]);
    acb_idft_rad2_ref(gd, out, n);

    free(fd);
    free(gd);
}
