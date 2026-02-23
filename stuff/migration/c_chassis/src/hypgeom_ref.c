#include "hypgeom_ref.h"

#include <math.h>

#define HG_ULP_FACTOR 4.440892098500626e-16
#define HG_HUGE 1e300
#define HG_TINY 1e-300
#define HG_DIGAMMA_ZERO 1.4616321449683623413
#define HG_PI 3.14159265358979323846
#define HG_TWO_OVER_SQRT_PI 1.12837916709551257390
#define HG_ERF_TERMS 48
#define HG_HYP_TERMS 80
#define HG_BESSEL_TERMS 60
#define HG_SIN_EPS 1e-8

static int HG_BESSEL_REAL_MODE = 1;

typedef struct
{
    double re;
    double im;
} hg_cplx_t;

static acb_box_t acb_from_complex(hg_cplx_t z);
static hg_cplx_t acb_mid(acb_box_t x);

static double hg_below(double x)
{
    double t;

    if (x <= HG_HUGE)
    {
        t = fabs(x) + HG_TINY;
        return x - t * HG_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return -INFINITY;
        return HG_HUGE;
    }
}

static double hg_above(double x)
{
    double t;

    if (x >= -HG_HUGE)
    {
        t = fabs(x) + HG_TINY;
        return x + t * HG_ULP_FACTOR;
    }
    else
    {
        if (isnan(x))
            return INFINITY;
        return -HG_HUGE;
    }
}

static double hg_min2(double a, double b)
{
    return a < b ? a : b;
}

static double hg_max2(double a, double b)
{
    return a > b ? a : b;
}

void hypgeom_ref_set_bessel_real_mode(int mode)
{
    HG_BESSEL_REAL_MODE = (mode == 0) ? 0 : 1;
}

static hg_cplx_t hg_cplx(double re, double im)
{
    hg_cplx_t out;
    out.re = re;
    out.im = im;
    return out;
}

static hg_cplx_t hg_add(hg_cplx_t a, hg_cplx_t b)
{
    return hg_cplx(a.re + b.re, a.im + b.im);
}

static hg_cplx_t hg_sub(hg_cplx_t a, hg_cplx_t b)
{
    return hg_cplx(a.re - b.re, a.im - b.im);
}

static hg_cplx_t hg_mul(hg_cplx_t a, hg_cplx_t b)
{
    return hg_cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

static hg_cplx_t hg_scale(hg_cplx_t a, double c)
{
    return hg_cplx(a.re * c, a.im * c);
}

static hg_cplx_t hg_div(hg_cplx_t a, hg_cplx_t b)
{
    double d = b.re * b.re + b.im * b.im;
    return hg_cplx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

static hg_cplx_t hg_log(hg_cplx_t z)
{
    return hg_cplx(log(hypot(z.re, z.im)), atan2(z.im, z.re));
}

static hg_cplx_t hg_exp(hg_cplx_t z)
{
    double e = exp(z.re);
    return hg_cplx(e * cos(z.im), e * sin(z.im));
}

static hg_cplx_t hg_sin(hg_cplx_t z)
{
    return hg_cplx(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

static hg_cplx_t hg_cos(hg_cplx_t z)
{
    return hg_cplx(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}

static hg_cplx_t hg_pow(hg_cplx_t a, hg_cplx_t b)
{
    return hg_exp(hg_mul(b, hg_log(a)));
}

static hg_cplx_t hg_erf_series(hg_cplx_t z)
{
    hg_cplx_t z2 = hg_mul(z, z);
    hg_cplx_t term = z;
    hg_cplx_t sum = term;
    int k;

    for (k = 0; k < HG_ERF_TERMS - 1; k++)
    {
        double den = (double) (k + 1) * (double) (2 * k + 3);
        term = hg_scale(hg_mul(term, hg_scale(z2, -1.0)), 1.0 / den);
        sum = hg_add(sum, term);
    }

    return hg_scale(sum, HG_TWO_OVER_SQRT_PI);
}

static hg_cplx_t hg_erfc_point(hg_cplx_t z)
{
    return hg_sub(hg_cplx(1.0, 0.0), hg_erf_series(z));
}

static hg_cplx_t hg_erfi_point(hg_cplx_t z)
{
    hg_cplx_t iz = hg_cplx(-z.im, z.re);
    hg_cplx_t w = hg_erf_series(iz);
    return hg_cplx(w.im, -w.re); /* -i * w */
}

static double hg_erfinv_point(double y)
{
    const double a = 0.147;
    double ln;
    double t;
    double x;
    int k;

    if (!isfinite(y) || y <= -1.0 || y >= 1.0)
        return NAN;

    ln = log(1.0 - y * y);
    t = 2.0 / (HG_PI * a) + 0.5 * ln;
    x = sqrt(fmax(0.0, sqrt(t * t - ln / a) - t));
    if (y < 0.0)
        x = -x;

    for (k = 0; k < 3; k++)
    {
        double err = erf(x) - y;
        double der = (2.0 / sqrt(HG_PI)) * exp(-x * x);
        x -= err / der;
    }

    return x;
}

static hg_cplx_t hg_log_gamma_lanczos(hg_cplx_t z)
{
    static const double c[9] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    const double g = 7.0;
    hg_cplx_t z1 = hg_sub(z, hg_cplx(1.0, 0.0));
    hg_cplx_t x = hg_cplx(c[0], 0.0);
    hg_cplx_t t;
    hg_cplx_t a, b;
    int i;

    for (i = 1; i <= 8; i++)
        x = hg_add(x, hg_div(hg_cplx(c[i], 0.0), hg_add(z1, hg_cplx((double) i, 0.0))));

    t = hg_add(z1, hg_cplx(g + 0.5, 0.0));
    a = hg_mul(hg_add(z1, hg_cplx(0.5, 0.0)), hg_log(t));
    b = hg_sub(a, t);
    b = hg_add(b, hg_log(x));
    return hg_add(b, hg_cplx(0.91893853320467274178, 0.0)); /* 0.5 * log(2*pi) */
}

static hg_cplx_t hg_log_gamma(hg_cplx_t z)
{
    if (z.re < 0.5)
    {
        hg_cplx_t pi_z = hg_mul(hg_cplx(HG_PI, 0.0), z);
        hg_cplx_t one_minus_z = hg_sub(hg_cplx(1.0, 0.0), z);
        return hg_sub(
            hg_sub(hg_cplx(log(HG_PI), 0.0), hg_log(hg_sin(pi_z))),
            hg_log_gamma_lanczos(one_minus_z));
    }

    return hg_log_gamma_lanczos(z);
}

static hg_cplx_t hg_bessel_series(hg_cplx_t nu, hg_cplx_t z, int sign)
{
    hg_cplx_t half = hg_scale(z, 0.5);
    hg_cplx_t pow_half = hg_pow(half, nu);
    hg_cplx_t gamma = hg_exp(hg_log_gamma(hg_add(nu, hg_cplx(1.0, 0.0))));
    hg_cplx_t term = hg_div(pow_half, gamma);
    hg_cplx_t sum = term;
    hg_cplx_t z2 = hg_mul(z, z);
    int k;

    for (k = 0; k < HG_BESSEL_TERMS - 1; k++)
    {
        double k1 = (double) (k + 1);
        hg_cplx_t den = hg_mul(hg_cplx(k1, 0.0), hg_add(nu, hg_cplx(k1, 0.0)));
        hg_cplx_t num = hg_scale(z2, 0.25 * (double) sign);
        term = hg_mul(term, num);
        term = hg_div(term, den);
        sum = hg_add(sum, term);
    }

    return sum;
}

static double hg_hyp1f1_real(double a, double b, double z)
{
    double term = 1.0;
    double sum = 1.0;
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        double k1 = (double) (k + 1);
        term *= (a + (double) k) / (b + (double) k);
        term *= z / k1;
        sum += term;
    }

    return sum;
}

static hg_cplx_t hg_hyp1f1_cplx(hg_cplx_t a, hg_cplx_t b, hg_cplx_t z)
{
    hg_cplx_t term = hg_cplx(1.0, 0.0);
    hg_cplx_t sum = term;
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        double k1 = (double) (k + 1);
        hg_cplx_t ak = hg_add(a, hg_cplx((double) k, 0.0));
        hg_cplx_t bk = hg_add(b, hg_cplx((double) k, 0.0));
        term = hg_mul(term, hg_div(ak, bk));
        term = hg_mul(term, z);
        term = hg_scale(term, 1.0 / k1);
        sum = hg_add(sum, term);
    }

    return sum;
}

static double hg_hypu_real(double a, double b, double z)
{
    double s = sin(HG_PI * b);
    double m1, m2;
    double t1, t2;

    if (fabs(s) < HG_SIN_EPS)
        return NAN;

    m1 = hg_hyp1f1_real(a, b, z);
    m2 = hg_hyp1f1_real(1.0 + a - b, 2.0 - b, z);
    t1 = m1 / exp(lgamma(1.0 + a - b));
    t2 = pow(z, 1.0 - b) * m2 / exp(lgamma(a));
    return HG_PI * (t1 - t2) / s;
}

static hg_cplx_t hg_hypu_cplx(hg_cplx_t a, hg_cplx_t b, hg_cplx_t z)
{
    hg_cplx_t s = hg_sin(hg_scale(b, HG_PI));
    hg_cplx_t m1;
    hg_cplx_t m2;
    hg_cplx_t t1;
    hg_cplx_t t2;
    hg_cplx_t zpow;
    hg_cplx_t ga;
    hg_cplx_t gb;
    double sn = hypot(s.re, s.im);

    if (sn < HG_SIN_EPS)
        return hg_cplx(NAN, NAN);

    m1 = hg_hyp1f1_cplx(a, b, z);
    m2 = hg_hyp1f1_cplx(hg_add(hg_sub(a, b), hg_cplx(1.0, 0.0)), hg_sub(hg_cplx(2.0, 0.0), b), z);
    ga = hg_exp(hg_log_gamma(a));
    gb = hg_exp(hg_log_gamma(hg_add(hg_sub(a, b), hg_cplx(1.0, 0.0))));
    t1 = hg_div(m1, gb);
    zpow = hg_pow(z, hg_sub(hg_cplx(1.0, 0.0), b));
    t2 = hg_div(hg_mul(zpow, m2), ga);
    return hg_div(hg_scale(hg_sub(t1, t2), HG_PI), s);
}

static double hg_bessel_series_real(double nu, double z, int sign)
{
    double half = 0.5 * z;
    double term = exp(nu * log(half) - lgamma(nu + 1.0));
    double sum = term;
    double z2 = z * z;
    int k;

    for (k = 0; k < HG_BESSEL_TERMS - 1; k++)
    {
        double k1 = (double) (k + 1);
        double den = k1 * (k1 + nu);
        double num = 0.25 * (double) sign * z2;
        term *= num / den;
        sum += term;
    }

    return sum;
}

static double hg_bessel_y_point(double nu, double z)
{
    double jnu = hg_bessel_series_real(nu, z, -1);
    double jneg = hg_bessel_series_real(-nu, z, -1);
    double s = sin(HG_PI * nu);
    double c = cos(HG_PI * nu);
    if (fabs(s) < HG_SIN_EPS)
        return NAN;
    return (jnu * c - jneg) / s;
}

static double hg_bessel_k_point(double nu, double z)
{
    double inu = hg_bessel_series_real(nu, z, 1);
    double ineg = hg_bessel_series_real(-nu, z, 1);
    double s = sin(HG_PI * nu);
    if (fabs(s) < HG_SIN_EPS)
        return NAN;
    return 0.5 * HG_PI * (ineg - inu) / s;
}

static int hg_interval_contains_nonpositive_integer(double a, double b)
{
    double kmin = ceil(-b);
    double kmax = floor(-a);
    return (kmin <= kmax) && (kmax >= 0.0);
}

acb_box_t acb_box(di_t real, di_t imag)
{
    acb_box_t out;
    out.real = real;
    out.imag = imag;
    return out;
}

acb_box_t acb_box_add_ui(acb_box_t x, unsigned long long k)
{
    di_t k_interval = di_interval((double) k, (double) k);
    x.real = di_fast_add(x.real, k_interval);
    return x;
}

static acb_box_t acb_box_add(acb_box_t x, acb_box_t y)
{
    acb_box_t out;
    out.real = di_fast_add(x.real, y.real);
    out.imag = di_fast_add(x.imag, y.imag);
    return out;
}

acb_box_t acb_box_mul(acb_box_t x, acb_box_t y)
{
    di_t ac = di_fast_mul(x.real, y.real);
    di_t bd = di_fast_mul(x.imag, y.imag);
    di_t ad = di_fast_mul(x.real, y.imag);
    di_t bc = di_fast_mul(x.imag, y.real);
    acb_box_t out;

    out.real = di_fast_sub(ac, bd);
    out.imag = di_fast_add(ad, bc);
    return out;
}

static acb_box_t acb_box_scale_real(acb_box_t x, di_t r)
{
    acb_box_t out;
    out.real = di_fast_mul(x.real, r);
    out.imag = di_fast_mul(x.imag, r);
    return out;
}

static acb_box_t acb_box_div(acb_box_t x, acb_box_t y)
{
    di_t den = di_fast_add(di_fast_mul(y.real, y.real), di_fast_mul(y.imag, y.imag));
    acb_box_t num = acb_box(y.real, di_neg(y.imag));
    acb_box_t out;
    out = acb_box_mul(x, num);
    out.real = di_fast_div(out.real, den);
    out.imag = di_fast_div(out.imag, den);
    return out;
}

di_t arb_hypgeom_rising_ui_forward_ref(di_t x, unsigned long long n)
{
    di_t res = di_interval(1.0, 1.0);
    unsigned long long k;

    for (k = 0; k < n; k++)
    {
        di_t t = di_fast_add(x, di_interval((double) k, (double) k));
        res = di_fast_mul(res, t);
    }

    return res;
}

di_t arb_hypgeom_rising_ui_ref(di_t x, unsigned long long n)
{
    return arb_hypgeom_rising_ui_forward_ref(x, n);
}

di_t arb_hypgeom_lgamma_ref(di_t x)
{
    di_t out;
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double vals[4];
    int count = 0;
    int i;
    double lo, hi;

    if (!(a > 0.0))
        return di_interval(-INFINITY, INFINITY);

    vals[count++] = lgamma(a);
    vals[count++] = lgamma(b);
    vals[count++] = lgamma(m);

    if (a <= HG_DIGAMMA_ZERO && HG_DIGAMMA_ZERO <= b)
        vals[count++] = lgamma(HG_DIGAMMA_ZERO);

    lo = vals[0];
    hi = vals[0];
    for (i = 1; i < count; i++)
    {
        lo = hg_min2(lo, vals[i]);
        hi = hg_max2(hi, vals[i]);
    }

    if (!isfinite(lo) || !isfinite(hi))
        return di_interval(-INFINITY, INFINITY);

    out.a = hg_below(lo);
    out.b = hg_above(hi);
    return out;
}

di_t arb_hypgeom_gamma_ref(di_t x)
{
    di_t out;
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double vals[4];
    int count = 0;
    int i;
    double lo, hi;

    if (!(a > 0.0))
        return di_interval(-INFINITY, INFINITY);

    vals[count++] = tgamma(a);
    vals[count++] = tgamma(b);
    vals[count++] = tgamma(m);

    if (a <= HG_DIGAMMA_ZERO && HG_DIGAMMA_ZERO <= b)
        vals[count++] = tgamma(HG_DIGAMMA_ZERO);

    lo = vals[0];
    hi = vals[0];
    for (i = 1; i < count; i++)
    {
        lo = hg_min2(lo, vals[i]);
        hi = hg_max2(hi, vals[i]);
    }

    if (!isfinite(lo) || !isfinite(hi))
        return di_interval(-INFINITY, INFINITY);

    out.a = hg_below(lo);
    out.b = hg_above(hi);
    return out;
}

di_t arb_hypgeom_rgamma_ref(di_t x)
{
    di_t g = arb_hypgeom_gamma_ref(x);
    di_t one = di_interval(1.0, 1.0);
    return di_fast_div(one, g);
}

di_t arb_hypgeom_erf_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double v1 = hg_erf_series(hg_cplx(a, 0.0)).re;
    double v2 = hg_erf_series(hg_cplx(b, 0.0)).re;
    double v3 = hg_erf_series(hg_cplx(m, 0.0)).re;
    double lo = hg_min2(hg_min2(v1, v2), v3);
    double hi = hg_max2(hg_max2(v1, v2), v3);
    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_erfc_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double v1 = hg_erfc_point(hg_cplx(a, 0.0)).re;
    double v2 = hg_erfc_point(hg_cplx(b, 0.0)).re;
    double v3 = hg_erfc_point(hg_cplx(m, 0.0)).re;
    double lo = hg_min2(hg_min2(v1, v2), v3);
    double hi = hg_max2(hg_max2(v1, v2), v3);
    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_erfi_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double v1 = hg_erfi_point(hg_cplx(a, 0.0)).re;
    double v2 = hg_erfi_point(hg_cplx(b, 0.0)).re;
    double v3 = hg_erfi_point(hg_cplx(m, 0.0)).re;
    double lo = hg_min2(hg_min2(v1, v2), v3);
    double hi = hg_max2(hg_max2(v1, v2), v3);
    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_erfinv_ref(di_t x)
{
    double a = x.a;
    double b = x.b;
    double m = 0.5 * (a + b);
    double vals[3];
    double lo, hi;
    int i;

    vals[0] = hg_erfinv_point(a);
    vals[1] = hg_erfinv_point(b);
    vals[2] = hg_erfinv_point(m);

    lo = vals[0];
    hi = vals[0];
    for (i = 0; i < 3; i++)
    {
        if (!isfinite(vals[i]))
            return di_interval(-INFINITY, INFINITY);
        lo = hg_min2(lo, vals[i]);
        hi = hg_max2(hi, vals[i]);
    }

    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_erfcinv_ref(di_t x)
{
    di_t y;
    y.a = 1.0 - x.b;
    y.b = 1.0 - x.a;
    return arb_hypgeom_erfinv_ref(y);
}

di_t arb_hypgeom_0f1_ref(di_t a, di_t z, int regularized)
{
    di_t term = di_interval(1.0, 1.0);
    di_t sum = di_interval(1.0, 1.0);
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        di_t ak = di_fast_add(a, di_interval((double) k, (double) k));
        di_t inv_k1 = di_interval(1.0 / (double) (k + 1), 1.0 / (double) (k + 1));
        di_t step = di_fast_div(z, ak);
        step = di_fast_mul(step, inv_k1);
        term = di_fast_mul(term, step);
        sum = di_fast_add(sum, term);
    }

    if (regularized)
        sum = di_fast_mul(sum, arb_hypgeom_rgamma_ref(a));

    return sum;
}

di_t arb_hypgeom_1f1_ref(di_t a, di_t b, di_t z)
{
    di_t term = di_interval(1.0, 1.0);
    di_t sum = di_interval(1.0, 1.0);
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        di_t ak = di_fast_add(a, di_interval((double) k, (double) k));
        di_t bk = di_fast_add(b, di_interval((double) k, (double) k));
        di_t step = di_fast_div(ak, bk);
        di_t inv_k1 = di_interval(1.0 / (double) (k + 1), 1.0 / (double) (k + 1));

        term = di_fast_mul(term, step);
        term = di_fast_mul(term, z);
        term = di_fast_mul(term, inv_k1);
        sum = di_fast_add(sum, term);
    }

    return sum;
}

di_t arb_hypgeom_1f1_full_ref(di_t a, di_t b, di_t z, int regularized)
{
    di_t res = arb_hypgeom_1f1_ref(a, b, z);
    if (regularized)
        res = di_fast_mul(res, arb_hypgeom_rgamma_ref(b));
    return res;
}

di_t arb_hypgeom_1f1_integration_ref(di_t a, di_t b, di_t z, int regularized)
{
    return arb_hypgeom_1f1_full_ref(a, b, z, regularized);
}

di_t arb_hypgeom_m_ref(di_t a, di_t b, di_t z, int regularized)
{
    return arb_hypgeom_1f1_full_ref(a, b, z, regularized);
}

di_t arb_hypgeom_2f1_ref(di_t a, di_t b, di_t c, di_t z)
{
    di_t term = di_interval(1.0, 1.0);
    di_t sum = di_interval(1.0, 1.0);
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        di_t ak = di_fast_add(a, di_interval((double) k, (double) k));
        di_t bk = di_fast_add(b, di_interval((double) k, (double) k));
        di_t ck = di_fast_add(c, di_interval((double) k, (double) k));
        di_t k1 = di_interval((double) (k + 1), (double) (k + 1));
        di_t num = di_fast_mul(ak, bk);
        di_t den = di_fast_mul(ck, k1);
        di_t step = di_fast_div(num, den);

        term = di_fast_mul(term, step);
        term = di_fast_mul(term, z);
        sum = di_fast_add(sum, term);
    }

    return sum;
}

di_t arb_hypgeom_2f1_full_ref(di_t a, di_t b, di_t c, di_t z, int regularized)
{
    di_t res = arb_hypgeom_2f1_ref(a, b, c, z);
    if (regularized)
        res = di_fast_mul(res, arb_hypgeom_rgamma_ref(c));
    return res;
}

di_t arb_hypgeom_2f1_integration_ref(di_t a, di_t b, di_t c, di_t z, int regularized)
{
    return arb_hypgeom_2f1_full_ref(a, b, c, z, regularized);
}

di_t arb_hypgeom_u_ref(di_t a, di_t b, di_t z)
{
    double a_vals[3];
    double b_vals[3];
    double z_vals[3];
    double v;
    double lo, hi;
    int i, j, k;

    a_vals[0] = a.a;
    a_vals[1] = a.b;
    a_vals[2] = 0.5 * (a.a + a.b);
    b_vals[0] = b.a;
    b_vals[1] = b.b;
    b_vals[2] = 0.5 * (b.a + b.b);
    z_vals[0] = z.a;
    z_vals[1] = z.b;
    z_vals[2] = 0.5 * (z.a + z.b);

    v = hg_hypu_real(a_vals[0], b_vals[0], z_vals[0]);
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    lo = v;
    hi = v;

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                v = hg_hypu_real(a_vals[i], b_vals[j], z_vals[k]);
                if (!isfinite(v))
                    return di_interval(-INFINITY, INFINITY);
                lo = hg_min2(lo, v);
                hi = hg_max2(hi, v);
            }
        }
    }

    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_u_integration_ref(di_t a, di_t b, di_t z)
{
    return arb_hypgeom_u_ref(a, b, z);
}

static di_t arb_bessel_interval(di_t nu, di_t z, int kind)
{
    double nu_m = 0.5 * (nu.a + nu.b);
    double a = z.a;
    double b = z.b;
    double m = 0.5 * (a + b);
    double v1, v2, v3;

    if (kind == 0)
    {
        v1 = hg_bessel_series_real(nu_m, a, -1);
        v2 = hg_bessel_series_real(nu_m, b, -1);
        v3 = hg_bessel_series_real(nu_m, m, -1);
    }
    else if (kind == 1)
    {
        v1 = hg_bessel_series_real(nu_m, a, 1);
        v2 = hg_bessel_series_real(nu_m, b, 1);
        v3 = hg_bessel_series_real(nu_m, m, 1);
    }
    else
    {
        double jnu_m = hg_bessel_series_real(nu_m, m, -1);
        double jneg_m = hg_bessel_series_real(-nu_m, m, -1);
        double c = cos(HG_PI * nu_m);
        double s = sin(HG_PI * nu_m);
        if (fabs(s) < HG_SIN_EPS)
            return di_interval(-INFINITY, INFINITY);
        v3 = (jnu_m * c - jneg_m) / s;
        v1 = v3;
        v2 = v3;
    }

    if (!isfinite(v1) || !isfinite(v2) || !isfinite(v3))
        return di_interval(-INFINITY, INFINITY);

    double lo = hg_min2(hg_min2(v1, v2), v3);
    double hi = hg_max2(hg_max2(v1, v2), v3);
    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_bessel_j_ref(di_t nu, di_t z)
{
    if (HG_BESSEL_REAL_MODE == 0)
    {
        double nu_m = 0.5 * (nu.a + nu.b);
        double z_m = 0.5 * (z.a + z.b);
        double v = hg_bessel_series_real(nu_m, z_m, -1);
        if (!isfinite(v))
            return di_interval(-INFINITY, INFINITY);
        return di_interval(hg_below(v), hg_above(v));
    }
    return arb_bessel_interval(nu, z, 0);
}

di_t arb_hypgeom_bessel_i_ref(di_t nu, di_t z)
{
    if (HG_BESSEL_REAL_MODE == 0)
    {
        double nu_m = 0.5 * (nu.a + nu.b);
        double z_m = 0.5 * (z.a + z.b);
        double v = hg_bessel_series_real(nu_m, z_m, 1);
        if (!isfinite(v))
            return di_interval(-INFINITY, INFINITY);
        return di_interval(hg_below(v), hg_above(v));
    }
    return arb_bessel_interval(nu, z, 1);
}

di_t arb_hypgeom_bessel_y_ref(di_t nu, di_t z)
{
    if (HG_BESSEL_REAL_MODE == 0)
    {
        double nu_m = 0.5 * (nu.a + nu.b);
        double z_m = 0.5 * (z.a + z.b);
        double v = hg_bessel_y_point(nu_m, z_m);
        if (!isfinite(v))
            return di_interval(-INFINITY, INFINITY);
        return di_interval(hg_below(v), hg_above(v));
    }
    double nu_vals[3];
    double z_vals[3];
    double v;
    double lo, hi;
    int i, j;

    nu_vals[0] = nu.a;
    nu_vals[1] = nu.b;
    nu_vals[2] = 0.5 * (nu.a + nu.b);
    z_vals[0] = z.a;
    z_vals[1] = z.b;
    z_vals[2] = 0.5 * (z.a + z.b);

    v = hg_bessel_y_point(nu_vals[0], z_vals[0]);
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    lo = v;
    hi = v;

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            v = hg_bessel_y_point(nu_vals[i], z_vals[j]);
            if (!isfinite(v))
                return di_interval(-INFINITY, INFINITY);
            lo = hg_min2(lo, v);
            hi = hg_max2(hi, v);
        }
    }

    return di_interval(hg_below(lo), hg_above(hi));
}

void arb_hypgeom_bessel_jy_ref(di_t *res1, di_t *res2, di_t nu, di_t z)
{
    *res1 = arb_hypgeom_bessel_j_ref(nu, z);
    *res2 = arb_hypgeom_bessel_y_ref(nu, z);
}

di_t arb_hypgeom_bessel_k_ref(di_t nu, di_t z)
{
    if (HG_BESSEL_REAL_MODE == 0)
    {
        double nu_m = 0.5 * (nu.a + nu.b);
        double z_m = 0.5 * (z.a + z.b);
        double v = hg_bessel_k_point(nu_m, z_m);
        if (!isfinite(v))
            return di_interval(-INFINITY, INFINITY);
        return di_interval(hg_below(v), hg_above(v));
    }
    double nu_vals[3];
    double z_vals[3];
    double v;
    double lo, hi;
    int i, j;

    nu_vals[0] = nu.a;
    nu_vals[1] = nu.b;
    nu_vals[2] = 0.5 * (nu.a + nu.b);
    z_vals[0] = z.a;
    z_vals[1] = z.b;
    z_vals[2] = 0.5 * (z.a + z.b);

    v = hg_bessel_k_point(nu_vals[0], z_vals[0]);
    if (!isfinite(v))
        return di_interval(-INFINITY, INFINITY);
    lo = v;
    hi = v;

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            v = hg_bessel_k_point(nu_vals[i], z_vals[j]);
            if (!isfinite(v))
                return di_interval(-INFINITY, INFINITY);
            lo = hg_min2(lo, v);
            hi = hg_max2(hi, v);
        }
    }

    return di_interval(hg_below(lo), hg_above(hi));
}

di_t arb_hypgeom_bessel_i_scaled_ref(di_t nu, di_t z)
{
    di_t i = arb_hypgeom_bessel_i_ref(nu, z);
    double m = 0.5 * (z.a + z.b);
    di_t s = di_interval(exp(-m), exp(-m));
    return di_fast_mul(i, s);
}

di_t arb_hypgeom_bessel_k_scaled_ref(di_t nu, di_t z)
{
    di_t k = arb_hypgeom_bessel_k_ref(nu, z);
    double m = 0.5 * (z.a + z.b);
    di_t s = di_interval(exp(m), exp(m));
    return di_fast_mul(k, s);
}

di_t arb_hypgeom_bessel_i_integration_ref(di_t nu, di_t z, int scaled)
{
    return scaled ? arb_hypgeom_bessel_i_scaled_ref(nu, z) : arb_hypgeom_bessel_i_ref(nu, z);
}

di_t arb_hypgeom_bessel_k_integration_ref(di_t nu, di_t z, int scaled)
{
    return scaled ? arb_hypgeom_bessel_k_scaled_ref(nu, z) : arb_hypgeom_bessel_k_ref(nu, z);
}

acb_box_t acb_hypgeom_rising_ui_forward_ref(acb_box_t x, unsigned long long n)
{
    acb_box_t res = acb_box(di_interval(1.0, 1.0), di_interval(0.0, 0.0));
    unsigned long long k;

    for (k = 0; k < n; k++)
    {
        acb_box_t t = acb_box_add_ui(x, k);
        res = acb_box_mul(res, t);
    }

    return res;
}

acb_box_t acb_hypgeom_rising_ui_ref(acb_box_t x, unsigned long long n)
{
    return acb_hypgeom_rising_ui_forward_ref(x, n);
}

acb_box_t acb_hypgeom_lgamma_ref(acb_box_t x)
{
    double re_lo = x.real.a;
    double re_hi = x.real.b;
    double im_lo = x.imag.a;
    double im_hi = x.imag.b;
    double re_vals[5];
    double im_vals[5];
    hg_cplx_t z;
    hg_cplx_t w;
    int i;
    double re_min, re_max, im_min, im_max;
    acb_box_t full = acb_box(di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY));

    if (im_lo <= 0.0 && im_hi >= 0.0 && hg_interval_contains_nonpositive_integer(re_lo, re_hi))
        return full;

    z = hg_cplx(re_lo, im_lo); w = hg_log_gamma(z); re_vals[0] = w.re; im_vals[0] = w.im;
    z = hg_cplx(re_lo, im_hi); w = hg_log_gamma(z); re_vals[1] = w.re; im_vals[1] = w.im;
    z = hg_cplx(re_hi, im_lo); w = hg_log_gamma(z); re_vals[2] = w.re; im_vals[2] = w.im;
    z = hg_cplx(re_hi, im_hi); w = hg_log_gamma(z); re_vals[3] = w.re; im_vals[3] = w.im;
    z = hg_cplx(0.5 * (re_lo + re_hi), 0.5 * (im_lo + im_hi));
    w = hg_log_gamma(z); re_vals[4] = w.re; im_vals[4] = w.im;

    re_min = re_vals[0]; re_max = re_vals[0];
    im_min = im_vals[0]; im_max = im_vals[0];

    for (i = 1; i < 5; i++)
    {
        if (!isfinite(re_vals[i]) || !isfinite(im_vals[i]))
            return full;
        re_min = hg_min2(re_min, re_vals[i]);
        re_max = hg_max2(re_max, re_vals[i]);
        im_min = hg_min2(im_min, im_vals[i]);
        im_max = hg_max2(im_max, im_vals[i]);
    }

    return acb_box(
        di_interval(hg_below(re_min), hg_above(re_max)),
        di_interval(hg_below(im_min), hg_above(im_max)));
}

acb_box_t acb_hypgeom_gamma_ref(acb_box_t x)
{
    double re_lo = x.real.a;
    double re_hi = x.real.b;
    double im_lo = x.imag.a;
    double im_hi = x.imag.b;
    double re_vals[5];
    double im_vals[5];
    hg_cplx_t z;
    hg_cplx_t w;
    int i;
    double re_min, re_max, im_min, im_max;
    acb_box_t full = acb_box(di_interval(-INFINITY, INFINITY), di_interval(-INFINITY, INFINITY));

    if (im_lo <= 0.0 && im_hi >= 0.0 && hg_interval_contains_nonpositive_integer(re_lo, re_hi))
        return full;

    z = hg_cplx(re_lo, im_lo); w = hg_exp(hg_log_gamma(z)); re_vals[0] = w.re; im_vals[0] = w.im;
    z = hg_cplx(re_lo, im_hi); w = hg_exp(hg_log_gamma(z)); re_vals[1] = w.re; im_vals[1] = w.im;
    z = hg_cplx(re_hi, im_lo); w = hg_exp(hg_log_gamma(z)); re_vals[2] = w.re; im_vals[2] = w.im;
    z = hg_cplx(re_hi, im_hi); w = hg_exp(hg_log_gamma(z)); re_vals[3] = w.re; im_vals[3] = w.im;
    z = hg_cplx(0.5 * (re_lo + re_hi), 0.5 * (im_lo + im_hi));
    w = hg_exp(hg_log_gamma(z)); re_vals[4] = w.re; im_vals[4] = w.im;

    re_min = re_vals[0]; re_max = re_vals[0];
    im_min = im_vals[0]; im_max = im_vals[0];

    for (i = 1; i < 5; i++)
    {
        if (!isfinite(re_vals[i]) || !isfinite(im_vals[i]))
            return full;
        re_min = hg_min2(re_min, re_vals[i]);
        re_max = hg_max2(re_max, re_vals[i]);
        im_min = hg_min2(im_min, im_vals[i]);
        im_max = hg_max2(im_max, im_vals[i]);
    }

    return acb_box(
        di_interval(hg_below(re_min), hg_above(re_max)),
        di_interval(hg_below(im_min), hg_above(im_max)));
}

acb_box_t acb_hypgeom_rgamma_ref(acb_box_t x)
{
    acb_box_t g = acb_hypgeom_gamma_ref(x);
    di_t den = di_fast_add(di_fast_mul(g.real, g.real), di_fast_mul(g.imag, g.imag));
    acb_box_t num = acb_box(g.real, di_neg(g.imag));
    acb_box_t out;

    out.real = di_fast_div(num.real, den);
    out.imag = di_fast_div(num.imag, den);
    return out;
}

acb_box_t acb_hypgeom_erf_ref(acb_box_t x)
{
    double re_lo = x.real.a;
    double re_hi = x.real.b;
    double im_lo = x.imag.a;
    double im_hi = x.imag.b;
    double re_vals[5];
    double im_vals[5];
    hg_cplx_t z;
    hg_cplx_t w;
    int i;
    double re_min, re_max, im_min, im_max;

    z = hg_cplx(re_lo, im_lo); w = hg_erf_series(z); re_vals[0] = w.re; im_vals[0] = w.im;
    z = hg_cplx(re_lo, im_hi); w = hg_erf_series(z); re_vals[1] = w.re; im_vals[1] = w.im;
    z = hg_cplx(re_hi, im_lo); w = hg_erf_series(z); re_vals[2] = w.re; im_vals[2] = w.im;
    z = hg_cplx(re_hi, im_hi); w = hg_erf_series(z); re_vals[3] = w.re; im_vals[3] = w.im;
    z = hg_cplx(0.5 * (re_lo + re_hi), 0.5 * (im_lo + im_hi));
    w = hg_erf_series(z); re_vals[4] = w.re; im_vals[4] = w.im;

    re_min = re_vals[0]; re_max = re_vals[0];
    im_min = im_vals[0]; im_max = im_vals[0];
    for (i = 1; i < 5; i++)
    {
        re_min = hg_min2(re_min, re_vals[i]);
        re_max = hg_max2(re_max, re_vals[i]);
        im_min = hg_min2(im_min, im_vals[i]);
        im_max = hg_max2(im_max, im_vals[i]);
    }

    return acb_box(
        di_interval(hg_below(re_min), hg_above(re_max)),
        di_interval(hg_below(im_min), hg_above(im_max)));
}

acb_box_t acb_hypgeom_erfc_ref(acb_box_t x)
{
    double re_lo = x.real.a;
    double re_hi = x.real.b;
    double im_lo = x.imag.a;
    double im_hi = x.imag.b;
    double re_vals[5];
    double im_vals[5];
    hg_cplx_t z;
    hg_cplx_t w;
    int i;
    double re_min, re_max, im_min, im_max;

    z = hg_cplx(re_lo, im_lo); w = hg_erfc_point(z); re_vals[0] = w.re; im_vals[0] = w.im;
    z = hg_cplx(re_lo, im_hi); w = hg_erfc_point(z); re_vals[1] = w.re; im_vals[1] = w.im;
    z = hg_cplx(re_hi, im_lo); w = hg_erfc_point(z); re_vals[2] = w.re; im_vals[2] = w.im;
    z = hg_cplx(re_hi, im_hi); w = hg_erfc_point(z); re_vals[3] = w.re; im_vals[3] = w.im;
    z = hg_cplx(0.5 * (re_lo + re_hi), 0.5 * (im_lo + im_hi));
    w = hg_erfc_point(z); re_vals[4] = w.re; im_vals[4] = w.im;

    re_min = re_vals[0]; re_max = re_vals[0];
    im_min = im_vals[0]; im_max = im_vals[0];
    for (i = 1; i < 5; i++)
    {
        re_min = hg_min2(re_min, re_vals[i]);
        re_max = hg_max2(re_max, re_vals[i]);
        im_min = hg_min2(im_min, im_vals[i]);
        im_max = hg_max2(im_max, im_vals[i]);
    }

    return acb_box(
        di_interval(hg_below(re_min), hg_above(re_max)),
        di_interval(hg_below(im_min), hg_above(im_max)));
}

acb_box_t acb_hypgeom_erfi_ref(acb_box_t x)
{
    double re_lo = x.real.a;
    double re_hi = x.real.b;
    double im_lo = x.imag.a;
    double im_hi = x.imag.b;
    double re_vals[5];
    double im_vals[5];
    hg_cplx_t z;
    hg_cplx_t w;
    int i;
    double re_min, re_max, im_min, im_max;

    z = hg_cplx(re_lo, im_lo); w = hg_erfi_point(z); re_vals[0] = w.re; im_vals[0] = w.im;
    z = hg_cplx(re_lo, im_hi); w = hg_erfi_point(z); re_vals[1] = w.re; im_vals[1] = w.im;
    z = hg_cplx(re_hi, im_lo); w = hg_erfi_point(z); re_vals[2] = w.re; im_vals[2] = w.im;
    z = hg_cplx(re_hi, im_hi); w = hg_erfi_point(z); re_vals[3] = w.re; im_vals[3] = w.im;
    z = hg_cplx(0.5 * (re_lo + re_hi), 0.5 * (im_lo + im_hi));
    w = hg_erfi_point(z); re_vals[4] = w.re; im_vals[4] = w.im;

    re_min = re_vals[0]; re_max = re_vals[0];
    im_min = im_vals[0]; im_max = im_vals[0];
    for (i = 1; i < 5; i++)
    {
        re_min = hg_min2(re_min, re_vals[i]);
        re_max = hg_max2(re_max, re_vals[i]);
        im_min = hg_min2(im_min, im_vals[i]);
        im_max = hg_max2(im_max, im_vals[i]);
    }

    return acb_box(
        di_interval(hg_below(re_min), hg_above(re_max)),
        di_interval(hg_below(im_min), hg_above(im_max)));
}

acb_box_t acb_hypgeom_0f1_ref(acb_box_t a, acb_box_t z, int regularized)
{
    acb_box_t term = acb_box(di_interval(1.0, 1.0), di_interval(0.0, 0.0));
    acb_box_t sum = term;
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        acb_box_t ak = acb_box_add_ui(a, (unsigned long long) k);
        di_t inv_k1 = di_interval(1.0 / (double) (k + 1), 1.0 / (double) (k + 1));
        acb_box_t step = acb_box_div(z, ak);
        step = acb_box_scale_real(step, inv_k1);
        term = acb_box_mul(term, step);
        sum = acb_box_add(sum, term);
    }

    if (regularized)
        sum = acb_box_mul(sum, acb_hypgeom_rgamma_ref(a));

    return sum;
}

acb_box_t acb_hypgeom_1f1_ref(acb_box_t a, acb_box_t b, acb_box_t z)
{
    acb_box_t term = acb_box(di_interval(1.0, 1.0), di_interval(0.0, 0.0));
    acb_box_t sum = term;
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        acb_box_t ak = acb_box_add_ui(a, (unsigned long long) k);
        acb_box_t bk = acb_box_add_ui(b, (unsigned long long) k);
        acb_box_t step = acb_box_div(ak, bk);
        di_t inv_k1 = di_interval(1.0 / (double) (k + 1), 1.0 / (double) (k + 1));

        term = acb_box_mul(term, step);
        term = acb_box_mul(term, z);
        term = acb_box_scale_real(term, inv_k1);
        sum = acb_box_add(sum, term);
    }

    return sum;
}

acb_box_t acb_hypgeom_1f1_full_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized)
{
    acb_box_t res = acb_hypgeom_1f1_ref(a, b, z);
    if (regularized)
        res = acb_box_mul(res, acb_hypgeom_rgamma_ref(b));
    return res;
}

acb_box_t acb_hypgeom_1f1_integration_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized)
{
    return acb_hypgeom_1f1_full_ref(a, b, z, regularized);
}

acb_box_t acb_hypgeom_m_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized)
{
    return acb_hypgeom_1f1_full_ref(a, b, z, regularized);
}

acb_box_t acb_hypgeom_2f1_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z)
{
    acb_box_t term = acb_box(di_interval(1.0, 1.0), di_interval(0.0, 0.0));
    acb_box_t sum = term;
    int k;

    for (k = 0; k < HG_HYP_TERMS - 1; k++)
    {
        acb_box_t ak = acb_box_add_ui(a, (unsigned long long) k);
        acb_box_t bk = acb_box_add_ui(b, (unsigned long long) k);
        acb_box_t ck = acb_box_add_ui(c, (unsigned long long) k);
        di_t k1 = di_interval((double) (k + 1), (double) (k + 1));
        acb_box_t num = acb_box_mul(ak, bk);
        acb_box_t den = acb_box_scale_real(ck, k1);
        acb_box_t step = acb_box_div(num, den);

        term = acb_box_mul(term, step);
        term = acb_box_mul(term, z);
        sum = acb_box_add(sum, term);
    }

    return sum;
}

acb_box_t acb_hypgeom_2f1_full_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z, int regularized)
{
    acb_box_t res = acb_hypgeom_2f1_ref(a, b, c, z);
    if (regularized)
        res = acb_box_mul(res, acb_hypgeom_rgamma_ref(c));
    return res;
}

acb_box_t acb_hypgeom_2f1_integration_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z, int regularized)
{
    return acb_hypgeom_2f1_full_ref(a, b, c, z, regularized);
}

acb_box_t acb_hypgeom_u_ref(acb_box_t a, acb_box_t b, acb_box_t z)
{
    return acb_from_complex(hg_hypu_cplx(acb_mid(a), acb_mid(b), acb_mid(z)));
}

acb_box_t acb_hypgeom_u_integration_ref(acb_box_t a, acb_box_t b, acb_box_t z)
{
    return acb_hypgeom_u_ref(a, b, z);
}

static acb_box_t acb_from_complex(hg_cplx_t z)
{
    return acb_box(di_interval(hg_below(z.re), hg_above(z.re)), di_interval(hg_below(z.im), hg_above(z.im)));
}

static hg_cplx_t acb_mid(acb_box_t x)
{
    return hg_cplx(di_midpoint(x.real), di_midpoint(x.imag));
}

static hg_cplx_t hg_bessel_j_point(hg_cplx_t nu, hg_cplx_t z)
{
    return hg_bessel_series(nu, z, -1);
}

static hg_cplx_t hg_bessel_i_point(hg_cplx_t nu, hg_cplx_t z)
{
    return hg_bessel_series(nu, z, 1);
}

static hg_cplx_t hg_bessel_y_point_cplx(hg_cplx_t nu, hg_cplx_t z)
{
    hg_cplx_t jnu = hg_bessel_series(nu, z, -1);
    hg_cplx_t jneg = hg_bessel_series(hg_scale(nu, -1.0), z, -1);
    hg_cplx_t pi_nu = hg_scale(nu, HG_PI);
    hg_cplx_t c = hg_cos(pi_nu);
    hg_cplx_t s = hg_sin(pi_nu);
    if (hypot(s.re, s.im) < HG_SIN_EPS)
        return hg_cplx(NAN, NAN);
    return hg_div(hg_sub(hg_mul(jnu, c), jneg), s);
}

static hg_cplx_t hg_bessel_k_point_cplx(hg_cplx_t nu, hg_cplx_t z)
{
    hg_cplx_t inu = hg_bessel_series(nu, z, 1);
    hg_cplx_t ineg = hg_bessel_series(hg_scale(nu, -1.0), z, 1);
    hg_cplx_t pi_nu = hg_scale(nu, HG_PI);
    hg_cplx_t s = hg_sin(pi_nu);
    hg_cplx_t num = hg_scale(hg_sub(ineg, inu), 0.5 * HG_PI);
    if (hypot(s.re, s.im) < HG_SIN_EPS)
        return hg_cplx(NAN, NAN);
    return hg_div(num, s);
}

acb_box_t acb_hypgeom_bessel_j_0f1_ref(acb_box_t nu, acb_box_t z)
{
    return acb_from_complex(hg_bessel_j_point(acb_mid(nu), acb_mid(z)));
}

acb_box_t acb_hypgeom_bessel_j_asymp_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_j_0f1_ref(nu, z);
}

acb_box_t acb_hypgeom_bessel_j_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_j_0f1_ref(nu, z);
}

acb_box_t acb_hypgeom_bessel_i_0f1_ref(acb_box_t nu, acb_box_t z, int scaled)
{
    hg_cplx_t v = hg_bessel_i_point(acb_mid(nu), acb_mid(z));
    if (scaled)
        v = hg_mul(v, hg_exp(hg_scale(acb_mid(z), -1.0)));
    return acb_from_complex(v);
}

acb_box_t acb_hypgeom_bessel_i_asymp_ref(acb_box_t nu, acb_box_t z, int scaled)
{
    return acb_hypgeom_bessel_i_0f1_ref(nu, z, scaled);
}

acb_box_t acb_hypgeom_bessel_i_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_i_0f1_ref(nu, z, 0);
}

acb_box_t acb_hypgeom_bessel_i_scaled_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_i_0f1_ref(nu, z, 1);
}

acb_box_t acb_hypgeom_bessel_k_0f1_ref(acb_box_t nu, acb_box_t z, int scaled)
{
    hg_cplx_t v = hg_bessel_k_point_cplx(acb_mid(nu), acb_mid(z));
    if (scaled)
        v = hg_mul(v, hg_exp(acb_mid(z)));
    return acb_from_complex(v);
}

acb_box_t acb_hypgeom_bessel_k_asymp_ref(acb_box_t nu, acb_box_t z, int scaled)
{
    return acb_hypgeom_bessel_k_0f1_ref(nu, z, scaled);
}

acb_box_t acb_hypgeom_bessel_k_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_k_0f1_ref(nu, z, 0);
}

acb_box_t acb_hypgeom_bessel_k_scaled_ref(acb_box_t nu, acb_box_t z)
{
    return acb_hypgeom_bessel_k_0f1_ref(nu, z, 1);
}

acb_box_t acb_hypgeom_bessel_y_ref(acb_box_t nu, acb_box_t z)
{
    return acb_from_complex(hg_bessel_y_point_cplx(acb_mid(nu), acb_mid(z)));
}

void acb_hypgeom_bessel_jy_ref(acb_box_t *res1, acb_box_t *res2, acb_box_t nu, acb_box_t z)
{
    *res1 = acb_hypgeom_bessel_j_ref(nu, z);
    *res2 = acb_hypgeom_bessel_y_ref(nu, z);
}

void arb_hypgeom_rising_ui_batch_ref(const di_t *x, di_t *out, size_t count, unsigned long long n)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_rising_ui_ref(x[i], n);
}

void acb_hypgeom_rising_ui_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count, unsigned long long n)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_rising_ui_ref(x[i], n);
}

void arb_hypgeom_lgamma_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_lgamma_ref(x[i]);
}

void acb_hypgeom_lgamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_lgamma_ref(x[i]);
}

void arb_hypgeom_gamma_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_gamma_ref(x[i]);
}

void acb_hypgeom_gamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_gamma_ref(x[i]);
}

void arb_hypgeom_rgamma_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_rgamma_ref(x[i]);
}

void acb_hypgeom_rgamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_rgamma_ref(x[i]);
}

void arb_hypgeom_erf_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_erf_ref(x[i]);
}

void acb_hypgeom_erf_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_erf_ref(x[i]);
}

void arb_hypgeom_erfc_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_erfc_ref(x[i]);
}

void acb_hypgeom_erfc_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_erfc_ref(x[i]);
}

void arb_hypgeom_erfi_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_erfi_ref(x[i]);
}

void acb_hypgeom_erfi_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_erfi_ref(x[i]);
}

void arb_hypgeom_0f1_batch_ref(const di_t *a, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_0f1_ref(a[i], z[i], regularized);
}

void acb_hypgeom_0f1_batch_ref(const acb_box_t *a, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_0f1_ref(a[i], z[i], regularized);
}

void arb_hypgeom_m_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_m_ref(a[i], b[i], z[i], regularized);
}

void acb_hypgeom_m_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_m_ref(a[i], b[i], z[i], regularized);
}

void arb_hypgeom_1f1_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_1f1_ref(a[i], b[i], z[i]);
}

void acb_hypgeom_1f1_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_1f1_ref(a[i], b[i], z[i]);
}

void arb_hypgeom_1f1_full_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_1f1_full_ref(a[i], b[i], z[i], regularized);
}

void acb_hypgeom_1f1_full_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_1f1_full_ref(a[i], b[i], z[i], regularized);
}

void arb_hypgeom_1f1_integration_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_1f1_integration_ref(a[i], b[i], z[i], regularized);
}

void acb_hypgeom_1f1_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_1f1_integration_ref(a[i], b[i], z[i], regularized);
}

void arb_hypgeom_2f1_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_2f1_ref(a[i], b[i], c[i], z[i]);
}

void acb_hypgeom_2f1_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_2f1_ref(a[i], b[i], c[i], z[i]);
}

void arb_hypgeom_2f1_full_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_2f1_full_ref(a[i], b[i], c[i], z[i], regularized);
}

void acb_hypgeom_2f1_full_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_2f1_full_ref(a[i], b[i], c[i], z[i], regularized);
}

void arb_hypgeom_2f1_integration_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_2f1_integration_ref(a[i], b[i], c[i], z[i], regularized);
}

void acb_hypgeom_2f1_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count, int regularized)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_2f1_integration_ref(a[i], b[i], c[i], z[i], regularized);
}

void arb_hypgeom_u_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_u_ref(a[i], b[i], z[i]);
}

void acb_hypgeom_u_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_u_ref(a[i], b[i], z[i]);
}

void arb_hypgeom_u_integration_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_u_integration_ref(a[i], b[i], z[i]);
}

void acb_hypgeom_u_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_u_integration_ref(a[i], b[i], z[i]);
}

void arb_hypgeom_bessel_j_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_j_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_y_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_y_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_jy_batch_ref(di_t *out_j, di_t *out_y, const di_t *nu, const di_t *z, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        arb_hypgeom_bessel_jy_ref(&out_j[i], &out_y[i], nu[i], z[i]);
}

void arb_hypgeom_bessel_i_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_i_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_k_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_k_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_i_scaled_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_i_scaled_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_k_scaled_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_k_scaled_ref(nu[i], z[i]);
}

void arb_hypgeom_bessel_i_integration_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count, int scaled)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_i_integration_ref(nu[i], z[i], scaled);
}

void arb_hypgeom_bessel_k_integration_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count, int scaled)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_bessel_k_integration_ref(nu[i], z[i], scaled);
}

void arb_hypgeom_erfinv_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_erfinv_ref(x[i]);
}

void arb_hypgeom_erfcinv_batch_ref(const di_t *x, di_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = arb_hypgeom_erfcinv_ref(x[i]);
}

void acb_hypgeom_bessel_j_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_j_ref(nu[i], z[i]);
}

void acb_hypgeom_bessel_y_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_y_ref(nu[i], z[i]);
}

void acb_hypgeom_bessel_jy_batch_ref(acb_box_t *out_j, acb_box_t *out_y, const acb_box_t *nu, const acb_box_t *z, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        acb_hypgeom_bessel_jy_ref(&out_j[i], &out_y[i], nu[i], z[i]);
}

void acb_hypgeom_bessel_i_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_i_ref(nu[i], z[i]);
}

void acb_hypgeom_bessel_k_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_k_ref(nu[i], z[i]);
}

void acb_hypgeom_bessel_i_scaled_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_i_scaled_ref(nu[i], z[i]);
}

void acb_hypgeom_bessel_k_scaled_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = acb_hypgeom_bessel_k_scaled_ref(nu[i], z[i]);
}
