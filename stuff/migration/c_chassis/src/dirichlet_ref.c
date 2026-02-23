#include "dirichlet_ref.h"

#include <math.h>

#define DIRICHLET_ULP_FACTOR 4.440892098500626e-16
#define DIRICHLET_HUGE 1e300
#define DIRICHLET_TINY 1e-300

static double dirichlet_below(double x)
{
    double t;

    if (x <= DIRICHLET_HUGE)
    {
        t = fabs(x) + DIRICHLET_TINY;
        return x - t * DIRICHLET_ULP_FACTOR;
    }

    if (isnan(x))
        return -INFINITY;

    return DIRICHLET_HUGE;
}

static double dirichlet_above(double x)
{
    double t;

    if (x >= -DIRICHLET_HUGE)
    {
        t = fabs(x) + DIRICHLET_TINY;
        return x + t * DIRICHLET_ULP_FACTOR;
    }

    if (isnan(x))
        return INFINITY;

    return -DIRICHLET_HUGE;
}

static double dirichlet_pow_real(double x, double s)
{
    return exp(-s * log(x));
}

static double dirichlet_zeta_series(double s, int n_terms)
{
    double sum = 0.0;
    int k;

    if (n_terms <= 0)
        n_terms = 1;

    for (k = 1; k <= n_terms; k++)
        sum += dirichlet_pow_real((double) k, s);

    return sum;
}

static di_t dirichlet_full(void)
{
    return di_interval(-INFINITY, INFINITY);
}

di_t dirichlet_zeta_ref(di_t s, int n_terms)
{
    double sm = di_midpoint(s);
    double v = dirichlet_zeta_series(sm, n_terms);
    if (!isfinite(v))
        return dirichlet_full();
    return di_interval(dirichlet_below(v), dirichlet_above(v));
}

di_t dirichlet_eta_ref(di_t s, int n_terms)
{
    double sm = di_midpoint(s);
    double zeta = dirichlet_zeta_series(sm, n_terms);
    double pow2 = exp((1.0 - sm) * log(2.0));
    double v = (1.0 - pow2) * zeta;
    if (!isfinite(v))
        return dirichlet_full();
    return di_interval(dirichlet_below(v), dirichlet_above(v));
}

void dirichlet_zeta_batch_ref(const di_t *s, di_t *out, size_t count, int n_terms)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = dirichlet_zeta_ref(s[i], n_terms);
}

void dirichlet_eta_batch_ref(const di_t *s, di_t *out, size_t count, int n_terms)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = dirichlet_eta_ref(s[i], n_terms);
}
