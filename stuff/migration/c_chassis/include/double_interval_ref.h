#ifndef DOUBLE_INTERVAL_REF_H
#define DOUBLE_INTERVAL_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(DI_REF_BUILD_DLL)
#define DI_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define DI_REF_API __declspec(dllimport)
#else
#define DI_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double a;
    double b;
} di_t;

DI_REF_API di_t di_interval(double a, double b);
DI_REF_API di_t di_neg(di_t x);
DI_REF_API di_t di_fast_add(di_t x, di_t y);
DI_REF_API di_t di_fast_sub(di_t x, di_t y);
DI_REF_API di_t di_fast_mul(di_t x, di_t y);
DI_REF_API di_t di_fast_div(di_t x, di_t y);
DI_REF_API di_t di_fast_sqr(di_t x);
DI_REF_API di_t di_fast_log_nonnegative(di_t x);
DI_REF_API double di_fast_ubound_radius(di_t x);
DI_REF_API double di_midpoint(di_t x);

DI_REF_API void di_fast_add_batch(const di_t *x, const di_t *y, di_t *out, size_t n);
DI_REF_API void di_fast_sub_batch(const di_t *x, const di_t *y, di_t *out, size_t n);
DI_REF_API void di_fast_mul_batch(const di_t *x, const di_t *y, di_t *out, size_t n);
DI_REF_API void di_fast_div_batch(const di_t *x, const di_t *y, di_t *out, size_t n);
DI_REF_API void di_fast_sqr_batch(const di_t *x, di_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif
