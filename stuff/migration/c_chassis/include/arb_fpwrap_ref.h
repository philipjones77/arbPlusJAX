#ifndef ARB_FPWRAP_REF_H
#define ARB_FPWRAP_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(ARB_FPWRAP_REF_BUILD_DLL)
#define ARB_FPWRAP_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARB_FPWRAP_REF_API __declspec(dllimport)
#else
#define ARB_FPWRAP_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double real;
    double imag;
} arb_fpwrap_cdouble_t;

ARB_FPWRAP_REF_API int arb_fpwrap_double_exp_ref(double *res, double x);
ARB_FPWRAP_REF_API int arb_fpwrap_double_log_ref(double *res, double x);
ARB_FPWRAP_REF_API int arb_fpwrap_cdouble_exp_ref(arb_fpwrap_cdouble_t *res, arb_fpwrap_cdouble_t x);
ARB_FPWRAP_REF_API int arb_fpwrap_cdouble_log_ref(arb_fpwrap_cdouble_t *res, arb_fpwrap_cdouble_t x);

#ifdef __cplusplus
}
#endif

#endif
