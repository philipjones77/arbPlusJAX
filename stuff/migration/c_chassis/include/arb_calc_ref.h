#ifndef ARB_CALC_REF_H
#define ARB_CALC_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(ARB_CALC_REF_BUILD_DLL)
#define ARB_CALC_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARB_CALC_REF_API __declspec(dllimport)
#else
#define ARB_CALC_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ARB_CALC_REF_INTEGRAND_EXP = 0,
    ARB_CALC_REF_INTEGRAND_SIN = 1,
    ARB_CALC_REF_INTEGRAND_COS = 2
} arb_calc_ref_integrand_t;

ARB_CALC_REF_API di_t arb_calc_integrate_line_ref(di_t a, di_t b, int integrand_id, int n_steps);
ARB_CALC_REF_API void arb_calc_integrate_line_batch_ref(
    const di_t *a, const di_t *b, di_t *out, size_t count, int integrand_id, int n_steps);

#ifdef __cplusplus
}
#endif

#endif
