#ifndef ACB_CALC_REF_H
#define ACB_CALC_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_CALC_REF_BUILD_DLL)
#define ACB_CALC_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_CALC_REF_API __declspec(dllimport)
#else
#define ACB_CALC_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ACB_CALC_REF_INTEGRAND_EXP = 0,
    ACB_CALC_REF_INTEGRAND_SIN = 1,
    ACB_CALC_REF_INTEGRAND_COS = 2
} acb_calc_ref_integrand_t;

ACB_CALC_REF_API acb_box_t acb_calc_integrate_line_ref(
    acb_box_t a, acb_box_t b, int integrand_id, int n_steps);

ACB_CALC_REF_API void acb_calc_integrate_line_batch_ref(
    const acb_box_t *a, const acb_box_t *b, acb_box_t *out,
    size_t count, int integrand_id, int n_steps);

#ifdef __cplusplus
}
#endif

#endif
