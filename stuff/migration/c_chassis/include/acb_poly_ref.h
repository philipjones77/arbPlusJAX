#ifndef ACB_POLY_REF_H
#define ACB_POLY_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_POLY_REF_BUILD_DLL)
#define ACB_POLY_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_POLY_REF_API __declspec(dllimport)
#else
#define ACB_POLY_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACB_POLY_REF_API acb_box_t acb_poly_eval_cubic_ref(const acb_box_t *coeffs, acb_box_t z);
ACB_POLY_REF_API void acb_poly_eval_cubic_batch_ref(
    const acb_box_t *coeffs, const acb_box_t *z, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
