#ifndef ARB_POLY_REF_H
#define ARB_POLY_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(ARB_POLY_REF_BUILD_DLL)
#define ARB_POLY_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARB_POLY_REF_API __declspec(dllimport)
#else
#define ARB_POLY_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ARB_POLY_REF_API di_t arb_poly_eval_cubic_ref(const di_t *coeffs, di_t x);
ARB_POLY_REF_API void arb_poly_eval_cubic_batch_ref(
    const di_t *coeffs, const di_t *x, di_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
