#ifndef ARB_MAT_REF_H
#define ARB_MAT_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(ARB_MAT_REF_BUILD_DLL)
#define ARB_MAT_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARB_MAT_REF_API __declspec(dllimport)
#else
#define ARB_MAT_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ARB_MAT_REF_API di_t arb_mat_2x2_det_ref(const di_t *a);
ARB_MAT_REF_API di_t arb_mat_2x2_trace_ref(const di_t *a);

ARB_MAT_REF_API void arb_mat_2x2_det_batch_ref(const di_t *a, di_t *out, size_t count);
ARB_MAT_REF_API void arb_mat_2x2_trace_batch_ref(const di_t *a, di_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
