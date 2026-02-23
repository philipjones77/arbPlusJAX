#ifndef ACB_MAT_REF_H
#define ACB_MAT_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_MAT_REF_BUILD_DLL)
#define ACB_MAT_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_MAT_REF_API __declspec(dllimport)
#else
#define ACB_MAT_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACB_MAT_REF_API acb_box_t acb_mat_2x2_det_ref(const acb_box_t *a);
ACB_MAT_REF_API acb_box_t acb_mat_2x2_trace_ref(const acb_box_t *a);

ACB_MAT_REF_API void acb_mat_2x2_det_batch_ref(const acb_box_t *a, acb_box_t *out, size_t count);
ACB_MAT_REF_API void acb_mat_2x2_trace_batch_ref(const acb_box_t *a, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
