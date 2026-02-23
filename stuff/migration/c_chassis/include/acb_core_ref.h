#ifndef ACB_CORE_REF_H
#define ACB_CORE_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(ACB_CORE_REF_BUILD_DLL)
#define ACB_CORE_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_CORE_REF_API __declspec(dllimport)
#else
#define ACB_CORE_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    di_t real;
    di_t imag;
} acb_box_t;

ACB_CORE_REF_API acb_box_t acb_exp_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_log_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_sqrt_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_sin_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_cos_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_tan_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_sinh_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_cosh_ref(acb_box_t x);
ACB_CORE_REF_API acb_box_t acb_tanh_ref(acb_box_t x);

ACB_CORE_REF_API void acb_exp_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_log_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_sqrt_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_sin_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_cos_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_tan_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_sinh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_cosh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
ACB_CORE_REF_API void acb_tanh_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
