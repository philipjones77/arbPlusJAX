#ifndef ARB_CORE_REF_H
#define ARB_CORE_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(ARB_CORE_REF_BUILD_DLL)
#define ARB_CORE_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARB_CORE_REF_API __declspec(dllimport)
#else
#define ARB_CORE_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ARB_CORE_REF_API di_t arb_exp_ref(di_t x);
ARB_CORE_REF_API di_t arb_log_ref(di_t x);
ARB_CORE_REF_API di_t arb_sqrt_ref(di_t x);
ARB_CORE_REF_API di_t arb_sin_ref(di_t x);
ARB_CORE_REF_API di_t arb_cos_ref(di_t x);
ARB_CORE_REF_API di_t arb_tan_ref(di_t x);
ARB_CORE_REF_API di_t arb_sinh_ref(di_t x);
ARB_CORE_REF_API di_t arb_cosh_ref(di_t x);
ARB_CORE_REF_API di_t arb_tanh_ref(di_t x);

ARB_CORE_REF_API void arb_exp_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_log_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_sqrt_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_sin_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_cos_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_tan_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_sinh_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_cosh_batch_ref(const di_t *x, di_t *out, size_t count);
ARB_CORE_REF_API void arb_tanh_batch_ref(const di_t *x, di_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
