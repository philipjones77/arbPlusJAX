#ifndef BOOL_MAT_REF_H
#define BOOL_MAT_REF_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(BOOL_MAT_REF_BUILD_DLL)
#define BOOL_MAT_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define BOOL_MAT_REF_API __declspec(dllimport)
#else
#define BOOL_MAT_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

BOOL_MAT_REF_API uint8_t bool_mat_2x2_det_ref(const uint8_t *a);
BOOL_MAT_REF_API uint8_t bool_mat_2x2_trace_ref(const uint8_t *a);

BOOL_MAT_REF_API void bool_mat_2x2_det_batch_ref(const uint8_t *a, uint8_t *out, size_t count);
BOOL_MAT_REF_API void bool_mat_2x2_trace_batch_ref(const uint8_t *a, uint8_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
