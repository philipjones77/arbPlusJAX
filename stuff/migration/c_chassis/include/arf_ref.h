#ifndef ARF_REF_H
#define ARF_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(ARF_REF_BUILD_DLL)
#define ARF_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ARF_REF_API __declspec(dllimport)
#else
#define ARF_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double v;
} arf_ref_t;

ARF_REF_API arf_ref_t arf_add_ref(arf_ref_t a, arf_ref_t b);
ARF_REF_API arf_ref_t arf_mul_ref(arf_ref_t a, arf_ref_t b);

ARF_REF_API void arf_add_batch_ref(const arf_ref_t *a, const arf_ref_t *b, arf_ref_t *out, size_t count);
ARF_REF_API void arf_mul_batch_ref(const arf_ref_t *a, const arf_ref_t *b, arf_ref_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
