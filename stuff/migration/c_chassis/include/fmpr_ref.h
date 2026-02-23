#ifndef FMPR_REF_H
#define FMPR_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(FMPR_REF_BUILD_DLL)
#define FMPR_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define FMPR_REF_API __declspec(dllimport)
#else
#define FMPR_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double v;
} fmpr_ref_t;

FMPR_REF_API fmpr_ref_t fmpr_add_ref(fmpr_ref_t a, fmpr_ref_t b);
FMPR_REF_API fmpr_ref_t fmpr_mul_ref(fmpr_ref_t a, fmpr_ref_t b);

FMPR_REF_API void fmpr_add_batch_ref(const fmpr_ref_t *a, const fmpr_ref_t *b, fmpr_ref_t *out, size_t count);
FMPR_REF_API void fmpr_mul_batch_ref(const fmpr_ref_t *a, const fmpr_ref_t *b, fmpr_ref_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
