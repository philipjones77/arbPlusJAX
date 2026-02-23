#ifndef ACF_REF_H
#define ACF_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(ACF_REF_BUILD_DLL)
#define ACF_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACF_REF_API __declspec(dllimport)
#else
#define ACF_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double re;
    double im;
} acf_ref_t;

ACF_REF_API acf_ref_t acf_add_ref(acf_ref_t a, acf_ref_t b);
ACF_REF_API acf_ref_t acf_mul_ref(acf_ref_t a, acf_ref_t b);

ACF_REF_API void acf_add_batch_ref(const acf_ref_t *a, const acf_ref_t *b, acf_ref_t *out, size_t count);
ACF_REF_API void acf_mul_batch_ref(const acf_ref_t *a, const acf_ref_t *b, acf_ref_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
