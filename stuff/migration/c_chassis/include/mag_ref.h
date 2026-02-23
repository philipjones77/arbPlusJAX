#ifndef MAG_REF_H
#define MAG_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(MAG_REF_BUILD_DLL)
#define MAG_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define MAG_REF_API __declspec(dllimport)
#else
#define MAG_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

MAG_REF_API double mag_add_ref(double a, double b);
MAG_REF_API double mag_mul_ref(double a, double b);

MAG_REF_API void mag_add_batch_ref(const double *a, const double *b, double *out, size_t count);
MAG_REF_API void mag_mul_batch_ref(const double *a, const double *b, double *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
