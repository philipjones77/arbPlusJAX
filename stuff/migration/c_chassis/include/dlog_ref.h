#ifndef DLOG_REF_H
#define DLOG_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(DLOG_REF_BUILD_DLL)
#define DLOG_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define DLOG_REF_API __declspec(dllimport)
#else
#define DLOG_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

DLOG_REF_API double dlog_log1p_ref(double x);
DLOG_REF_API void dlog_log1p_batch_ref(const double *x, double *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
