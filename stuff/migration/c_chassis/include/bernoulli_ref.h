#ifndef BERNOULLI_REF_H
#define BERNOULLI_REF_H

#include <stddef.h>

#if defined(_WIN32) && defined(BERNOULLI_REF_BUILD_DLL)
#define BERNOULLI_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define BERNOULLI_REF_API __declspec(dllimport)
#else
#define BERNOULLI_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

BERNOULLI_REF_API double bernoulli_number_ref(int n);
BERNOULLI_REF_API void bernoulli_number_batch_ref(const int *n, double *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
