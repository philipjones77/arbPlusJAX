#ifndef FMPZI_REF_H
#define FMPZI_REF_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(FMPZI_REF_BUILD_DLL)
#define FMPZI_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define FMPZI_REF_API __declspec(dllimport)
#else
#define FMPZI_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int64_t lo;
    int64_t hi;
} fmpzi_ref_t;

FMPZI_REF_API fmpzi_ref_t fmpzi_interval_ref(int64_t lo, int64_t hi);
FMPZI_REF_API fmpzi_ref_t fmpzi_add_ref(fmpzi_ref_t a, fmpzi_ref_t b);
FMPZI_REF_API fmpzi_ref_t fmpzi_sub_ref(fmpzi_ref_t a, fmpzi_ref_t b);

FMPZI_REF_API void fmpzi_add_batch_ref(const fmpzi_ref_t *a, const fmpzi_ref_t *b, fmpzi_ref_t *out, size_t count);
FMPZI_REF_API void fmpzi_sub_batch_ref(const fmpzi_ref_t *a, const fmpzi_ref_t *b, fmpzi_ref_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
