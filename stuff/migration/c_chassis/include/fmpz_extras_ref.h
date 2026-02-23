#ifndef FMPZ_EXTRAS_REF_H
#define FMPZ_EXTRAS_REF_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(FMPZ_EXTRAS_REF_BUILD_DLL)
#define FMPZ_EXTRAS_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define FMPZ_EXTRAS_REF_API __declspec(dllimport)
#else
#define FMPZ_EXTRAS_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int64_t v;
} fmpz_extras_ref_t;

FMPZ_EXTRAS_REF_API fmpz_extras_ref_t fmpz_extras_add_ref(fmpz_extras_ref_t a, fmpz_extras_ref_t b);
FMPZ_EXTRAS_REF_API fmpz_extras_ref_t fmpz_extras_mul_ref(fmpz_extras_ref_t a, fmpz_extras_ref_t b);

FMPZ_EXTRAS_REF_API void fmpz_extras_add_batch_ref(
    const fmpz_extras_ref_t *a, const fmpz_extras_ref_t *b, fmpz_extras_ref_t *out, size_t count);
FMPZ_EXTRAS_REF_API void fmpz_extras_mul_batch_ref(
    const fmpz_extras_ref_t *a, const fmpz_extras_ref_t *b, fmpz_extras_ref_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
