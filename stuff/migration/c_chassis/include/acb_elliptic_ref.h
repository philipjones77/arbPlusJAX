#ifndef ACB_ELLIPTIC_REF_H
#define ACB_ELLIPTIC_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_ELLIPTIC_REF_BUILD_DLL)
#define ACB_ELLIPTIC_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_ELLIPTIC_REF_API __declspec(dllimport)
#else
#define ACB_ELLIPTIC_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACB_ELLIPTIC_REF_API acb_box_t acb_elliptic_k_ref(acb_box_t m);
ACB_ELLIPTIC_REF_API acb_box_t acb_elliptic_e_ref(acb_box_t m);

ACB_ELLIPTIC_REF_API void acb_elliptic_k_batch_ref(const acb_box_t *m, acb_box_t *out, size_t count);
ACB_ELLIPTIC_REF_API void acb_elliptic_e_batch_ref(const acb_box_t *m, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
