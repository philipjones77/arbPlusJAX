#ifndef ACB_MODULAR_REF_H
#define ACB_MODULAR_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_MODULAR_REF_BUILD_DLL)
#define ACB_MODULAR_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_MODULAR_REF_API __declspec(dllimport)
#else
#define ACB_MODULAR_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACB_MODULAR_REF_API acb_box_t acb_modular_j_ref(acb_box_t tau);
ACB_MODULAR_REF_API void acb_modular_j_batch_ref(const acb_box_t *tau, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
