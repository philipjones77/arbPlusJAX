#ifndef ACB_DIRICHLET_REF_H
#define ACB_DIRICHLET_REF_H

#include <stddef.h>

#include "acb_core_ref.h"

#if defined(_WIN32) && defined(ACB_DIRICHLET_REF_BUILD_DLL)
#define ACB_DIRICHLET_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define ACB_DIRICHLET_REF_API __declspec(dllimport)
#else
#define ACB_DIRICHLET_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACB_DIRICHLET_REF_API acb_box_t acb_dirichlet_zeta_ref(acb_box_t s, int n_terms);
ACB_DIRICHLET_REF_API acb_box_t acb_dirichlet_eta_ref(acb_box_t s, int n_terms);

ACB_DIRICHLET_REF_API void acb_dirichlet_zeta_batch_ref(
    const acb_box_t *s, acb_box_t *out, size_t count, int n_terms);
ACB_DIRICHLET_REF_API void acb_dirichlet_eta_batch_ref(
    const acb_box_t *s, acb_box_t *out, size_t count, int n_terms);

#ifdef __cplusplus
}
#endif

#endif
