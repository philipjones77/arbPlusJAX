#ifndef DIRICHLET_REF_H
#define DIRICHLET_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(DIRICHLET_REF_BUILD_DLL)
#define DIRICHLET_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define DIRICHLET_REF_API __declspec(dllimport)
#else
#define DIRICHLET_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

DIRICHLET_REF_API di_t dirichlet_zeta_ref(di_t s, int n_terms);
DIRICHLET_REF_API di_t dirichlet_eta_ref(di_t s, int n_terms);

DIRICHLET_REF_API void dirichlet_zeta_batch_ref(const di_t *s, di_t *out, size_t count, int n_terms);
DIRICHLET_REF_API void dirichlet_eta_batch_ref(const di_t *s, di_t *out, size_t count, int n_terms);

#ifdef __cplusplus
}
#endif

#endif
