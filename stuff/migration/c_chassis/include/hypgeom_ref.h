#ifndef HYPGEOM_REF_H
#define HYPGEOM_REF_H

#include <stddef.h>

#include "double_interval_ref.h"

#if defined(_WIN32) && defined(HYPGEOM_REF_BUILD_DLL)
#define HYPGEOM_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define HYPGEOM_REF_API __declspec(dllimport)
#else
#define HYPGEOM_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    di_t real;
    di_t imag;
} acb_box_t;

HYPGEOM_REF_API acb_box_t acb_box(di_t real, di_t imag);
HYPGEOM_REF_API acb_box_t acb_box_add_ui(acb_box_t x, unsigned long long k);
HYPGEOM_REF_API acb_box_t acb_box_mul(acb_box_t x, acb_box_t y);

HYPGEOM_REF_API di_t arb_hypgeom_rising_ui_forward_ref(di_t x, unsigned long long n);
HYPGEOM_REF_API di_t arb_hypgeom_rising_ui_ref(di_t x, unsigned long long n);
HYPGEOM_REF_API di_t arb_hypgeom_lgamma_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_gamma_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_rgamma_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_erf_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_erfc_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_erfi_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_0f1_ref(di_t a, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_m_ref(di_t a, di_t b, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_1f1_ref(di_t a, di_t b, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_1f1_full_ref(di_t a, di_t b, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_1f1_integration_ref(di_t a, di_t b, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_2f1_ref(di_t a, di_t b, di_t c, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_2f1_full_ref(di_t a, di_t b, di_t c, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_2f1_integration_ref(di_t a, di_t b, di_t c, di_t z, int regularized);
HYPGEOM_REF_API di_t arb_hypgeom_u_ref(di_t a, di_t b, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_u_integration_ref(di_t a, di_t b, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_j_ref(di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_y_ref(di_t nu, di_t z);
HYPGEOM_REF_API void arb_hypgeom_bessel_jy_ref(di_t *res1, di_t *res2, di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_i_ref(di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_k_ref(di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_i_scaled_ref(di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_k_scaled_ref(di_t nu, di_t z);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_i_integration_ref(di_t nu, di_t z, int scaled);
HYPGEOM_REF_API di_t arb_hypgeom_bessel_k_integration_ref(di_t nu, di_t z, int scaled);
HYPGEOM_REF_API di_t arb_hypgeom_erfinv_ref(di_t x);
HYPGEOM_REF_API di_t arb_hypgeom_erfcinv_ref(di_t x);

HYPGEOM_REF_API acb_box_t acb_hypgeom_rising_ui_forward_ref(acb_box_t x, unsigned long long n);
HYPGEOM_REF_API acb_box_t acb_hypgeom_rising_ui_ref(acb_box_t x, unsigned long long n);
HYPGEOM_REF_API acb_box_t acb_hypgeom_lgamma_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_gamma_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_rgamma_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_erf_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_erfc_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_erfi_ref(acb_box_t x);
HYPGEOM_REF_API acb_box_t acb_hypgeom_0f1_ref(acb_box_t a, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_m_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_1f1_ref(acb_box_t a, acb_box_t b, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_1f1_full_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_1f1_integration_ref(acb_box_t a, acb_box_t b, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_2f1_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_2f1_full_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_2f1_integration_ref(acb_box_t a, acb_box_t b, acb_box_t c, acb_box_t z, int regularized);
HYPGEOM_REF_API acb_box_t acb_hypgeom_u_ref(acb_box_t a, acb_box_t b, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_u_integration_ref(acb_box_t a, acb_box_t b, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_j_0f1_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_j_asymp_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_j_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_i_0f1_ref(acb_box_t nu, acb_box_t z, int scaled);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_i_asymp_ref(acb_box_t nu, acb_box_t z, int scaled);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_i_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_i_scaled_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_k_0f1_ref(acb_box_t nu, acb_box_t z, int scaled);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_k_asymp_ref(acb_box_t nu, acb_box_t z, int scaled);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_k_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_k_scaled_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API acb_box_t acb_hypgeom_bessel_y_ref(acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API void acb_hypgeom_bessel_jy_ref(acb_box_t *res1, acb_box_t *res2, acb_box_t nu, acb_box_t z);
HYPGEOM_REF_API void hypgeom_ref_set_bessel_real_mode(int mode);

HYPGEOM_REF_API void arb_hypgeom_rising_ui_batch_ref(const di_t *x, di_t *out, size_t count, unsigned long long n);
HYPGEOM_REF_API void acb_hypgeom_rising_ui_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count, unsigned long long n);
HYPGEOM_REF_API void arb_hypgeom_lgamma_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_lgamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_gamma_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_gamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_rgamma_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_rgamma_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_erf_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_erf_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_erfc_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_erfc_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_erfi_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_erfi_batch_ref(const acb_box_t *x, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_0f1_batch_ref(const di_t *a, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_0f1_batch_ref(const acb_box_t *a, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_m_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_m_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_1f1_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_1f1_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_1f1_full_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_1f1_full_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_1f1_integration_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_1f1_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_2f1_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_2f1_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_2f1_full_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_2f1_full_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_2f1_integration_batch_ref(const di_t *a, const di_t *b, const di_t *c, const di_t *z, di_t *out, size_t count, int regularized);
HYPGEOM_REF_API void acb_hypgeom_2f1_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *c, const acb_box_t *z, acb_box_t *out, size_t count, int regularized);
HYPGEOM_REF_API void arb_hypgeom_u_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_u_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_u_integration_batch_ref(const di_t *a, const di_t *b, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_u_integration_batch_ref(const acb_box_t *a, const acb_box_t *b, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_j_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_y_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_jy_batch_ref(di_t *out_j, di_t *out_y, const di_t *nu, const di_t *z, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_i_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_k_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_i_scaled_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_k_scaled_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_bessel_i_integration_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count, int scaled);
HYPGEOM_REF_API void arb_hypgeom_bessel_k_integration_batch_ref(const di_t *nu, const di_t *z, di_t *out, size_t count, int scaled);
HYPGEOM_REF_API void arb_hypgeom_erfinv_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void arb_hypgeom_erfcinv_batch_ref(const di_t *x, di_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_j_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_y_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_jy_batch_ref(acb_box_t *out_j, acb_box_t *out_y, const acb_box_t *nu, const acb_box_t *z, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_i_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_k_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_i_scaled_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);
HYPGEOM_REF_API void acb_hypgeom_bessel_k_scaled_batch_ref(const acb_box_t *nu, const acb_box_t *z, acb_box_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
