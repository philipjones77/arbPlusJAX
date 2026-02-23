#ifndef DFT_REF_H
#define DFT_REF_H

#include <stddef.h>
#include "double_interval_ref.h"

#if defined(_WIN32) && defined(DFT_REF_BUILD_DLL)
#define DFT_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define DFT_REF_API __declspec(dllimport)
#else
#define DFT_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    double re;
    double im;
} cplx_t;

typedef struct
{
    di_t real;
    di_t imag;
} dft_acb_box_t;

DFT_REF_API cplx_t cplx_make(double re, double im);
DFT_REF_API dft_acb_box_t dft_acb_box(di_t real, di_t imag);

/* Main DFT functions */
DFT_REF_API void cplx_dft_naive_ref(const cplx_t *x, cplx_t *out, size_t n);
DFT_REF_API void cplx_idft_naive_ref(const cplx_t *x, cplx_t *out, size_t n);
DFT_REF_API void cplx_dft_ref(const cplx_t *x, cplx_t *out, size_t n);
DFT_REF_API void cplx_idft_ref(const cplx_t *x, cplx_t *out, size_t n);

/* DFT on products */
DFT_REF_API void cplx_dft_prod_ref(const cplx_t *x, cplx_t *out, const size_t *cyc, size_t num);

/* Convolution */
DFT_REF_API void cplx_convol_circular_naive_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n);
DFT_REF_API void cplx_convol_circular_dft_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n);
DFT_REF_API void cplx_convol_circular_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n);

/* FFT algorithms */
DFT_REF_API void cplx_dft_rad2_ref(const cplx_t *x, cplx_t *out, size_t n);
DFT_REF_API void cplx_idft_rad2_ref(const cplx_t *x, cplx_t *out, size_t n);
DFT_REF_API void cplx_convol_circular_rad2_ref(const cplx_t *f, const cplx_t *g, cplx_t *out, size_t n);

/* Interval-box acb-style variants */
DFT_REF_API void acb_dft_naive_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_idft_naive_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_dft_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_idft_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_dft_prod_ref(const dft_acb_box_t *x, dft_acb_box_t *out, const size_t *cyc, size_t num);

DFT_REF_API void acb_convol_circular_naive_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_convol_circular_dft_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_convol_circular_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n);

DFT_REF_API void acb_dft_rad2_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_idft_rad2_ref(const dft_acb_box_t *x, dft_acb_box_t *out, size_t n);
DFT_REF_API void acb_convol_circular_rad2_ref(const dft_acb_box_t *f, const dft_acb_box_t *g, dft_acb_box_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif
