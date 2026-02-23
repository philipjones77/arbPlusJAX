#include "bool_mat_ref.h"

uint8_t bool_mat_2x2_det_ref(const uint8_t *a)
{
    uint8_t a00 = a[0] & 1u;
    uint8_t a01 = a[1] & 1u;
    uint8_t a10 = a[2] & 1u;
    uint8_t a11 = a[3] & 1u;
    return (a00 & a11) ^ (a01 & a10);
}

uint8_t bool_mat_2x2_trace_ref(const uint8_t *a)
{
    uint8_t a00 = a[0] & 1u;
    uint8_t a11 = a[3] & 1u;
    return (a00 ^ a11) & 1u;
}

void bool_mat_2x2_det_batch_ref(const uint8_t *a, uint8_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = bool_mat_2x2_det_ref(a + 4 * i);
}

void bool_mat_2x2_trace_batch_ref(const uint8_t *a, uint8_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = bool_mat_2x2_trace_ref(a + 4 * i);
}
