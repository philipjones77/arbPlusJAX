#include "bernoulli_ref.h"

double bernoulli_number_ref(int n)
{
    switch (n)
    {
        case 0:
            return 1.0;
        case 1:
            return -0.5;
        case 2:
            return 1.0 / 6.0;
        case 4:
            return -1.0 / 30.0;
        default:
            return 0.0;
    }
}

void bernoulli_number_batch_ref(const int *n, double *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = bernoulli_number_ref(n[i]);
}
