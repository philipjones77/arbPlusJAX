#include "partitions_ref.h"

#include <stdlib.h>

uint64_t partitions_p_ref(int n)
{
    if (n < 0)
        return 0;

    uint64_t *p = (uint64_t *) calloc((size_t) (n + 1), sizeof(uint64_t));
    if (!p)
        return 0;

    p[0] = 1;

    for (int k = 1; k <= n; k++)
    {
        int64_t sum = 0;
        for (int m = 1; m <= k; m++)
        {
            int g1 = (m * (3 * m - 1)) / 2;
            int g2 = (m * (3 * m + 1)) / 2;
            if (g1 > k && g2 > k)
                break;
            int sign = (m & 1) ? 1 : -1;
            if (g1 <= k)
                sum += sign * (int64_t) p[k - g1];
            if (g2 <= k)
                sum += sign * (int64_t) p[k - g2];
        }
        if (sum < 0)
            sum = 0;
        p[k] = (uint64_t) sum;
    }

    uint64_t out = p[n];
    free(p);
    return out;
}

void partitions_p_batch_ref(const int *n, uint64_t *out, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        out[i] = partitions_p_ref(n[i]);
}
