#ifndef PARTITIONS_REF_H
#define PARTITIONS_REF_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PARTITIONS_REF_BUILD_DLL)
#define PARTITIONS_REF_API __declspec(dllexport)
#elif defined(_WIN32)
#define PARTITIONS_REF_API __declspec(dllimport)
#else
#define PARTITIONS_REF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

PARTITIONS_REF_API uint64_t partitions_p_ref(int n);
PARTITIONS_REF_API void partitions_p_batch_ref(const int *n, uint64_t *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif
