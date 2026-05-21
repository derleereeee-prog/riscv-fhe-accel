#include "automorph.h"

// Reference scalar implementation: dst[(idx*k)%N] = src[idx]
// This is the correctness oracle for the RVV versions.
void automorph_scalar(const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k) {
    for (uint32_t idx = 0; idx < N; idx++) {
        uint64_t new_idx = ((uint64_t)idx * k) % N;
        dst[new_idx] = src[idx];
    }
}
