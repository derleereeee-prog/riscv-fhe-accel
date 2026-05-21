#ifndef AUTOMORPH_H
#define AUTOMORPH_H

#include <stdint.h>

// Automorphism: dst[(idx * k) % N] = src[idx]
// Preconditions: k is odd, k < 2N, src != dst, len(src)==len(dst)==N

void automorph_scalar   (const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k);
void automorph_naive_rvv(const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k);
void automorph_rvv      (const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k);

// Cycle counter (RISC-V CSR)
static inline uint64_t rdcycle(void) {
    uint64_t v;
    __asm__ volatile ("rdcycle %0" : "=r"(v));
    return v;
}

#endif
