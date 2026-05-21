#pragma once
/*
 * Barrett modular multiplication — external function declaration.
 *
 * Two implementations share this interface:
 *   barrett_rvv.o    — RVV 1.0 vector implementation (compile with -march=rv64gcv)
 *   barrett_scalar.o — plain scalar C fallback (for baseline benchmarks)
 *
 * Link whichever .o you need; OpenFHE itself is compiled only once.
 *
 * out[i] = (a[i] * b[i]) mod q,  i in [0, n)
 * Constraint: n_shift + 7 < 64  (i.e. GetMSB(q) <= 58)
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void barrett_mul_rvv_u64m1(
        const uint64_t* a,
        const uint64_t* b,
        uint64_t* out,
        size_t n,
        uint64_t q, uint64_t mu,
        int n_shift, int n7, int n64, int n764);

#ifdef __cplusplus
}
#endif
