#include "barrett_rvv.h"
#include <stdint.h>

/* Scalar Barrett: same algorithm as RVV version, no vector instructions */
void barrett_mul_rvv_u64m1(
        const uint64_t* a,
        const uint64_t* b,
        uint64_t* out,
        size_t n,
        uint64_t q, uint64_t mu,
        int n_shift, int n7, int n64, int n764)
{
    for (size_t i = 0; i < n; i++) {
        __uint128_t ab   = (__uint128_t)a[i] * b[i];
        uint64_t lo_ab   = (uint64_t)ab;
        uint64_t hi_ab   = (uint64_t)(ab >> 64);
        uint64_t x1      = (lo_ab >> n_shift) | (hi_ab << n64);
        __uint128_t x1mu = (__uint128_t)x1 * mu;
        uint64_t lo2     = (uint64_t)x1mu;
        uint64_t hi2     = (uint64_t)(x1mu >> 64);
        uint64_t q_est   = (lo2 >> n7) | (hi2 << n764);
        uint64_t res     = lo_ab - q_est * q;
        if (res >= q) res -= q;
        if (res >= q) res -= q;
        out[i] = res;
    }
}
