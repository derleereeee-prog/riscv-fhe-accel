#include <riscv_vector.h>
#include "automorph.h"

// Compute k^{-1} mod N (N is a power of 2, k is odd => coprime with N)
static uint32_t mod_inverse(uint32_t k, uint32_t N) {
    // Brute force: acceptable for small N (N <= 2^16)
    for (uint32_t i = 1; i < N; i += 2) {
        if (((uint64_t)k * i) % N == 1)
            return i;
    }
    return 0; // unreachable if k is coprime with N
}

// Naive RVV: process N/vl chunks of consecutive DESTINATION elements.
// For each dst[d..d+vl-1], compute source indices via k_inv and use
// vloxei64 (indexed/gather load) to fetch from src.
// This avoids the 2D decomposition but still benefits from vector
// register width for the load/store width.
void automorph_naive_rvv(const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k) {
    uint32_t k_inv = mod_inverse(k, N);
    uint64_t N_mask = (uint64_t)N - 1; // N is power-of-2, so mod = & mask

    size_t vl = __riscv_vsetvl_e64m1(8); // VLEN=512 → 8 u64 elements

    // Identity vector [0, 1, 2, ..., vl-1] for building dst index groups
    vuint64m1_t v_iota   = __riscv_vid_v_u64m1(vl);
    vuint64m1_t v_kinv   = __riscv_vmv_v_x_u64m1((uint64_t)k_inv, vl);
    vuint64m1_t v_nmask  = __riscv_vmv_v_x_u64m1(N_mask, vl);

    for (uint32_t d = 0; d < N; d += (uint32_t)vl) {
        // dst indices: [d, d+1, ..., d+vl-1]
        vuint64m1_t v_didx = __riscv_vadd_vx_u64m1(v_iota, (uint64_t)d, vl);

        // src index = (dst_idx * k_inv) & (N-1)
        vuint64m1_t v_sidx = __riscv_vmul_vv_u64m1(v_didx, v_kinv, vl);
        v_sidx = __riscv_vand_vv_u64m1(v_sidx, v_nmask, vl);

        // byte offset = src_idx * 8
        vuint64m1_t v_boff = __riscv_vsll_vx_u64m1(v_sidx, 3, vl);

        // Indexed (gather) load from src
        vuint64m1_t v_data = __riscv_vloxei64_v_u64m1(src, v_boff, vl);

        // Sequential store to dst[d..d+vl-1]
        __riscv_vse64_v_u64m1(dst + d, v_data, vl);
    }
}
