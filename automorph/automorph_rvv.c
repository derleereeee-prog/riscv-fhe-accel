#include <riscv_vector.h>
#include <stdlib.h>
#include "automorph.h"

// 2D HFAuto algorithm adapted for VLEN=512 (R=8 elements per register).
//
// Logical layout: treat the N-element polynomial as a 2D matrix of
// shape R x C where R=8, C=N/8.  Index mapping:
//   src element at (row i, col j), i.e. src[i*C + j]
//   goes to dst  at (row I, col J):
//     J = (j * k) mod C          -- column map (only depends on j)
//     I = (i * k mod R + floor(j*k/C) mod R) mod R   -- row map
//
// Per-column iteration (C outer steps, R=8 vector ops each):
//   1. vloxei64.v  – indexed (gather) load of column j
//   2. vle64.v     – load precomputed inverse permutation for this delta
//   3. vrgather.vv – permute the 8 elements
//   4. vsoxei64.v  – indexed (scatter) store to column colmap[j]
//
// We use vloxei64/vsoxei64 (indexed loads/stores) instead of
// vlse64/vsse64 (strided loads/stores) because gem5 v24.1.0.2 has a
// known bug with vsse64 when stride == page_size (4096 bytes for N=4096).
// Indexed loads/stores with precomputed byte-offset vectors work correctly.
//
// Precomputed tables (cached by (N, k), recomputed on change):
//   rowmap[R]         – rowmap[i] = (i*k) mod R
//   inv_idx_table[R][R] – inv_idx_table[d][i] = rowmap[(i-d+R) mod R]
//   delta_arr[C]      – delta_arr[j] = floor(j*k/C) mod R
//   colmap_arr[C]     – colmap_arr[j] = (j*k) mod C
//   stride_offsets[R] – stride_offsets[i] = i * C * sizeof(uint64_t)

#define AUTOMORPH_R 8
#define AUTOMORPH_R_MASK ((uint64_t)(AUTOMORPH_R - 1))  // 7

static uint32_t  cached_k;
static uint32_t  cached_N;
static uint64_t  rowmap_table[AUTOMORPH_R];
static uint64_t  inv_idx_table[AUTOMORPH_R][AUTOMORPH_R];
static uint64_t  stride_offsets[AUTOMORPH_R]; // [0, C*8, 2*C*8, ..., 7*C*8]
static uint64_t *delta_arr  = NULL;
static uint64_t *colmap_arr = NULL;

static void precompute(uint32_t N, uint32_t k) {
    uint32_t C = N >> 3; // N / R

    if (delta_arr == NULL || cached_N != N) {
        free(delta_arr);
        free(colmap_arr);
        delta_arr  = (uint64_t *)malloc(C * sizeof(uint64_t));
        colmap_arr = (uint64_t *)malloc(C * sizeof(uint64_t));
    }

    // rowmap[i] = (i * k) mod R
    for (uint32_t i = 0; i < AUTOMORPH_R; i++)
        rowmap_table[i] = ((uint64_t)i * k) & AUTOMORPH_R_MASK;

    // inv_idx_table[d][i] = rowmap[(i - d + R) mod R]
    for (uint32_t d = 0; d < AUTOMORPH_R; d++)
        for (uint32_t i = 0; i < AUTOMORPH_R; i++)
            inv_idx_table[d][i] = rowmap_table[(i + AUTOMORPH_R - d) & AUTOMORPH_R_MASK];

    // stride_offsets[i] = i * C * 8 bytes
    for (uint32_t i = 0; i < AUTOMORPH_R; i++)
        stride_offsets[i] = (uint64_t)i * C * sizeof(uint64_t);

    uint64_t C_mask  = (uint64_t)C - 1;
    uint32_t C_shift = 0;
    { uint32_t tmp = C; while (tmp >>= 1) C_shift++; } // log2(C)

    for (uint32_t j = 0; j < C; j++) {
        uint64_t jk   = (uint64_t)j * k;
        colmap_arr[j] = jk & C_mask;
        delta_arr[j]  = (jk >> C_shift) & AUTOMORPH_R_MASK;
    }

    cached_k = k;
    cached_N = N;
}

void automorph_rvv(const uint64_t *src, uint64_t *dst, uint32_t N, uint32_t k) {
    if (k != cached_k || N != cached_N)
        precompute(N, k);

    uint32_t C = N >> 3;
    size_t vl = __riscv_vsetvl_e64m1(AUTOMORPH_R); // always 8

    // Load the stride offset vector [0, C*8, 2*C*8, ..., 7*C*8]
    vuint64m1_t v_strides = __riscv_vle64_v_u64m1(stride_offsets, vl);

    for (uint32_t j = 0; j < C; j++) {
        // Step 1: indexed (gather) load of column j
        //   src byte offsets: (j + i*C) * 8 = j*8 + stride_offsets[i]
        vuint64m1_t v_src_offs = __riscv_vadd_vx_u64m1(
                                     v_strides, (uint64_t)j * sizeof(uint64_t), vl);
        vuint64m1_t v_col = __riscv_vloxei64_v_u64m1(src, v_src_offs, vl);

        // Step 2: load precomputed inverse permutation for this delta
        vuint64m1_t v_idx = __riscv_vle64_v_u64m1(
                                inv_idx_table[delta_arr[j]], vl);

        // Step 3: gather: v_perm[i] = v_col[v_idx[i]]
        vuint64m1_t v_perm = __riscv_vrgather_vv_u64m1(v_col, v_idx, vl);

        // Step 4: indexed (scatter) store to column colmap[j]
        //   dst byte offsets: (colmap[j] + i*C) * 8 = colmap[j]*8 + stride_offsets[i]
        vuint64m1_t v_dst_offs = __riscv_vadd_vx_u64m1(
                                     v_strides, (uint64_t)colmap_arr[j] * sizeof(uint64_t), vl);
        __riscv_vsoxei64_v_u64m1(dst, v_dst_offs, v_perm, vl);
    }
}
