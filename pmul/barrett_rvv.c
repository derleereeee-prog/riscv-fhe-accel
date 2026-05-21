#include "barrett_rvv.h"
#include <riscv_vector.h>

static vuint64m1_t barrett_mul_vl(
        vuint64m1_t va, vuint64m1_t vb,
        vuint64m1_t vmu, vuint64m1_t vq,
        int n_shift, int n7, int n64, int n764,
        size_t vl)
{
    vuint64m1_t hi_ab = __riscv_vmulhu_vv_u64m1(va, vb, vl);
    vuint64m1_t lo_ab = __riscv_vmul_vv_u64m1(va, vb, vl);

    vuint64m1_t x1 = __riscv_vor_vv_u64m1(
                         __riscv_vsrl_vx_u64m1(lo_ab, n_shift, vl),
                         __riscv_vsll_vx_u64m1(hi_ab, n64, vl), vl);

    vuint64m1_t hi2 = __riscv_vmulhu_vv_u64m1(x1, vmu, vl);
    vuint64m1_t lo2 = __riscv_vmul_vv_u64m1(x1, vmu, vl);

    vuint64m1_t q_est = __riscv_vor_vv_u64m1(
                            __riscv_vsrl_vx_u64m1(lo2, n7, vl),
                            __riscv_vsll_vx_u64m1(hi2, n764, vl), vl);

    vuint64m1_t q_rnd = __riscv_vmul_vv_u64m1(q_est, vq, vl);
    vuint64m1_t res   = __riscv_vsub_vv_u64m1(lo_ab, q_rnd, vl);

    vbool64_t m = __riscv_vmsgeu_vv_u64m1_b64(res, vq, vl);
    res = __riscv_vsub_vv_u64m1_mu(m, res, res, vq, vl);
    m   = __riscv_vmsgeu_vv_u64m1_b64(res, vq, vl);
    res = __riscv_vsub_vv_u64m1_mu(m, res, res, vq, vl);

    return res;
}

void barrett_mul_rvv_u64m1(
        const uint64_t* a,
        const uint64_t* b,
        uint64_t* out,
        size_t n,
        uint64_t q, uint64_t mu,
        int n_shift, int n7, int n64, int n764)
{
    for (size_t off = 0; off < n; ) {
        size_t vl = __riscv_vsetvl_e64m1(n - off);
        vuint64m1_t vmu = __riscv_vmv_v_x_u64m1(mu, vl);
        vuint64m1_t vq  = __riscv_vmv_v_x_u64m1(q,  vl);
        vuint64m1_t va  = __riscv_vle64_v_u64m1(a + off, vl);
        vuint64m1_t vb  = __riscv_vle64_v_u64m1(b + off, vl);
        vuint64m1_t res = barrett_mul_vl(va, vb, vmu, vq,
                                         n_shift, n7, n64, n764, vl);
        __riscv_vse64_v_u64m1(out + off, res, vl);
        off += vl;
    }
}
