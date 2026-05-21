/*
 * test_mulhu.c — minimal test for vmulhu_vv_u64m1 correctness
 * Verifies vmulhu and vmul against scalar __uint128_t.
 */
#include <stdint.h>
#include <stdio.h>

#ifdef __riscv_vector
#include <riscv_vector.h>

static int test_vmulhu_vv(uint64_t a, uint64_t b) {
    /* scalar reference */
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t want_hi = (uint64_t)(prod >> 64);
    uint64_t want_lo = (uint64_t)prod;

    /* RVV: put a and b into 1-element vectors */
    size_t vl = __riscv_vsetvl_e64m1(1);
    vuint64m1_t va = __riscv_vmv_v_x_u64m1(a, vl);
    vuint64m1_t vb = __riscv_vmv_v_x_u64m1(b, vl);

    vuint64m1_t v_hi = __riscv_vmulhu_vv_u64m1(va, vb, vl);
    vuint64m1_t v_lo = __riscv_vmul_vv_u64m1(va, vb, vl);

    uint64_t got_hi, got_lo;
    __riscv_vse64_v_u64m1(&got_hi, v_hi, vl);
    __riscv_vse64_v_u64m1(&got_lo, v_lo, vl);

    int ok = (got_hi == want_hi && got_lo == want_lo);
    if (!ok) {
        printf("  FAIL a=0x%llx b=0x%llx\n"
               "       want hi=0x%llx lo=0x%llx\n"
               "       got  hi=0x%llx lo=0x%llx\n",
               (unsigned long long)a, (unsigned long long)b,
               (unsigned long long)want_hi, (unsigned long long)want_lo,
               (unsigned long long)got_hi, (unsigned long long)got_lo);
    }
    return ok;
}

/* Test vsrl/vsll and vor that compose RShiftD */
static int test_rshiftd(uint64_t prod_hi, uint64_t prod_lo, int shift) {
    /* scalar: RShiftD = (lo >> shift) | (hi << (64-shift)) */
    uint64_t want = (prod_lo >> shift) | (prod_hi << (64 - shift));

    size_t vl = __riscv_vsetvl_e64m1(1);
    vuint64m1_t v_hi = __riscv_vmv_v_x_u64m1(prod_hi, vl);
    vuint64m1_t v_lo = __riscv_vmv_v_x_u64m1(prod_lo, vl);

    vuint64m1_t v_res = __riscv_vor_vv_u64m1(
                            __riscv_vsrl_vx_u64m1(v_lo, shift, vl),
                            __riscv_vsll_vx_u64m1(v_hi, 64 - shift, vl), vl);

    uint64_t got;
    __riscv_vse64_v_u64m1(&got, v_res, vl);

    int ok = (got == want);
    if (!ok)
        printf("  FAIL RShiftD(hi=0x%llx lo=0x%llx shift=%d) want=0x%llx got=0x%llx\n",
               (unsigned long long)prod_hi, (unsigned long long)prod_lo, shift,
               (unsigned long long)want, (unsigned long long)got);
    return ok;
}

/* Trace one full Barrett computation element-by-element */
static void trace_barrett(uint64_t a, uint64_t b, uint64_t q, uint64_t mu,
                           int n_shift, int n7, int n64, int n764)
{
    /* scalar step-by-step */
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t prod_hi = (uint64_t)(prod >> 64);
    uint64_t prod_lo = (uint64_t)prod;
    uint64_t x1      = (prod_lo >> n_shift) | (prod_hi << n64);
    __uint128_t prod2 = (__uint128_t)x1 * mu;
    uint64_t p2_hi   = (uint64_t)(prod2 >> 64);
    uint64_t p2_lo   = (uint64_t)prod2;
    uint64_t q_est   = (p2_lo >> n7) | (p2_hi << n764);
    uint64_t q_rnd   = (uint64_t)((__uint128_t)q_est * q);
    uint64_t res     = prod_lo - q_rnd;
    if (res >= q) res -= q;
    if (res >= q) res -= q;

    printf("  scalar: prod(%llx:%llx) x1=%llx q_est=%llx q_rnd=%llx res=%llu\n",
           (unsigned long long)prod_hi, (unsigned long long)prod_lo,
           (unsigned long long)x1, (unsigned long long)q_est,
           (unsigned long long)q_rnd, (unsigned long long)res);

    /* RVV step-by-step, 1 element */
    size_t vl = __riscv_vsetvl_e64m1(1);
    vuint64m1_t va = __riscv_vmv_v_x_u64m1(a, vl);
    vuint64m1_t vb_vec = __riscv_vmv_v_x_u64m1(b, vl);

    vuint64m1_t v_hi_ab = __riscv_vmulhu_vv_u64m1(va, vb_vec, vl);
    vuint64m1_t v_lo_ab = __riscv_vmul_vv_u64m1(va, vb_vec, vl);
    uint64_t rvv_hi, rvv_lo;
    __riscv_vse64_v_u64m1(&rvv_hi, v_hi_ab, vl);
    __riscv_vse64_v_u64m1(&rvv_lo, v_lo_ab, vl);

    vuint64m1_t v_x1 = __riscv_vor_vv_u64m1(
                           __riscv_vsrl_vx_u64m1(v_lo_ab, n_shift, vl),
                           __riscv_vsll_vx_u64m1(v_hi_ab, n64, vl), vl);
    uint64_t rvv_x1;
    __riscv_vse64_v_u64m1(&rvv_x1, v_x1, vl);

    vuint64m1_t vmu = __riscv_vmv_v_x_u64m1(mu, vl);
    vuint64m1_t v_hi2 = __riscv_vmulhu_vv_u64m1(v_x1, vmu, vl);
    vuint64m1_t v_lo2 = __riscv_vmul_vv_u64m1(v_x1, vmu, vl);
    uint64_t rvv_hi2, rvv_lo2;
    __riscv_vse64_v_u64m1(&rvv_hi2, v_hi2, vl);
    __riscv_vse64_v_u64m1(&rvv_lo2, v_lo2, vl);

    vuint64m1_t v_qest = __riscv_vor_vv_u64m1(
                             __riscv_vsrl_vx_u64m1(v_lo2, n7, vl),
                             __riscv_vsll_vx_u64m1(v_hi2, n764, vl), vl);
    uint64_t rvv_qest;
    __riscv_vse64_v_u64m1(&rvv_qest, v_qest, vl);

    vuint64m1_t vq = __riscv_vmv_v_x_u64m1(q, vl);
    vuint64m1_t v_qrnd = __riscv_vmul_vv_u64m1(v_qest, vq, vl);
    uint64_t rvv_qrnd;
    __riscv_vse64_v_u64m1(&rvv_qrnd, v_qrnd, vl);

    printf("  rvv:    prod(%llx:%llx) x1=%llx q_est=%llx q_rnd=%llx\n",
           (unsigned long long)rvv_hi, (unsigned long long)rvv_lo,
           (unsigned long long)rvv_x1, (unsigned long long)rvv_qest,
           (unsigned long long)rvv_qrnd);

    printf("  match: prod_hi=%s x1=%s p2_hi=%s q_est=%s q_rnd=%s\n",
           rvv_hi==prod_hi?"OK":"FAIL",
           rvv_x1==x1?"OK":"FAIL",
           rvv_hi2==p2_hi?"OK":"FAIL",
           rvv_qest==q_est?"OK":"FAIL",
           rvv_qrnd==q_rnd?"OK":"FAIL");
}
#endif

int main(void) {
#ifndef __riscv_vector
    printf("No RVV\n");
    return 0;
#else
    int errors = 0;

    printf("=== vmulhu_vv correctness ===\n");
    /* small values */
    errors += !test_vmulhu_vv(2, 3);
    errors += !test_vmulhu_vv(0xFFFFFFFFFFFFFFFFULL, 2);
    errors += !test_vmulhu_vv(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
    /* 50-bit values */
    errors += !test_vmulhu_vv(1058257143813929ULL, 985158809405133ULL);
    errors += !test_vmulhu_vv(597707249538636ULL, 246609054067617ULL);
    /* 55-bit values */
    errors += !test_vmulhu_vv(31019554379626150ULL, 562799706797575ULL);

    printf("vmulhu_vv: %d errors\n\n", errors);

    printf("=== RShiftD (vsrl+vsll+vor) ===\n");
    /* Test with known prod_hi/lo from scalar computation */
    {
        uint64_t a=1058257143813929ULL, b=985158809405133ULL;
        __uint128_t p = (__uint128_t)a*b;
        uint64_t phi=(uint64_t)(p>>64), plo=(uint64_t)p;
        errors += !test_rshiftd(phi, plo, 48);  /* n_shift for ~50bit q */
        errors += !test_rshiftd(phi, plo, 53);  /* n_shift for ~55bit q */
    }
    printf("RShiftD: %d total errors so far\n\n", errors);

    printf("=== Full Barrett trace (failing element) ===\n");
    /* q~50bit failing case from Part2 */
    {
        uint64_t q=1125899906842597ULL;
        int msb=50, n_shift=48, n7=55, n64=16, n764=9;
        /* compute mu */
        __uint128_t num = (__uint128_t)1 << (2*msb+3);
        uint64_t mu = (uint64_t)(num / q);
        printf("q=%llu n=%d n7=%d mu=%llu\n",
               (unsigned long long)q, n_shift, n7, (unsigned long long)mu);
        /* failing element: a=1058257143813929, b=985158809405133 */
        trace_barrett(1058257143813929ULL, 985158809405133ULL,
                      q, mu, n_shift, n7, n64, n764);
    }

    printf("\n%s\n", errors==0 ? "ALL PASS" : "FAILURES FOUND");
    return errors != 0;
#endif
}
