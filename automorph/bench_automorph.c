#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "automorph.h"

#define BENCH_REPS 5

static void fill_src(uint64_t *arr, uint32_t N) {
    for (uint32_t i = 0; i < N; i++)
        arr[i] = (uint64_t)i * 0x9E3779B185EBCA87ULL;
}

static int verify(const uint64_t *ref, const uint64_t *got, uint32_t N,
                  const char *label, uint32_t k) {
    for (uint32_t i = 0; i < N; i++) {
        if (ref[i] != got[i]) {
            printf("  FAIL %s k=%u: mismatch at idx %u (ref=%llu got=%llu)\n",
                   label, k, i,
                   (unsigned long long)ref[i],
                   (unsigned long long)got[i]);
            return 1;
        }
    }
    return 0;
}

static void run_k(uint64_t *src, uint64_t *ref, uint64_t *buf_naive,
                  uint64_t *buf_rvv, uint32_t N, uint32_t k) {
    printf("  k=%-6u", k);

    fill_src(src, N);

    // --- scalar reference ---
    memset(ref, 0, N * sizeof(uint64_t));
    automorph_scalar(src, ref, N, k);

    uint64_t t0, t1, best_scalar = UINT64_MAX;
    for (int r = 0; r < BENCH_REPS; r++) {
        memset(ref, 0, N * sizeof(uint64_t));
        t0 = rdcycle();
        automorph_scalar(src, ref, N, k);
        t1 = rdcycle();
        if (t1 - t0 < best_scalar) best_scalar = t1 - t0;
    }

    // --- naive RVV ---
    memset(buf_naive, 0, N * sizeof(uint64_t));
    automorph_naive_rvv(src, buf_naive, N, k);

    uint64_t best_naive = UINT64_MAX;
    for (int r = 0; r < BENCH_REPS; r++) {
        memset(buf_naive, 0, N * sizeof(uint64_t));
        t0 = rdcycle();
        automorph_naive_rvv(src, buf_naive, N, k);
        t1 = rdcycle();
        if (t1 - t0 < best_naive) best_naive = t1 - t0;
    }
    int naive_ok = !verify(ref, buf_naive, N, "naive_rvv", k);

    // --- optimized RVV (2D HFAuto) ---
    memset(buf_rvv, 0, N * sizeof(uint64_t));
    automorph_rvv(src, buf_rvv, N, k);

    uint64_t best_rvv = UINT64_MAX;
    for (int r = 0; r < BENCH_REPS; r++) {
        memset(buf_rvv, 0, N * sizeof(uint64_t));
        t0 = rdcycle();
        automorph_rvv(src, buf_rvv, N, k);
        t1 = rdcycle();
        if (t1 - t0 < best_rvv) best_rvv = t1 - t0;
    }
    int rvv_ok = !verify(ref, buf_rvv, N, "automorph_rvv", k);

    uint64_t speedup_x10 = (best_rvv > 0) ? (best_scalar * 10) / best_rvv : 0;

    printf("  scalar=%7llu  naive=[%s]%7llu  rvv=[%s]%7llu  speedup=%lu.%lux\n",
           (unsigned long long)best_scalar,
           naive_ok ? "OK" : "!!", (unsigned long long)best_naive,
           rvv_ok   ? "OK" : "!!", (unsigned long long)best_rvv,
           (unsigned long)(speedup_x10 / 10),
           (unsigned long)(speedup_x10 % 10));
}

static void run_N(uint32_t N) {
    printf("\n=== N=%-6u (R=8, C=%u, stride=%u bytes) ===\n",
           N, N / 8, (N / 8) * 8);

    uint64_t *src      = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *ref      = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *buf_naive = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *buf_rvv  = (uint64_t *)malloc(N * sizeof(uint64_t));

    // Fixed k values + N-1 (largest valid odd k for this N)
    uint32_t k_fixed[] = {1, 3, 7, 31, 257};
    int nk_fixed = sizeof(k_fixed) / sizeof(k_fixed[0]);

    for (int i = 0; i < nk_fixed; i++)
        run_k(src, ref, buf_naive, buf_rvv, N, k_fixed[i]);

    // N-1 is always odd when N is a power of 2
    run_k(src, ref, buf_naive, buf_rvv, N, N - 1);

    free(src); free(ref); free(buf_naive); free(buf_rvv);
}

int main(void) {
    printf("bench_automorph: VLEN=512, BENCH_REPS=%d\n", BENCH_REPS);
    printf("  columns: scalar / naive_rvv / automorph_rvv (cycles, best-of-%d)\n",
           BENCH_REPS);

    uint32_t n_vals[] = {4096, 8192, 16384, 32768};
    int nn = sizeof(n_vals) / sizeof(n_vals[0]);

    for (int i = 0; i < nn; i++)
        run_N(n_vals[i]);

    return 0;
}
