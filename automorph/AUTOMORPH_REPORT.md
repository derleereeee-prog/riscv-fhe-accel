# RISC-V RVV Automorphism Transform Optimization

> **Date**: 2026-05-21  
> **Git**: `e9995d6` (`git log --oneline automorph/`)  
> **Platform**: gem5 v24.1.0.2, RiscvO3CPU, VLEN=512 bits, 1 GHz  
> **Result file**: `automorph/results/bench_20260521_221222.txt`

---

## 1. Background

### 1.1 Role of Automorphism in CKKS

CKKS homomorphic encryption packs $N/2$ plaintext values into a single ciphertext. A slot rotation by $r$ positions is realized via the **automorphism** map $\phi_k$, where $k = 5^r \bmod 2N$.

The call chain for one `EvalRotate`:

```
EvalRotate(ct, r)
  └─ AutomorphismTransform(ct.b, k)   ← polynomial permutation  [this work]
  └─ AutomorphismTransform(ct.a, k)
  └─ KeySwitch(ct')                   ← restore key structure
       └─ NTT → ModMulNoCheckEq → INTT
```

In Baby-Step Giant-Step (BSGS) matrix-vector multiplication, approximately $\sqrt{N_\text{slots}}$ distinct rotations are applied per layer. For $N=4096$ this means ~64 `EvalRotate` calls per BSGS layer, making `AutomorphismTransform` a dominant cost. Profiling confirms **EvalRotate accounts for 69.2% of BSGS runtime**.

### 1.2 Mathematical Definition

Given an odd integer $k$ and a polynomial represented as $N$ coefficients in the NTT (evaluation) domain:

$$\text{dst}[(i \cdot k) \bmod N] = \text{src}[i], \quad \forall\, i \in [0, N)$$

This is a length-$N$ permutation of `uint64_t` elements. The permutation changes with each rotation amount $k$.

---

## 2. Implementations

### 2.1 Scalar Reference (`automorph_scalar.c`)

```c
for (uint32_t i = 0; i < N; i++)
    dst[((uint64_t)i * k) % N] = src[i];
```

Direct scatter write with integer modulo. Used as correctness reference and performance baseline.

### 2.2 Naïve RVV (`automorph_naive_rvv.c`)

Computes the inverse mapping $k^{-1} \bmod N$ via brute-force search, then uses `vloxei64` (indexed gather) to read permuted source elements and writes sequentially to dst. Performance is highly dependent on $k$ because the gather pattern is fully random.

### 2.3 Optimized RVV: 2D HFAuto Decomposition (`automorph_rvv.c`)

#### 2.3.1 Core Idea

Treat the $N$-element array as an $R \times C$ matrix stored **row-major**, where $R = \text{VLEN}/64$ (= 8 for VLEN=512) and $C = N/R$:

$$\text{src}[i \cdot C + j] \;\equiv\; \text{element at row } i,\; \text{col } j$$

The automorphism $\phi_k$ decomposes in 2D as:

$$\text{Destination col:}\quad J = (j \cdot k) \bmod C$$
$$\text{Destination row:}\quad I = \bigl(i \cdot k + \lfloor j \cdot k / C \rfloor\bigr) \bmod R$$

The column mapping $J$ depends only on $j$. The row mapping depends on both $i$ and a per-column offset $\delta[j] = \lfloor j \cdot k / C \rfloor \bmod R$.

#### 2.3.2 Per-Column Kernel (4 RVV instructions)

For each column $j$ (C outer iterations), all $R=8$ rows are processed as a single vector:

| Step | Instruction | Operation |
|------|-------------|-----------|
| 1 | `vloxei64.v` | Gather column $j$ from src (8 rows, stride $C \times 8$ bytes) |
| 2 | `vle64.v` | Load precomputed inverse permutation for $\delta[j]$ |
| 3 | `vrgather.vv` | In-register row permutation: `v_perm[d] = v_col[idx[d]]` |
| 4 | `vsoxei64.v` | Scatter to destination column $J = \text{colmap}[j]$ |

#### 2.3.3 Why Inverse Permutation for `vrgather`

The row mapping is a **scatter** (source row $i$ → destination row $I$). `vrgather` performs a **gather** (`dst[d] = src[idx[d]]`). To express scatter as gather, the index must satisfy:

$$\text{idx}[d] = \sigma^{-1}(d) = \text{``which source row maps to destination row } d\text{''}$$

For $R = 2^m$ and any odd $k$: $k^2 \equiv 1 \pmod{R}$, so $k^{-1} \equiv k \pmod{R}$ and $\sigma = \sigma^{-1}$ (self-inverse). The lookup table becomes:

$$\text{inv\_idx\_table}[\delta][d] = \bigl((d - \delta + R) \cdot k\bigr) \bmod R$$

#### 2.3.4 Precomputed Tables

| Table | Size | Content |
|-------|------|---------|
| `rowmap_table[R]` | 8 entries | $(i \cdot k) \bmod R$ |
| `inv_idx_table[R][R]` | 64 entries | $((d-\delta+R) \cdot k) \bmod R$ |
| `stride_offsets[R]` | 8 entries | $[0,\, C{\cdot}8,\, 2C{\cdot}8,\; \ldots,\; 7C{\cdot}8]$ bytes |
| `delta_arr[C]` | $C$ entries | $\lfloor j \cdot k / C \rfloor \bmod R$ |
| `colmap_arr[C]` | $C$ entries | $(j \cdot k) \bmod C$ |

Cached by $(N, k)$; recomputed only on change.

#### 2.3.5 Indexed vs Strided Load/Store

Column access requires stride $= C \times 8$ bytes. For $N=4096$, stride $= 4096 =$ page size. **gem5 v24.1.0.2 has a known bug** where `vsse64.v` with stride equal to page size produces incorrect results. All strided accesses are replaced with `vloxei64`/`vsoxei64` using precomputed byte-offset vectors.

---

## 3. Debugging History

### Bug 1 — Wrong Permutation Direction

**Symptom**: $k=7$ and $k=4095$ pass; all other $k$ fail.  
**Cause**: `vrgather` indexed with forward permutation $\sigma(i)$ instead of inverse $\sigma^{-1}(d)$. For $k \equiv 7 \pmod{8}$: $49 \equiv 1 \pmod{8}$, so $\sigma=\sigma^{-1}$, masking the bug.  
**Fix**: Precompute `inv_idx_table[delta][d]` and load with `vle64`.

### Bug 2 — `vid.v` Wrong Output in gem5

**Symptom**: All $k$ fail, outputs are `UINT64_MAX`.  
**Cause**: `__riscv_vid_v_u64m1()` has a known gem5 v24.1 implementation bug producing incorrect values.  
**Fix**: Replace with `vle64` loading from a precomputed static array.

### Bug 3 — `vsse64.v` Fails at Stride = Page Size

**Symptom**: $k=1$ passes; $k \in \{3,5,7,31,257\}$ output `UINT64_MAX`; $k=4095$ passes.  
**Diagnosis**: $N=64$ (stride = 64 B) passes all $k$ → isolated to stride = 4096 B = page size.  
**Cause**: gem5 v24.1.0.2 bug in `vsse64.v` when stride equals page size.  
**Fix**: Replace `vlse64`/`vsse64` with `vloxei64`/`vsoxei64` + precomputed offsets.

---

## 4. Benchmark Results

**Setup**: gem5 RiscvO3CPU, VLEN=512, 1 GHz. Metric: `rdcycle` CSR, best-of-5 repetitions.  
**Source**: `automorph/results/bench_20260521_221222.txt`, git `e9995d6`.

### 4.1 N = 4096 (32 KB — fits in L1 cache)

| k | scalar | naive\_rvv | automorph\_rvv | speedup |
|---|-------:|----------:|--------------:|--------:|
| 1 | 86,032 | 2,103 ✓ | 2,619 ✓ | **32.8×** |
| 3 | 86,031 | 32,166 ✓ | 2,637 ✓ | **32.6×** |
| 7 | 86,031 | 40,753 ✓ | 2,639 ✓ | **32.5×** |
| 31 | 86,031 | 35,556 ✓ | 2,617 ✓ | **32.8×** |
| 257 | 86,031 | 44,372 ✓ | 2,624 ✓ | **32.7×** |
| 4095 | 86,031 | 47,176 ✓ | 2,623 ✓ | **32.7×** |

**Speedup range**: 32.5×–32.8× (variation < 1%)

### 4.2 N = 8192 (64 KB — L1/L2 boundary)

| k | scalar | naive\_rvv | automorph\_rvv | speedup |
|---|-------:|----------:|--------------:|--------:|
| 1 | 172,047 | 5,876 ✓ | 7,905 ✓ | **21.7×** |
| 3 | 172,047 | 35,610 ✓ | 7,803 ✓ | **22.0×** |
| 7 | 172,047 | 44,180 ✓ | 8,686 ✓ | **19.8×** |
| 31 | 172,047 | 84,433 ✓ | 8,191 ✓ | **21.0×** |
| 257 | 172,047 | 94,631 ✓ | 11,273 ✓ | **15.2×** |
| 8191 | 172,047 | 95,777 ✓ | 7,279 ✓ | **23.6×** |

**Speedup range**: 15.2×–23.6×

### 4.3 N = 16384 (128 KB — L2 cache)

| k | scalar | naive\_rvv | automorph\_rvv | speedup |
|---|-------:|----------:|--------------:|--------:|
| 1 | 344,079 | 12,176 ✓ | 20,097 ✓ | **17.1×** |
| 3 | 344,079 | 132,366 ✓ | 25,064 ✓ | **13.7×** |
| 7 | 344,079 | 140,951 ✓ | 37,307 ✓ | **9.2×** |
| 31 | 344,079 | 180,957 ✓ | 65,856 ✓ | **5.2×** |
| 257 | 344,079 | 191,041 ✓ | 60,728 ✓ | **5.6×** |
| 16383 | 344,079 | 192,419 ✓ | 18,653 ✓ | **18.4×** |

**Speedup range**: 5.2×–18.4×

### 4.4 N = 32768 (256 KB — L2/L3)

| k | scalar | naive\_rvv | automorph\_rvv | speedup |
|---|-------:|----------:|--------------:|--------:|
| 1 | 688,143 | 24,448 ✓ | 38,596 ✓ | **17.8×** |
| 3 | 688,143 | 144,669 ✓ | 57,683 ✓ | **11.9×** |
| 7 | 688,143 | 333,479 ✓ | 101,742 ✓ | **6.7×** |
| 31 | 688,143 | 373,485 ✓ | 211,802 ✓ | **3.2×** |
| 257 | 688,143 | 383,569 ✓ | 206,123 ✓ | **3.3×** |
| 32767 | 688,143 | 384,947 ✓ | 41,923 ✓ | **16.4×** |

**Speedup range**: 3.2×–17.8×

### 4.5 Summary Table

| N | Data size | Speedup range | Best k | Worst k |
|---|-----------|--------------|--------|---------|
| 4,096 | 32 KB | **32.5–32.8×** | any | — (stable) |
| 8,192 | 64 KB | 15.2–23.6× | 8191 | 257 |
| 16,384 | 128 KB | 5.2–18.4× | 16383 | 31 |
| 32,768 | 256 KB | 3.2–17.8× | 1 | 31 |

---

## 5. Analysis

### 5.1 Cache Sensitivity

The dominant performance factor for large $N$ is **cache behavior of the scatter write** (`vsoxei64` to dst).

For $N=4096$: the array is 32 KB, fitting within a typical 32–64 KB L1 data cache. Both src and dst reside in L1 after the first pass; indexed accesses hit cache, yielding consistent ~32.7× speedup regardless of $k$.

For larger $N$: each `vsoxei64` scatter writes to column $\text{colmap}[j]$ whose position is determined by $(j \cdot k) \bmod C$. When $k$ produces a "random-looking" column map (e.g. $k=31, 257$), scatter writes span the entire dst array, generating cache misses proportional to $N$.

For $k=1$ or $k=N-1$: the column map is nearly sequential (identity or reversal), giving much better cache locality and higher speedup even at large $N$.

### 5.2 Cycles per Element

| N | scalar (cyc/elem) | automorph\_rvv (cyc/elem, median k) |
|---|------------------|------------------------------------|
| 4,096 | 21.0 | **0.64** |
| 8,192 | 21.0 | **1.0** |
| 16,384 | 21.0 | **5.1** (k=31) |
| 32,768 | 21.0 | **6.5** (k=31) |

Scalar cycles/element is constant (modulo is equally expensive for all $N$). RVV degrades at large $N$ due to cache miss pressure from scatter writes.

### 5.3 k=N-1 Performance Anomaly

$k=N-1$ consistently achieves speedup comparable to $k=1$, even at large $N$. This is because $(j \cdot (N-1)) \bmod C = (jN - j) \bmod C = (-j) \bmod C = C - j$, which is a simple column reversal — a regular pattern with good cache locality.

### 5.4 Implication for FHE

In CKKS BSGS with $N=4096$ (ring dimension), all `AutomorphismTransform` calls see $N=4096$ with consistent **~32.7× speedup**. This is the target configuration for this work.

For $N=8192$, the average speedup across typical BSGS rotation amounts is approximately **15–22×**, still substantial. For $N \geq 16384$, a different memory layout (e.g. pre-transposing the polynomial or block-wise processing) would be needed to recover cache locality.

---

## 6. OpenFHE Integration

### 6.1 Architecture

OpenFHE precomputes a permutation index array once per rotation key:

```
precomp[j] = source index for destination position j
dst[j] = src[precomp[j]]   for all j in [0, N)
```

This reduces `AutomorphismTransform` to a **pure indexed gather**, implemented in `automorph-rvv.h`:

```c
// LMUL=m2: 16 elements per operation
vuint32m1_t v_idx32 = vle32(precomp + j, vl);         // load 16 indices
vuint64m2_t v_boff  = vzext_vf2(v_idx32, vl);         // zero-extend
v_boff = vsll_vx(v_boff, 3, vl);                      // byte offset = idx * 8
vuint64m2_t v_data  = vloxei64(src, v_boff, vl);      // gather from src
vse64(dst + j, v_data, vl);                            // sequential write to dst
```

Write to dst is **unit-stride** (`vse64`), which is always cache-friendly.

### 6.2 Standalone vs OpenFHE Integration

| | `automorph_rvv.c` | `automorph-rvv.h` |
|---|---|---|
| Needs precomp | No (computes on-the-fly) | Yes (from OpenFHE) |
| dst write | `vsoxei64` (scatter) | `vse64` (sequential) ✓ |
| LMUL | m1 (8 elem/op) | m2 (16 elem/op) ✓ |
| Use case | Standalone benchmark | OpenFHE EvalRotate |

### 6.3 Pending Integration Work

- [ ] Refactor `automorph-rvv.h` from `static inline` to link-time `.o` (consistent with Barrett/NTT pattern)
- [ ] Create `automorph.patch` for clean OpenFHE checkout
- [ ] Apply to `openfhe-install/` (currently only in `openfhe-install-clang18/`)
- [ ] End-to-end BSGS benchmark with all three optimizations combined

---

## 7. Reproducibility

```bash
# Environment
gem5 v24.1.0.2, clang 18.1.3, --target=riscv64-linux-gnu, -march=rv64gcv

# Build
cd ~/NTT-RVV-project/automorph
make clean && make benchmark

# Run with saved output
STAMP=$(date +%Y%m%d_%H%M%S)
GIT=$(git -C .. rev-parse --short HEAD)
{ echo "# git: $GIT"; echo "# date: $(date)"; } > results/bench_${STAMP}.txt
~/gem5.opt ../gem5-model/main.py bin/bench_automorph >> results/bench_${STAMP}.txt 2>&1
```

gem5 config: `gem5-model/main.py` — RiscvO3CPU, VLEN=512, 8 GiB RAM, SimdDiv FU added.

---

## 8. File Index

```
automorph/
├── automorph.h                — declarations + rdcycle() inline asm
├── automorph_scalar.c         — scalar reference
├── automorph_naive_rvv.c      — naïve RVV (k-dependent performance)
├── automorph_rvv.c            — 2D HFAuto RVV [this work]
├── bench_automorph.c          — benchmark driver (N=4096..32768)
├── Makefile
├── REPORT.md                  — development log
├── AUTOMORPH_REPORT.md        — this document
└── results/
    └── bench_20260521_221222.txt   — benchmark output, git e9995d6

openfhe-install-clang18/include/openfhe/core/
├── automorph/automorph-rvv.h       — OpenFHE integration (precomp gather)
└── lattice/hal/default/poly-impl.h — patched AutomorphismTransform
```
