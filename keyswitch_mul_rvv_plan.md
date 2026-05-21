# KeySwitchCore 密钥乘法 RVV 向量化详细方案

> 作者：分析日期 2026-05-18  
> 涉及文件：`keyswitch-hybrid.cpp`、`mubintvecnat.h`、`ubintnat.h`、`poly.h`

---

## 一、热路径精确定位

### 1.1 调用链

```
EvalFastKeySwitchCoreExt()          [keyswitch-hybrid.cpp:456]
  ↓ 双重循环 j × i
  PolyImpl::operator*()             [poly.h:262]
    ↓
  NativeVectorT::ModMulNoCheckEq()  [mubintvecnat.h:492]   ← 向量化目标函数
    ↓ 对每个元素
  NativeIntegerT::ModMulFastEq()    [ubintnat.h:1306]      ← 标量 Barrett 规约
    ↓
  MultD / RShiftD / SubtractD       [ubintnat.h:1892-1950]  ← 128-bit 整数操作
```

### 1.2 EvalFastKeySwitchCoreExt 主循环（keyswitch-hybrid.cpp:456-477）

```cpp
for (uint32_t j = 0; j < digits->size(); j++) {         // j=0,1（digits=2）
    const DCRTPoly& cj = (*digits)[j];                  // 分解后的 j-th digit
    const DCRTPoly& bj = bv[j];                         // 密钥多项式 b[j]
    const DCRTPoly& aj = av[j];                         // 密钥多项式 a[j]

    for (usint i = 0; i < sizeQl; i++) {               // i ∈ [0, sizeQl)，每个 RNS limb
        const auto& cji = cj.GetElementAtIndex(i);     // N个元素的多项式，模 q_i
        const auto& bji = bj.GetElementAtIndex(i);
        const auto& aji = aj.GetElementAtIndex(i);

        cTilda0.SetElementAtIndex(i,
            cTilda0.GetElementAtIndex(i) + cji * bji); // += NTT域逐元素乘法
        cTilda1.SetElementAtIndex(i,
            cTilda1.GetElementAtIndex(i) + cji * aji);
    }
    // 第二段循环 [sizeQl, sizeQlP) 结构相同
}
```

**关键数字（N=4096, MULT_DEPTH=5, SCALE_MOD=50, HEStd_NotSet）：**

| 参数 | 值 |
|------|-----|
| `digits->size()` | 2（dnum=2，hybrid key switching） |
| `sizeQl` | ≈6（当前层 RNS 基底 limb 数） |
| `sizeQlP` | ≈12（扩展基底，含特殊素数） |
| `N`（每个 limb 的元素数） | 4096 |
| 每次 `cji * bji` 调用量 | N=4096 次 Barrett 规约 |
| 总 Barrett 乘法次数 | 2 × 12 × 2 × 4096 ≈ **196K** 次 |

### 1.3 poly::operator* 的实现路径（poly.h:262）

```cpp
PolyImpl Times(const PolyImpl& rhs) const override {
    // 要求 EVALUATION 格式（NTT 域，逐元素乘）
    auto tmp(*this);
    tmp.m_values->ModMulNoCheckEq(*rhs.m_values);  // ← 核心
    return tmp;
}
```

### 1.4 ModMulNoCheckEq（mubintvecnat.h:492）

```cpp
NativeVectorT& ModMulNoCheckEq(const NativeVectorT& b) {
    size_t size{m_data.size()};  // = N = 4096
    auto mv{m_modulus};          // q（单个素数，该 limb 的模数）
#ifdef NATIVEINT_BARRET_MOD
    auto mu{m_modulus.ComputeMu()};  // Barrett 常数
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv, mu);  // ← 每元素 Barrett
#else
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv);      // ← 每元素 128-bit 乘 + 模运算
#endif
    return *this;
}
```

**重要约束**：`m_modulus` 对整个向量是**同一个值**（NativeVector 是单模数向量）。这意味着 q 和 mu 可以作为广播标量送入 RVV —— 不需要每元素 gather 不同的模数。

---

## 二、标量 Barrett 规约的完整数学推导

### 2.1 数据类型

```
NativeInt  = uint64_t
DNativeInt = uint128_t（RISCV 有 HAVE_INT128）
typeD = { uint64_t hi; uint64_t lo; }  // 128-bit 以两个 64-bit 存储
```

关键辅助函数：

```cpp
// 128-bit 右移，返回低 64 位（相当于提取第 shift～shift+63 位）
static NativeInt RShiftD(const typeD& x, int64_t shift) {
    return (x.lo >> shift) | (x.hi << (64 - shift));
}
// 64×64 → 128-bit 乘（RISCV 编译为 mul + mulhu，或直接 uint128_t 乘）
static void MultD(NativeInt a, NativeInt b, typeD& res) {
    uint128_t c = (uint128_t)a * b;
    res.hi = (uint64_t)(c >> 64);
    res.lo = (uint64_t)c;
}
// 128-bit 减法（原地，结果写入第一参数）
static void SubtractD(typeD& a, const typeD& b) {
    // a = a - b (128-bit)
    bool borrow = (a.lo < b.lo);
    a.lo -= b.lo;
    a.hi -= b.hi + (borrow ? 1 : 0);
}
```

### 2.2 Barrett 参数

- `n = modulus.GetMSB() - 2`  
  对 50-bit 素数：GetMSB=50，n=48；对 60-bit 素数：GetMSB=60，n=58
- `mu = ComputeMu()` = ⌊2^(2·GetMSB(q)+3) / q⌋  
  = ⌊2^(2n+7) / q⌋  
  对 50-bit 素数：mu = ⌊2^103/q⌋，约 53 位

### 2.3 ModMulFastEq 步骤（DNativeInt = uint64_t 路径）

```cpp
NativeIntegerT& ModMulFastEq(const NativeIntegerT& b,
                              const NativeIntegerT& modulus,
                              const NativeIntegerT& mu) {
    int64_t n{modulus.GetMSB() - 2};   // 例：n = 48（50-bit 素数）
    uint64_t mv = modulus.m_value;

    typeD prod;
    MultD(m_value, b.m_value, prod);    // Step 1: prod = a*b (128-bit)
    typeD r = prod;                      // 保存原始积（用于最终减法）

    MultD(RShiftD(prod, n), mu.m_value, prod);  // Step 2: prod = (a*b>>n)*mu (128-bit)
    MultD(RShiftD(prod, n+7), mv, prod);         // Step 3: prod = ((a*b>>n)*mu>>(n+7))*q
    SubtractD(r, prod);                          // Step 4: r = a*b - prod（≈ a*b mod q）

    m_value = r.lo;                              // Step 5: 取低 64 位
    if (r.lo >= mv) m_value -= mv;              // Step 6: 条件修正（最多差 q）
    return *this;
}
```

### 2.4 各步骤的位操作含义

设 a, b < q ≈ 2^50，则 a*b < 2^100。

| 步骤 | 计算 | 结果范围 | 说明 |
|------|------|----------|------|
| Step 1 | prod = a×b (128-bit) | < 2^100 | hi=prod>>64, lo=prod&mask |
| RShiftD(prod,n) | (prod>>n) 的低 64 位 | < 2^(100-n)=2^52 | 提取 [n+63:n] 位 |
| Step 2 | x1×mu (128-bit), x1<2^52, mu<2^54 | < 2^106 | |
| RShiftD(step2,n+7) | (step2>>(n+7)) 的低 64 位 | < 2^(106-55)=2^51 | = ⌊a*b/q⌋ 估计值 |
| Step 3 | q_est × q (128-bit) | < q² < 2^100 | q_est×q ≈ ⌊a*b/q⌋×q |
| Step 4 | a*b - q_est×q (128-bit) | ∈ [0, 3q) | Barrett 误差界 ≤ 3q |
| Step 5 | 取 r.lo | ∈ [0, 3q) | r.hi = 0（两 128-bit 抵消） |
| Step 6 | 条件减 q | ∈ [0, 2q) | 实际代码只减一次；在正确参数下 ∈ [0, q) |

**关键结论**：Step 3 中 `MultD(RShiftD(prod, n+7), mv, prod)` 只用 prod 的**低 64 位**（q_est < 2^51）乘以 mv（< 2^50），结果 < 2^101，`prod.lo` 已包含完整信息（SubtractD 只用到低 64 位差，因为 hi 抵消）。

---

## 三、为何 Shoup 乘法不能直接套用

### 3.1 Shoup 乘法原理

NTT 内核用的 Shoup 公式（`ntt-rvv.cpp`）：

```
# 预计算（W 固定）
W_shoup = floor(W × 2^64 / q)   -- 离线计算一次

# 在线（每次调用 V 不同）
q_est = floor(V × W_shoup / 2^64)  ← vmulh(V, W_shoup)
result = V × W - q_est × q          ← vmul 两次 + vsub
```

```c
// ntt-rvv.cpp: 6 条向量指令
vint64m1_t v_q    = __riscv_vmulh_vv_i64m1(v_V, v_precomp_aux, vl); // V×W_shoup>>64
vint64m1_t v_mult = __riscv_vmul_vv_i64m1(v_V, v_S, vl);            // V×W (低64)
vint64m1_t v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);     // q_est×q
v_V = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);                      // 相减
// + 2 条条件修正 = 共 6 条
```

### 3.2 Shoup 的前提条件

Shoup 的预计算 `W_shoup = floor(W × 2^64 / q)` 必须在**已知 W**（旋转因子/密钥系数）时离线完成。

在 keyswitch 内层循环中：
- `bji` 是密钥多项式（在每次 EvalRotate 前 **固定不变**，可预计算）
- `cji` 是当前密文系数（**每次调用不同**，不可预计算 Shoup 常数）

NTT 中，W（旋转因子）在整个 NTT 过程中固定，V（输入）是变量 → 可用 Shoup。  
KeySwitch 中，bji 固定，cji 是变量 → **可以**用 Shoup（以 bji 预计算 bji_shoup）。

### 3.3 方案对比

| 方案 | 原理 | 指令数（每8元素） | 预计算开销 |
|------|------|-----------------|-----------|
| 标量 Barrett（现状） | 每次完整 128-bit 乘 | 40+（标量指令） | 无 |
| **RVV Barrett（推荐）** | vmulhu+vmul 实现 128-bit | ~14 条向量指令 | 无（mu 已有） |
| RVV Shoup（进阶） | 预计算 bji_shoup，vmulh | ~6 条向量指令 | 需存储 bji_shoup |

---

## 四、RVV Barrett 规约完整实现

### 4.1 前置：VLEN=512，LMUL=1，vl=8（int64）

- `vmulhu.vv`：两个 uint64 向量，取乘积的**高 64 位**（对应 hi）
- `vmul.vv`：取乘积的**低 64 位**（对应 lo）
- `vsrl.vx`：逻辑右移（向量 × 标量）
- `vsll.vx`：逻辑左移
- `vor.vv`：按位 OR（拼接 hi:lo 的移位窗口）
- `vmsgeu.vx`：无符号 ≥ 比较，生成 mask
- `vsub.vx_mu`：条件减法（仅 mask=1 处执行）

### 4.2 Barrett 规约 RVV 代码（14 条指令，处理 8 元素/迭代）

```c
// 入参（均为广播标量，对整个 limb 向量相同）：
//   q    = mv（模数）
//   mu   = mu.m_value（Barrett 常数）
//   n    = modulus.GetMSB() - 2
//   n7   = n + 7
//   n64  = 64 - n
//   n764 = 64 - n - 7

// 每次循环处理 vl 个元素：
for (size_t off = 0; off < N; ) {
    size_t vl = __riscv_vsetvl_e64m1(N - off);

    vuint64m1_t va = __riscv_vle64_v_u64m1(a + off, vl);  // 加载 cji[off:off+vl]
    vuint64m1_t vb = __riscv_vle64_v_u64m1(b + off, vl);  // 加载 bji[off:off+vl]

    // ── Step 1：128-bit 乘 a×b ──────────────────────────────────
    vuint64m1_t hi_ab = __riscv_vmulhu_vv_u64m1(va, vb, vl);   // (a*b) >> 64
    vuint64m1_t lo_ab = __riscv_vmul_vv_u64m1(va, vb, vl);     // (a*b) & mask

    // ── Step 2：x1 = RShiftD(prod, n) = (lo_ab >> n) | (hi_ab << n64) ──
    vuint64m1_t lo_shr = __riscv_vsrl_vx_u64m1(lo_ab, n, vl);  // lo_ab >> n
    vuint64m1_t hi_shl = __riscv_vsll_vx_u64m1(hi_ab, n64, vl);// hi_ab << (64-n)
    vuint64m1_t x1    = __riscv_vor_vv_u64m1(lo_shr, hi_shl, vl);

    // ── Step 3：128-bit 乘 x1×mu ─────────────────────────────────
    vuint64m1_t hi2   = __riscv_vmulhu_vx_u64m1(x1, mu, vl);   // (x1*mu) >> 64
    vuint64m1_t lo2   = __riscv_vmul_vx_u64m1(x1, mu, vl);     // (x1*mu) & mask

    // ── Step 4：q_est = RShiftD(prod2, n+7) = (lo2 >> n7) | (hi2 << n764) ──
    vuint64m1_t lo2_shr= __riscv_vsrl_vx_u64m1(lo2, n7, vl);
    vuint64m1_t hi2_shl= __riscv_vsll_vx_u64m1(hi2, n764, vl);
    vuint64m1_t q_est  = __riscv_vor_vv_u64m1(lo2_shr, hi2_shl, vl);

    // ── Step 5：q_rounded = q_est × q（只取低 64 位） ─────────────
    vuint64m1_t q_rnd  = __riscv_vmul_vx_u64m1(q_est, q, vl);

    // ── Step 6：result = lo_ab - q_rounded（128-bit 差的低 64 位） ──
    vuint64m1_t result = __riscv_vsub_vv_u64m1(lo_ab, q_rnd, vl);

    // ── Step 7：条件修正（若 result ≥ q，减去 q） ─────────────────
    vbool64_t mask = __riscv_vmsgeu_vx_u64m1_b64(result, q, vl); // result >= q?
    result = __riscv_vsub_vx_u64m1_mu(mask, result, result, q, vl);

    // ── 存储结果 ──────────────────────────────────────────────────
    __riscv_vse64_v_u64m1(out + off, result, vl);
    off += vl;
}
```

**总计：每 vl(=8) 个元素，14 条 RVV 指令 + 2 次 vle64 + 1 次 vse64 = 17 条**

### 4.3 Step 6 正确性验证

SubtractD(r, prod) 提取 r.lo，结论如下：

- `a, b < q < 2^50`：a×b < 2^100，因此 hi_ab < 2^36（远小于 2^64）
- q_est = ⌊a*b/q⌋ (Barrett 误差 ≤ 2)：q_est < 2^50
- q_rounded = q_est × q < 2^50 × 2^50 = 2^100
- 对应的 128-bit 表示：hi_{q_rnd} = q_rounded >> 64，lo_{q_rnd} = q_rounded & mask
- **关键**：hi_ab ≈ hi_{q_rnd}（两个 ~100-bit 数的高位相等或差 1）
- 因此 lo_ab - lo_{q_rnd}（64-bit 减法，可能下溢一次，但差值 ∈ [0, 3q)）

**准确结论**：由于 a*b - q_est×q ∈ [0, 3q) ⊂ [0, 2^52)，远小于 2^64，所以该差值的高 128-bit 部分为 0，即 hi_ab = hi_{q_rnd}，64-bit 减法 lo_ab - lo_{q_rnd} **不会下溢**，直接给出正确结果。✓

---

## 五、累加器的 Lazy Reduction 优化

### 5.1 现状：每步都做 ModAdd

当前代码：
```cpp
cTilda0.SetElementAtIndex(i, cTilda0.GetElementAtIndex(i) + cji * bji);
```
PolyImpl::operator+ 调用 NativeVector::ModAddEq，对每个元素做 `add + conditional sub` (3 ops)。

每轮 j：multiply(N ops) → modular_add(N ops) → store。

### 5.2 Lazy Reduction：digits=2 时可以消除中间 ModAdd

`cTilda0` 初始化时全零（keyswitch-hybrid.cpp:453-454 中 `true` 参数意味着 zero-init）。

```
j=0: cTilda0[i] += cji0 * bji0  →  cTilda0[i] ∈ [0, q)
j=1: cTilda0[i] += cji1 * bji1  →  cTilda0[i] ∈ [0, 2q)
```

由于 q < 2^50，2q < 2^51 < 2^64，不会溢出 uint64_t。

**优化方案**：两轮 j 的乘法结果直接累加（无中间 ModReduce），最后一次做条件减法：

```c
// Fused Multiply-Accumulate（无中间 mod reduce）
for (size_t off = 0; off < N; ) {
    size_t vl = __riscv_vsetvl_e64m1(N - off);
    vuint64m1_t acc0 = __riscv_vle64_v_u64m1(ctilda0_i + off, vl); // 初始为 0

    // j=0: acc0 += Barrett(cj0, bj0)
    vuint64m1_t r0 = barrett_mul_rvv(cj0 + off, bj0 + off, vl, q, mu, n, n7);
    acc0 = __riscv_vadd_vv_u64m1(acc0, r0, vl);  // ∈ [0, 2q)

    // j=1: acc0 += Barrett(cj1, bj1)，结果 ∈ [0, 3q)
    vuint64m1_t r1 = barrett_mul_rvv(cj1 + off, bj1 + off, vl, q, mu, n, n7);
    acc0 = __riscv_vadd_vv_u64m1(acc0, r1, vl);  // ∈ [0, 3q)

    // 最终修正：减至 [0, q)（最多需要两次，但实践中 Barrett 精度足够一次）
    vbool64_t m1 = __riscv_vmsgeu_vx_u64m1_b64(acc0, q, vl);
    acc0 = __riscv_vsub_vx_u64m1_mu(m1, acc0, acc0, q, vl);
    // 如果结果仍 ≥ q（当累计两次乘法积超过 q）：
    vbool64_t m2 = __riscv_vmsgeu_vx_u64m1_b64(acc0, q, vl);
    acc0 = __riscv_vsub_vx_u64m1_mu(m2, acc0, acc0, q, vl);

    __riscv_vse64_v_u64m1(ctilda0_i + off, acc0, vl);
    off += vl;
}
```

**节省**：省去 digits-1 次中间 ModAdd（约 N=4096 次 vadd + 条件 vsub）。

---

## 六、内存访问分析

### 6.1 每次 limb 处理的内存

以 sizeQlP=12、digits=2 估算：

| 数组 | 大小（字节） | 说明 |
|------|------------|------|
| cTilda0 / cTilda1（12个limb） | 12×32KB = 384KB | 累加器，写入 |
| cji（digits=2，sizeQlP=12） | 2×12×32KB = 768KB | digit 分解结果，只读 |
| bji / aji（digits=2，sizeQlP=12） | 4×12×32KB = 1.5MB | 密钥多项式，只读 |
| **总计** | **~2.6MB** | L2=2MB（gem5默认）会溢出 |

### 6.2 访问模式对比

**现状（标量）**：对每个 limb 的 N=4096 个 uint64_t 顺序遍历，访问模式 **步长=8 字节**，对硬件预取友好。

**RVV**：vle64/vse64 步长=8字节，完全连续，同样预取友好。不改变访问模式，仅提升计算吞吐。

---

## 七、性能预估

### 7.1 标量当前代价（每元素）

一次 ModMulFastEq（Barrett）的标量操作：

```
MultD(a, b, prod)         → 1× mulq（≈3 cycles on in-order）
RShiftD(prod, n)          → 2× (srl + sll + or) = 3 cycles
MultD(x1, mu, prod2)      → 1× mulq = 3 cycles
RShiftD(prod2, n+7)       → 2× (srl + sll + or) = 3 cycles  
MultD(q_est, q, prod3)    → 1× mulq = 3 cycles
SubtractD + r.lo          → 2-3 cycles
条件修正（branch）         → 1-2 cycles
────────────────────────────
总计：约 15-20 cycles/元素（顺序执行，无流水线重叠）
```

外加 ModAddFastEq（条件 add）约 3-5 cycles/元素。  
合计：约 **20-25 cycles/元素**。

### 7.2 RVV 向量化后（每元素）

VLEN=512，vl=8，每 8 元素：

| 操作 | 指令数 | gem5 延迟（估算） |
|------|-------|----------------|
| Barrett 乘（14条） | 14 | ~20 cycles（流水） |
| vle64 × 2 | 2 | ~6 cycles |
| vsub/vadd/vmsgeu/mask | 4 | ~5 cycles |
| vse64 | 1 | ~3 cycles |
| **总计** | **21条/8元素** | **~35 cycles/8元素 = 4.4 cycles/元素** |

**理论加速比（计算）：20-25 / 4.4 ≈ 4.5-5.7×**

### 7.3 端到端 EvalRotate 影响

| 路径 | 当前 cycles | 修改后预测 cycles | 节省 |
|------|------------|-----------------|------|
| AutomorphismTransform | ~1,600 | ~1,600（已优化） | 0 |
| ModMulNoCheckEq（总） | ~13M × 90% ≈ 11.7M | 11.7M / 5 ≈ 2.3M | **9.4M** |
| 其余（NTT, ModDown 等） | ~1.3M | ~1.3M | 0 |
| **EvalRotate 合计** | **13.46M** | **~5.2M** | **~61% ↓** |

> 注意：以上 90% 假设 KeySwitchCore 中 90% 时间用于多项式乘法。需要插桩实测确认。  
> 保守估计（70% 用于乘法，3× 实际加速）：节省 ≈ 33%，EvalRotate 降至 ~9M cycles。  
> 乐观估计（90% 用于乘法，5× 加速）：节省 ≈ 61%，降至 ~5M cycles。

---

## 八、两种集成方案详细对比

### 方案 A：修改安装目录头文件（推荐用于快速原型）

**修改文件**：
```
openfhe-install-clang18/include/openfhe/core/math/hal/intnat/mubintvecnat.h
```

**优点**：
- 不需要重新编译 OpenFHE（benchmark 编译时直接实例化 inline 函数）
- 修改范围小（只改一个 inline 函数）
- 与 automorph 集成方式相同（已验证可行）

**缺点**：
- inline 函数改动需要**重新编译 benchmark**（会触发 poly-impl.h 重新实例化）
- mubintvecnat.h 中无法访问 keyswitch 层的循环结构，无法做 lazy reduction

**具体修改**（ModMulNoCheckEq，行492）：

```cpp
NativeVectorT& ModMulNoCheckEq(const NativeVectorT& b) {
    size_t size{m_data.size()};
    auto mv{m_modulus};
#ifdef __riscv_vector
    // --- RVV Barrett 批量模乘 ---
    uint64_t q   = (uint64_t)mv.m_value;
    uint64_t mu  = (uint64_t)mv.ComputeMu().m_value;
    int n        = (int)mv.GetMSB() - 2;
    int n7       = n + 7;
    int n64      = 64 - n;
    int n764     = 64 - n - 7;  // 假设 n+7 < 64，即 n < 57

    const uint64_t* pa = reinterpret_cast<const uint64_t*>(m_data.data());
    const uint64_t* pb = reinterpret_cast<const uint64_t*>(b.m_data.data());
    uint64_t*       pc = reinterpret_cast<uint64_t*>(m_data.data());

    for (size_t off = 0; off < size; ) {
        size_t vl = __riscv_vsetvl_e64m1(size - off);

        vuint64m1_t va = __riscv_vle64_v_u64m1(pa + off, vl);
        vuint64m1_t vb = __riscv_vle64_v_u64m1(pb + off, vl);

        vuint64m1_t hi_ab = __riscv_vmulhu_vv_u64m1(va, vb, vl);
        vuint64m1_t lo_ab = __riscv_vmul_vv_u64m1(va, vb, vl);

        vuint64m1_t x1    = __riscv_vor_vv_u64m1(
                                __riscv_vsrl_vx_u64m1(lo_ab, n, vl),
                                __riscv_vsll_vx_u64m1(hi_ab, n64, vl), vl);

        vuint64m1_t hi2   = __riscv_vmulhu_vx_u64m1(x1, mu, vl);
        vuint64m1_t lo2   = __riscv_vmul_vx_u64m1(x1, mu, vl);

        vuint64m1_t q_est = __riscv_vor_vv_u64m1(
                                __riscv_vsrl_vx_u64m1(lo2, n7, vl),
                                __riscv_vsll_vx_u64m1(hi2, n764, vl), vl);

        vuint64m1_t q_rnd = __riscv_vmul_vx_u64m1(q_est, q, vl);
        vuint64m1_t res   = __riscv_vsub_vv_u64m1(lo_ab, q_rnd, vl);

        vbool64_t mask = __riscv_vmsgeu_vx_u64m1_b64(res, q, vl);
        res = __riscv_vsub_vx_u64m1_mu(mask, res, res, q, vl);

        __riscv_vse64_v_u64m1(pc + off, res, vl);
        off += vl;
    }
#elif defined(NATIVEINT_BARRET_MOD)
    auto mu{m_modulus.ComputeMu()};
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv, mu);
#else
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv);
#endif
    return *this;
}
```

**重新编译命令**：

```bash
cd ~/NTT-RVV-project/benchmark
make clean && make small
```

**验证命令**：

```bash
~/gem5.opt ~/NTT-RVV-project/gem5-model/main.py \
    ~/NTT-RVV-project/benchmark/bin/small/bench_rotate
```

**风险点**：
- `n764 = 64 - n - 7 = 64 - 48 - 7 = 9`（50-bit 素数时），是合法移位量 ✓
- 若 benchmark 参数导致某些素数 > 57 bit（即 n > 57，n+7 > 64），第二个 RShiftD 会 UB。需要运行前断言 `assert(n < 57);` 或添加安全检查。
- 需要包含 `<riscv_vector.h>` 且条件编译对 non-RISC-V 路径不引入 RVV 头文件。

---

### 方案 B：修改 keyswitch-hybrid.cpp（推荐用于最终优化）

**修改文件**：
```
openfhe-development/src/pke/lib/keyswitch/keyswitch-hybrid.cpp
```

**优点**：
- 可以重写整个双重循环，在 limb 级别做 lazy reduction
- 可以 fuse multiply+accumulate（单次 load，两次 mul，一次 store）
- 更精细控制（例如：先加载所有 cji，在寄存器中 fuse）

**缺点**：
- 需要重新编译 OpenFHE（约 20-30 分钟）：
  ```bash
  cd ~/NTT-RVV-project/openfhe-development/build
  cmake --build . --target OPENFHEpke -j$(nproc)
  cmake --install .
  ```
- 需要在 C++ 模板代码中嵌入 C intrinsics（比较 hacky）

**修改策略**：在 EvalFastKeySwitchCoreExt 内层 poly 乘法处，绕过 PolyImpl::operator*，直接调用自定义 RVV 函数：

```cpp
// 头部添加：
#include "keyswitch_mul_rvv.h"  // 新建的 RVV 乘累加函数

// 内层循环改写：
for (usint i = 0; i < sizeQlP; i++) {
    // 获取底层指针：cji, bji, acc 都是 NativeVector<uint64_t>
    const uint64_t* pcj = /* cj limb i 的原始指针 */;
    const uint64_t* pbj = /* bj limb i 的原始指针 */;
    uint64_t*       pacc = /* cTilda0 limb i 的原始指针 */;
    uint64_t q = /* limb i 的模数 */;
    uint64_t mu = /* limb i 的 Barrett 常数 */;

    barrett_muladd_rvv(pcj, pbj, pacc, N, q, mu, n);
}
```

---

## 九、难度评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 算法正确性 | 中 | Barrett 规约有详细标量参考，RVV 翻译直接 |
| RVV 实现难度 | 中 | 14 条指令，不涉及 gather/scatter，无 gem5 已知 bug |
| 正确性验证 | 低 | 用标量版本逐元素对比即可，不涉及随机性 |
| 集成风险（方案A） | 低 | inline 函数，改 installed header，与 automorph 方案相同 |
| 集成风险（方案B） | 中 | 需要重编 OpenFHE，需要拿到 NativeVector 底层指针 |
| 边界条件 | 中 | n+7 必须 < 64（验证可以加断言） |
| **总体难度** | **中** | 主要工作在实现和验证 Barrett RVV，集成路径清晰 |

---

## 十、实施步骤清单

### 阶段一：准备与单元测试（1-2天）

- [ ] **10.1** 写一个 standalone C 文件 `test_barrett_rvv.c`，包含：
  - RVV Barrett 函数（见上文 §4.2）
  - 标量参考实现（逐元素调用 `__uint128_t` 乘法 + 取模）
  - 随机测试：生成 N=4096 对随机数 (a, b)，两版本结果对比
  - 不同 q（50-bit / 55-bit）的参数测试

- [ ] **10.2** 编译测试：
  ```bash
  clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
        -O2 -o test_barrett test_barrett_rvv.c
  ~/gem5.opt ~/NTT-RVV-project/gem5-model/main.py ./test_barrett
  ```

- [ ] **10.3** 单算子 benchmark：测量 N=4096 的单次 ModMulNoCheckEq 延迟

### 阶段二：集成（方案A，1天）

- [ ] **10.4** 备份 mubintvecnat.h：
  ```bash
  cp openfhe-install-clang18/include/openfhe/core/math/hal/intnat/mubintvecnat.h \
     mubintvecnat.h.bak
  ```

- [ ] **10.5** 在 mubintvecnat.h 的 ModMulNoCheckEq 添加 RVV 分支（见 §8 方案A）

- [ ] **10.6** 添加 RVV 头文件引用（检查文件顶部是否已有 `#include <riscv_vector.h>`）

- [ ] **10.7** 重新编译 benchmark 并运行：
  ```bash
  cd ~/NTT-RVV-project/benchmark && make clean && make small
  ~/gem5.opt ~/NTT-RVV-project/gem5-model/main.py \
      ~/NTT-RVV-project/benchmark/bin/small/bench_rotate
  ```

- [ ] **10.8** 记录结果，对比集成前 13,458,398 cycles

### 阶段三：验证正确性（0.5天）

- [ ] **10.9** 确认 bench_rotate 输出的 ciphertext 正确性（解密后与已知明文对比）
  - bench_rotate 有结果验证吗？如果没有，需要添加 decrypt+check。

- [ ] **10.10** 运行 full 配置（RING_DIM=16384）验证

---

## 十一、一句话总结

密钥乘法向量化的核心是用 **vmulhu + vmul 实现 128-bit 乘**，完整实现 Barrett 规约需要 **14 条 RVV 向量指令**处理 8 个元素（vs 标量 ~20 cycles/元素），理论加速比 **4-6×**，对应 EvalRotate 端到端加速 **30-60%**（取决于密钥乘法占比的实测值）。集成方案 A（改 mubintvecnat.h）可以在**不重编 OpenFHE** 的情况下完成。
