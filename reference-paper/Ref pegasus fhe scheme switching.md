# 文献参考：PEGASUS: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption

## 元信息

| 字段 | 内容 |
|------|------|
| **标题** | PEGASUS: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption |
| **作者** | Wen-jie Lu, Zhicong Huang, Cheng Hong, Yiping Ma, Hunter Qu |
| **机构** | Gemini Lab, Alibaba Group（Lu/Huang/Hong/Qu）；University of Pennsylvania（Ma） |
| **发表** | 2021 IEEE Symposium on Security and Privacy (S&P), pp. 1057–1069 |
| **DOI** | 10.1109/SP40001.2021.00043 |
| **代码** | https://github.com/Alibaba-Gemini-Lab/OpenPEGASUS |
| **关键词** | Homomorphic Computation, Floating Point Computation, CKKS/FHEW, Scheme Switching |

---

## 一、研究背景与动机

### 核心问题

现有 HE 方案的根本矛盾：

| 方案类型 | 代表 | 优势 | 劣势 |
|---------|------|------|------|
| **Word-wise（字级）** | BFV, BGV, CKKS | SIMD 批处理，高效线性运算（加法/乘法/旋转） | sigmoid、min/max、division 等非多项式函数难以计算 |
| **Bit-wise（位级）** | FHEW, TFHE | 支持任意布尔电路，灵活非线性运算 | 加法/乘法极慢，扩展比大，通信开销高 |

**典型应用需求**（如神经网络推理）同时包含：卷积/全连接层（线性，适合 CKKS）+ sigmoid/ReLU/max-pooling（非线性，需要 LUT）

### 现有方法的不足（Table II）

| 方案 | SIMD支持 | 灵活性 | 密钥大小 | 效率 |
|------|---------|-------|---------|------|
| [16] Cheon et al. | 是 | 极低（仅 min/max） | ≈4GB | 高（摊销意义） |
| [43] Micciancio et al. | 否 | 限于小域 LUT | >5GB | 高 |
| [7] CHIMERA | 是 | 大域 LUT | ≫10GB | 低 |
| **PEGASUS（本文）** | **是** | **大域 LUT** | **≈1GB** | **高** |

**CHIMERA 的主要问题**：
- 将 CKKS/BFV 导出到 Torus 端需 Multi-Precision FFT（MP-FFT），比 NTT 慢数个数量级
- repacking key 高达 80GB（参数 n̄=2^16, n=2^10, |g|=7）
- repacking 计算复杂度高，受限于 ℓ=Ω(log N)

### PEGASUS 的核心贡献

1. **放宽 LUT 输入约束**：接受更大输入域（>40 bits），引入有界近似误差
2. **高效 repacking 算法**：key 大小从 80GB 降至 **12MB**，计算复杂度从线性降为次线性
3. **NTT/RNS 友好**：不导出到 Torus，保留 CKKS 的高效多项式运算
4. **全面应用验证**：sigmoid/ReLU/sqrt/min/max/sorting/max-pooling + 私有决策树 + 安全 K-means

---

## 二、数学基础与符号约定

### 符号系统

| 符号 | 含义 |
|------|------|
| R_n | Z[X]/(X^n+1)，n 为2的幂 |
| R_{n,q} | Z_q[X]/(X^n+1) |
| n̄, n, n̲ | 不同（R）LWE 维度，大→小（n̲ < n < n̄） |
| s̄, s, s̲ | 对应维度的密钥 |
| LWE^{n,q}_s(m) | 维度 n、模 q、密钥 s 下的 LWE 加密 |
| RLWE^{n̄,q}_s(m̂) | 维度 n̄、模 q 下的 RLWE 加密 |
| Ecd(v, Δ) | 向量 v 以缩放因子 Δ 编码为 R_{n,q} 元素 |
| Dcd(v̂, Δ, ℓ) | 以因子 Δ 和长度 ℓ 解码 |
| RotL^k(ct; RotK) | 对密文 ct 执行左旋转 k 位 |

### 关键 RLWE 操作

- **Addition（+）**：ct₀ + ct₁ → 加密 p̂₀ + p̂₁
- **Small Plaintext Multiplication（·）**：r̂ · ct → 加密 r̂ · p̂（r̂ 低范数）
- **Arbitrary Plaintext Multiplication（◇）**：扩展加密 RLWẼ(·; g) 支持任意环元素乘法
- **External Multiplication（⊙）**：RGSW 密文支持，ct ⊙ ct̃ → 加密 p̂₀ · p̂₁
- **Rotation**：RotL^k(ct; RotK) → 加密左旋转向量 v ≪ k
- **Rescale**：Rescale(ct, Δ') → 减小模数同时保持编码稳定
- **Slots To Coefficients（S2C）**：将编码向量的 v[i] 展开为多项式系数
- **Coefficients Extraction（Extract^k）**：提取第 k 个系数得 LWE 密文

---

## 三、PEGASUS 的四个核心函数（Figure 1）

### F_KS：密钥切换（Key Switching）

```
输入：ct_in ∈ LWE^{n̄,q}_s(m)
输出：ct_out ∈ LWE^{n̲,q}_s(m)，其中 n̲ ≤ n̄
```
将 LWE 密文从大维度 n̄ 切换到小维度 n̲，为后续 LUT 计算降低维度。

### F_LUT：查找表评估（Look-up Table Evaluation）

```
输入：ct_in ∈ LWE^{n̲,q}_s(⌈Δm⌉)，查找表函数 T(x): R → R
输出：ct_out ∈ LWE^{n,q}_s(⌈ΔT(m)⌉) + 小近似误差 e
```
核心是在 FHEW-like LWE 上评估任意函数，实现非多项式计算。

### F_LT：线性变换（Linear Transform）

```
输入：ct_in ∈ RLWE^{n̄,q}_s(Ecd(z, Δ))，矩阵 M ∈ R^{ℓ×n̲}，向量 t ∈ R^ℓ
      满足 ℓ, n̲ < n̄/2
输出：ct_out ∈ RLWE^{n̄,q'}_s(Ecd(Mz + t, Δ))
```
**这是 Repacking 的核心操作**，即密文矩阵向量乘法。通过 BSGS 算法实现，是本文对后续工作（Chameleon、RVV MatVec 加速）最重要的贡献之一。

### F_mod：近似模运算（Approximated Modulo）

```
输入：ct_in ∈ RLWE^{n̄,q'}_s(ẑ + qê)，ê 小范数多项式
输出：ct_out ∈ RLWE^{n̄,q''}_s(ẑ + ê') + 小近似误差
```
将模 q 操作同态近似，是连接 LWE 和 RLWE 的关键一步。

---

## 四、PEGASUS 的三个关键技术

### 技术 1：大域 LUT 评估（PEGASUS.LUT，Figure 2）

**问题根源**：传统 Ducas-like LUT 方法要求 eq = 2n，通常取 q=2^9，限制了输入域大小（仅适合 10–12 bit 整数）。CKKS 应用通常使用 Δ > 2^30，需要更大的输入域。

**PEGASUS 的解法：近似 LWE 解密**

传统 LWE 解密：`b + a^T s mod q`，但在 RLWE 方案中只能模 n 进行

创新：将模 q 的解密**下采样**为模 n 的近似解密：
```
b̃ = ⌊(ẽn/q) b⌋，ã = ⌊(ẽn/q) a⌋（ẽ = 2）
近似解密 ≈ b̃ + ã^T s mod n ≈ ⌈ẽnm/q⌉
```

这将有效输入范围扩展到 [-q/2ẽ, q/2ẽ)，对 CKKS 典型参数 q ≈ 2^45 可支持约 40+ bit 输入精度。

**算法（PEGASUS.LUT，Figure 2）**：
```
输入：LWE 密文 (b,a)，评估密钥 EK（RGSW 形式），查找表 T(x)
EK_{j,0} = RGSW(I+(s[j]))，EK_{j,1} = RGSW(I-(s[j]))

1. 计算 η_k = kq/(2nΔ) for 1 ≤ k ≤ n/2
   构造多项式 f̂，系数为：
     f_j = ⌈ΔT(0)⌉         (j=0)
     f_j = ⌈ΔT(-η_j)⌉      (1 ≤ j ≤ n/2)
     f_j = ⌈-ΔT(η_{n-j})⌉  (n/2 < j < n)

2. b̃ = ⌊(2n/q)b⌋，ã = ⌊(2n/q)a⌋

3. 初始化 AC_0 = (f̂ · X^{b̃ mod n}, 0) ∈ R^2_{n,q}

4. 循环 j = 0,...,n̲-1：
   t_j = ((X^{ã[j] mod n} - 1) · AC_j) ⊙ EK_{j,0}
   AC_{j+1} = ((X^{-ã[j] mod n} - 1) · t_j) ⊙ EK_{j,1} + t_j

5. 输出 Extract^0(AC_{n̲}) 作为 ct_out
```

**误差分析**：近似误差 ≤ Lq/(2ẽn)，L 是 T(·) 的 Lipschitz 常数。

**密钥大小**：EK 大小 O(|g|n̲n log q) bits，参数 n̄=2^16, n̲=2^10, q≈2^45 时约 **420 MB**。

### 技术 2：高效 Repacking（PEGASUS.LT）

**目标**：将一批 LWE 密文 {ct_i ∈ LWE^{n̲,q}(m_i)} 打包为单个 RLWE 密文 Ecd(T(v), Δ)

**两步分解**：
1. **F_LT**：同态地部分解密 LWE，得 RLWE^{n̄}(Ecd(As+b, Δ))
2. **F_mod**：同态地对 q 取模，将结果规整

**F_LT 的 PEGASUS.LT 算法（Figure 5）**：

关键洞察：ℓ 个 LWE 密文的部分解密 = 矩阵向量乘法 `As + b`，其中：
- A ∈ Z^{ℓ×n̲}（每行来自一个 LWE 密文）
- s = LWE 密钥的 RLWE 加密（Repacking Key RK）

**算法流程**：
```
输入：ct_in ∈ RLWE（Ecd(z, Δ_r)），旋转密钥 RotK，
      明文矩阵 M ∈ R^{ℓ×n̲}，向量 t ∈ R^ℓ，其中 ℓ, n̲ < n̄/2

1. Tiling and Diagonals：
   ñ = max(ℓ, n̲)，ñ = min(ℓ, n̲)
   构造 ñ 个对角向量 {m̃_j}^{ñ-1}_{j=0}，按行列遍历 M
   m̃_j[r] = M[r mod ℓ, r + j mod n̲]，r ∈ 〈ñ〉

2. Baby-Step：g̃ = ⌈√ñ⌉，对 g ∈ 〈g̃〉
   c_g = RotL^g(ct_in)

3. Giant-Step：b̃ = ⌈ñ/g̃⌉
   ct̃ = Σ_{b∈〈b̃〉} RotL^{bg̃}(Σ_{g∈〈g̃〉} Ecd(m̃_{bg̃+g} ≫ bg̃, Δ'_r) · c_g)

4. 若 ℓ ≥ n̲：
   输出 Rescale(ct̃, Δ_r) + Ecd(t, Δ'_r) 作为 ct_out

5. 否则（ℓ < n̲，"Sum Columns" 步骤）：
   γ = log(n̲/ℓ)，ct_0 = ct̃
   迭代 1 ≤ j ≤ γ：ct_j = RotL^{ℓ·2^j}(ct_{j-1}) + ct_{j-1}
   输出 Rescale(ct_γ, Δ_r) + Ecd(t, Δ'_r) 作为 ct_out
```

**旋转次数复杂度对比（Table III）**：

| 方法 | ℓ < n̲ 的旋转次数 | 深度 |
|------|----------------|------|
| [30] Halevi-Shoup | O(√n̄) | 1 |
| [38] Juvekar et al. | O(ℓn̄ + log(n̄/n̲)) | 1 |
| [10] Chen et al. | O(ε + ⌈ℓn̲/(εn̄)⌉ · log(n̲/ε)) | 2 |
| **PEGASUS（ℓ < n̲）** | **O(√ℓ + log(n̲/ℓ))** | **1** |
| [30] Halevi-Shoup（ℓ ≥ n̲） | O(√ℓ) | 1 |
| **PEGASUS（ℓ ≥ n̲）** | **O(√n̲)** | **1** |

**关键性质**：PEGASUS.LT 的复杂度与 repacking 大小 ℓ 无关（当 ℓ ≥ n̲ 时），可 repack 任意数量 LWE 密文而不引入额外开销。

**Repacking Key RK 大小**：O(2n̄ log q) bits，约 **12MB**（vs CHIMERA 的 80GB）。

### 技术 3：FHEW→CKKS 转换（全流程）

三步转换：
```
ct（RLWE）→ {ct_i = Extract^i(S2C(ct))} → {ct'_i = LUT(T, ct_i)} → ct''（RLWE，打包结果）
      S2C + Extract（得 LWE）         F_LUT（各 LWE 独立）     Repacking（F_LT + F_mod）
```

---

## 五、PEGASUS 完整协议（Figure 6）

### 公开参数
- 密文模数 q_0, q_1, ..., q_{L-1} 和特殊模数 q'，Q_i = Π_{l∈〈i〉} q_l
- 数字分解 gadget 向量 g_digit = [1, B_ks, ..., B_ks^{d_ks}]
- RNS 分解 gadget 向量 g_rns
- 缩放因子 0 < Δ, Δ_r, Δ'_r < q_0

### 公开密钥
- **SwK_{s̄→s}**：切换密钥（RLWE 向量）
- **SwK_{s→s̲}**：维度下降切换密钥
- **EK**：LUT 评估密钥（RGSW 格式，对 g_rns），大小约 420MB
- **RK**：Repacking 密钥（单个 RLWE 密文，大小约 12MB）
- **RotK**：CKKS 旋转密钥

### 完整 10 步流程

```
输入：level-l RLWE 密文（l>1）Ecd(v, Δ)，查找表 T(x)
输出：RLWE 密文 Ecd(T(v), Δ)

步骤1：ct' = S2C(ct_in)
       ▷ ct' ∈ RLWE^{n̄,q_0}(Δv̂)
步骤2：ct_i = Extract^i(ct') for each i ∈ 〈ℓ〉
       ▷ ct_i ∈ LWE^{n̄,q_0}(Δv[i])，并行
步骤3（for i ∈ 〈ℓ〉，并行）：
步骤4：  ĉt_i = PEGASUS.KS(ct_i, SwK_{s̄→s̲})
         ▷ ĉt_i ∈ LWE^{n̲,q_0}(Δv[i])
步骤5：  c̈t_i = PEGASUS.LUT(ĉt_i, EK, T(x))
         ▷ c̈t_i ∈ LWE^{n̲,q_0}(ΔT(v[i]))
步骤6：  c̈t_i = (b_i, a_i) = PEGASUS.KS(c̈t_i, SwK_{s→s̲})
         ▷ c̈t_i ∈ LWE^{n̲,q_0}(ΔT(v[i]))
步骤7（end for）

步骤8：定义 b = [b_0,...,b_{ℓ-1}] 和 A ∈ Z^{ℓ×n̲}（第 i 行为 a_i）
步骤9：ct̃ = PEGASUS.LT(RK, RotK, Δ'_r, A, b)
       ▷ ct̃ ∈ RLWE^{n̄,Q_{L-1}}(Ecd(As+b, Δ'_r))
步骤10：对 ct̃ 通过 F_mod 执行模 q_0 操作并输出 ct_out
        ▷ ct_out ∈ RLWE^{n̄,Q_{L'}}(Ecd(T(v), Δ))
```

**注意**：步骤 4–6 在维度 LWE 格式下操作（固定模 q_0），利用最少模数降低计算量。步骤 3–6 完全独立，可多核并行。

---

## 六、高级特性

### Multiple LUTs
可在步骤 5 和 6 之间插入多次 LUT 序列，以及在 LWE 密文上进行加减法。min-index 函数即利用此特性。

### Customizable Encoding Layout
步骤 9 中可通过重排 A 和 b 的行顺序改变编码布局，适配不同 HE 应用的混合编码需求（如 [38] 的卷积+全连接混合布局）。

### Tunable Output Level
步骤 10 的 F_mod 输出模数 L' 可调节，在保证 Chebyshev 近似深度的前提下选最小值（例如卷积网络推理中 L'=3 已足够）。

---

## 七、具体参数（Table IV）

| 加密类型 | 参数 |
|---------|------|
| RLWE^{n̲,q_0}(·) | n̲ = 2^10, q_0 ≈ 2^45, σ_ks = 2^13, B_ks = 2^7, d_ks = 7 |
| RGSW^{n,q'q_0}(·) | n = 2^12, q' ≈ 2^60, σ_lut = 2^10 |
| RLWE^{n̄,·}(·) | n̄ = 2^16, q_i ≈ 2^45, σ_ckks = 3.19, log Q̄ = 795 |

| 密钥类型 | 大小 |
|---------|------|
| SwK_{s̄→s} | 2d_ks n̄ log q_0 bits ≈ **5.0 MB** |
| SwK_{s→s̲} | 2d_ks n log q_0 bits ≈ **315 KB** |
| EK | 8n̲n log(q'q_0) bits ≈ **420 MB** |
| RK | 2n̄ log(Π_{i∈〈L〉} q_i) bits ≈ **12 MB** |

---

## 八、实验配置

### 硬件平台
- **CPU**：Intel Xeon Platinum 8269CY，20核，2.50 GHz
- **编译**：gcc 7.5.0
- **所有实验**：运行在上述服务器上

### 软件实现
- 基于 **SEAL 库** [47]，并集成额外优化：
  - 更快 NTT [46]
  - 更快 (◇) 操作（lazy-reduction）
  - S2C 和 F_mod 实现
- L = 16 个 RNS 素数用于密文模数
- q_0 ≈ 2^45，特殊模数 q' ≈ 2^60
- 秘密密钥分布：hamming weight = 64 的三元多项式（系数 ∈ {0, ±1}）
- 每个实验 L' = 6 模数（repacking 后密文剩余 6 个模数层）

---

## 九、实验结果

### 结果 1：微基准测试（Table V，单线程）

| 操作 | [19] CHIMERA | PEGASUS | 加速比 |
|------|-------------|---------|-------|
| KS(s̄→s) | 4192ms | 20.14ms | **208×** |
| LUT | 60s | 0.93s | **64×** |
| KS(s→s̲) | 260ms | 1.49ms | **174×** |
| S2C（log ℓ=8,10,12） | 0.78s, 1.28s, 2.02s | — | — |
| LT（log ℓ=8,10,12） | 16.76s, 44.50s, 44.65s | — | — |
| F_mod [33] | — | 7.06s | — |

**KS 加速主要来自**：PEGASUS 不使用 MP-FFT，直接用 NTT/RNS

### 结果 2：非多项式函数吞吐（Table VI，多线程）

| 函数类型 | 1线程 | 4线程 | 8线程 | 16线程 | 20线程 |
|---------|------|------|------|-------|-------|
| sigmoid/ReLU/sqrt/reciprocal（吞吐 LUT/s） | 1.06/s | 3.95/s | 7.95/s | 15.34/s | 20.77/s |
| Max-Pooling 2×2（吞吐/s） | 0.25/s | 0.91/s | 1.97/s | 3.90/s | 4.83/s |
| Max-Pooling 4×4（吞吐/s） | 0.07/s | 0.26/s | 0.50/s | 0.98/s | 1.21/s |

| 函数（延迟） | t=2^8 | t=2^7 | t=2^4 | t=2^6（Sort）| t=2^5（Sort） |
|------------|------|------|------|------------|------------|
| Min-Index（1线程） | 791.84s | 395.39s | 43.24s | — | — |
| Min-Index（20线程） | 44.73s | 25.26s | 5.17s | — | — |
| Sort（1线程） | — | — | — | 1380.50s | 493.30s |
| Sort（20线程） | — | — | — | 84.03s | 34.10s |

**对比 CHIMERA**：[16] 计算 t=2^4 的 max-index 需约 236s（24×慢）；[25] 计算同等 sorting 需 43min（6.3×慢）

### 结果 3：私有决策树评估（Table VII）

数据集：UCI Iris/Housing/Spambase

| 数据集 | N | d | C | [37]计算 | [51]计算 | PEGASUS计算 | [37]通信 | [51]通信 | PEGASUS通信 | PEGASUS准确率 |
|-------|---|---|---|---------|---------|-----------|---------|---------|-----------|------------|
| Iris | ≈10 | 4 | 3 | 0.59s | 0.94s | **1.87s** | 1.65MB | 1.19MB | **16.89KB** | 94.74% |
| Housing | ≈100 | 13 | 2 | 10.27s | 6.30s | **10.71s** | 13.12MB | 6.63MB | **16.89KB** | 98.04% |
| Spambase | ≈60 | 57 | 2 | 6.88s | 3.66s | **6.75s** | 11.54MB | 7.36MB | **16.89KB** | 85.03% |

**PEGASUS 优势**：通信量约为其他方法的 **1%**（只发送单个 RLWE 密文），计算速度与现有最优方法相当。

### 结果 4：安全 K-means 聚类（Table VIII，N×K 数据点，d=16，20线程）

参数设置：n=2^12，n̄=2^16，调优后每轮更新 O(2K) 个 LUT（vs 原始 O(3K)）

**性能分解（以 n=2^12 为例）**：

| N | K | Distance | Extract | Min-Index & Recip | Repacking | Centroid | Total | vs [34] 加速 |
|---|---|---------|---------|-----------------|-----------|---------|-------|------------|
| 256 | 2 | 1.56s | 1.83s | 51.53s | 25.93s | 0.35s | 19.81min | 14× |
| 256 | 4 | 2.69s | 2.60s | 107.78s | 25.93s | 0.50s | 39.61min | 17× |
| 256 | 8 | 3.11s | 4.36s | 211.31s | 25.93s | 0.87s | 79.23min | 19× |
| 1024 | 2 | 1.58s | 5.33s | 109.27s | 55.22s | 0.33s | 79.23min | 21× |
| 1024 | 4 | 2.79s | 8.95s | 387.08s | 55.22s | 0.45s | 158.45min | 21× |
| 4096 | 2 | 1.61s | 17.78s | 756.33s | 60.69s | 0.32s | 316.89min | 22× |
| 4096 | 8 | 3.03s | 57.04s | 3000.85s | 60.69s | 1.04s | 1267.58min | 24× |

**关键观察**：
- Min-Index 占约 **95%** 的总计算时间（O(2KN) 个 LUT）
- Repacking 时间随 N 增大而增长（O(√N) when N ≤ n̲，O(60s) when N > n̲）
- 比现有最优 TFHE-based 方案 [34] 快 **14–24×**

---

## 十、误差分析

- **Key Switching 误差**：量级约 2^{-14}（小）
- **LUT 近似误差**：在 Table IV 参数下约 [2^{-10}, 2^{-7}]
- **LUT 精度与格维度 n 的关系**（Figure 7）：LUT 精度 = -log|f(x) - T_f(x)|，随 log₂(n) 线性增长（约 8–15 bits 精度，对应 n = 2^12–2^15）
- 对 ML 应用（机器学习、信息检索）而言误差可接受

---

## 十一、对 RVV 密文矩阵向量乘法（MatVec）工作的直接参考价值

### PEGASUS.LT 是 MatVec 加速的直接目标

PEGASUS.LT（Figure 5）是本工作针对 RVV 加速的**最直接算法参考**：

**算法结构映射到 RVV**：

```
Baby Step（步骤2）：c_g = RotL^g(ct_in)，g ∈ 〈g̃〉，共 g̃ = ⌈√ñ⌉ 次旋转
  - 每次旋转 = NTT → 逐元素乘法 → INTT（可用 TCHES RVV NTT 加速）

Giant Step（步骤3内层）：Ecd(m̃_{bg̃+g} ≫ bg̃, Δ'_r) · c_g
  - 明密文乘法（PMUL）= NTT域逐元素乘法 → 直接 RVV Shoup Reduction

Giant Step（步骤3外层）：RotL^{bg̃}(Σ ...) 
  - 旋转 = NTT → 乘法 → INTT

累加（Σ）：向量加法 → RVV vadd
```

**关键参数（与 RVV 实现相关）**：
- n̄ = 2^16：多项式长度，每个 NTT 长 65536 个 64-bit 元素
- ñ = min(ℓ, n̲) = min(ℓ, 1024)：对角向量数量
- g̃ = ⌈√ñ⌉：Baby step 数量（决定旋转次数）
- b̃ = ⌈ñ/g̃⌉：Giant step 数量

**内存访问分析**：
- Repacking Key RK：单个 RLWE 密文，2 × n̄ × 8 bytes = 2 × 65536 × 8 ≈ **1 MB**（L2 友好！）
- 旋转密文（Baby step 预计算）：g̃ × 2 × n̄ × 8 bytes = 32 × 2 × 65536 × 8 ≈ **32 MB**（需 L3）
- 对角向量：ñ × n̄ × 8 bytes = 1024 × 65536 × 8 ≈ **512 MB**（主存，需流式加载）

### PEGASUS 是 Chameleon 和后续工作的基准

- Chameleon 的方案切换性能以 **CPU Pegasus 为基准**（67.3× 加速）
- Chameleon 的 Repacking 优化（HRF-MatVec）直接改进了 PEGASUS.LT
- 理解 PEGASUS.LT 的算法结构对设计 RVV 版本 MatVec 至关重要

### 可直接参考的数值

| 指标 | PEGASUS 数值 | RVV 实现参考意义 |
|------|------------|----------------|
| LT 执行时间（log ℓ=8,10,12，单线程） | 16.76s, 44.50s, 44.65s | RVV 加速目标（~5×预期→3–9s） |
| Repacking 时间（N=256,K=4） | 25.93s | 完整应用中 MatVec 占比 |
| 旋转次数 | O(√ℓ + log(n̲/ℓ)) | 决定 NTT 调用次数 |
| RK 大小 | 12 MB | L3 Cache 可容纳 |

### 与 TCHES NTT-RVV 论文的衔接

| TCHES 提供 | PEGASUS 需要 | 融合点 |
|-----------|-------------|-------|
| KL NTT（5–7× 加速） | PEGASUS.LT 内每次旋转含 NTT | 旋转 = NTT + 逐元素乘 + INTT，全程 RVV 化 |
| Shoup Reduction RVV | PEGASUS.LT 的明密文乘法 | Baby/Giant step 的 PMUL 直接用 |
| OpenFHE 集成框架 | PEGASUS 基于 SEAL | 注入点：`EvalLinTransform`（OpenFHE 等价函数） |

---

## 十二、符号速查

| 符号 | 含义 |
|------|------|
| n̄ | 最大（R）LWE 维度（CKKS 侧，n̄=2^16） |
| n | 中间 LWE 维度（FHEW 侧，n=2^12） |
| n̲ | 最小 LWE 维度（切换后，n̲=2^10） |
| ℓ | repacking 大小（要打包的 LWE 密文数） |
| g̃ | Baby step 大小（≈ √min(ℓ,n̲)） |
| b̃ | Giant step 大小（≈ min(ℓ,n̲)/g̃） |
| ñ | min(ℓ, n̲），对角向量数 |
| EK | LUT 评估密钥（RGSW 格式，约 420MB） |
| RK | Repacking 密钥（单 RLWE 密文，约 12MB） |
| SwK | 密钥切换密钥 |
| RotK | 旋转密钥（CKKS） |
| S2C | Slots to Coefficients（槽→系数变换） |
| F_LT | 线性变换函数（= PEGASUS.LT） |
| F_LUT | 查找表评估函数（= PEGASUS.LUT） |
| F_mod | 近似模运算函数 |
| g_digit | 数字分解 gadget 向量（用于 SwK） |
| g_rns | RNS 分解 gadget 向量（用于 EK） |
| BSGS | Baby-Step Giant-Step 算法 |
| ρ | K-means 中最接近质心分配比例 |