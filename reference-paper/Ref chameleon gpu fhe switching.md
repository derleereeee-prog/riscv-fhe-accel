# 文献参考：Chameleon: An Efficient FHE Scheme Switching Acceleration on GPUs

## 元信息

| 字段 | 内容 |
|------|------|
| **标题** | Chameleon: An Efficient FHE Scheme Switching Acceleration on GPUs |
| **作者** | Zhiwei Wang, Haoqi He, Lutan Zhao, Peinan Li, Zhihao Li, Dan Meng, Rui Hou |
| **机构** | 中国科学院信息工程研究所（IIE, CAS）/ 中国科学院大学；蚂蚁集团（Zhihao Li） |
| **发表** | IEEE Transactions on Parallel and Distributed Systems, Vol. 36, No. 11, pp. 2264–2280, November 2025 |
| **DOI** | 10.1109/TPDS.2025.3604866 |
| **收稿/接收** | 2024-10-07 / 2025-08-28 |
| **关键词** | FHE, scheme switching, GPU acceleration, NTT, CKKS, TFHE |
| **基金** | 国家重点研发计划 2023YFB4503200；国家自然科学基金杰青 62125208；中科院战略先导专项 XDB0690100 |

---

## 一、研究背景与动机

### 单一 FHE 方案的局限

| 方案类型 | 代表 | 优势 | 劣势 |
|---------|------|------|------|
| **Word-wise（字级）** | BGV, BFV, CKKS | 高效 SIMD 线性运算（矩阵乘、旋转） | 无法高效执行非线性操作 |
| **Bit-wise（位级）** | TFHE, FHEW | 支持非线性布尔门、每次操作即自举 | 线性运算效率低，无批处理 |

**现实需求**：加密数据库查询、私有推理等应用同时需要线性运算和非线性运算，单一方案无法满足。

### 方案切换（Scheme Switching）

将 CKKS 和 TFHE 结合，利用各自优势：

```
CKKS（线性运算：MatMul, 加密神经网络的线性层）
    ↓ CKKS → TFHE（Slot2Coeff + Sample Extract + Key Switch）
TFHE（非线性运算：LUT, 激活函数, 比较, 排序）
    ↓ TFHE → CKKS（Functional Bootstrapping + Repacking）
CKKS（继续线性运算）
```

### 性能瓶颈分析（基于 Pegasus CPU 实现，Figure 2）

参数：nckks=2^16, nlwe=2^10, nlut=2^12, ciphertext modulus=599 bits

| 操作 | 占比（256 slots） | 占比（1024 slots） | 特点 |
|------|----------------|-----------------|------|
| **LUT evaluation（TFHE功能自举）** | 74.1% | 88.8% | 随槽数增大急剧上升，并行潜力大 |
| **Repacking（TFHE→CKKS转换）** | ~20% | ~10% | 槽数少时超过LUT成为主瓶颈，不受CPU线程数影响 |
| Slot2Coeff | 少量 | 少量 | 旋转次数少 |

**结论**：LUT evaluation + Repacking 合计约占 **96%** 总时间，是优化重点。

---

## 二、CKKS 和 TFHE 方案回顾

### CKKS 方案

**密文结构**：RLWE 密文 `(B(x), A(x)) ∈ R²_Q`，其中 `B(x) = A(x)·S(x) + Δ·P(x) + E(x)`
- nckks：多项式次数（2的幂次）
- nslot ≤ nckks/2：可打包的明文槽数
- Δ：缩放因子

**基本操作**：
- HADD：同态加法
- HMUL：同态乘法（消耗一个模数层，需 Rescale）
- PMUL：明密文乘法
- HROT：同态旋转（需要旋转密钥）

**CKKS Bootstrapping**（4步）：Modulus Raising → Coeff2Slot → Approximated Modulo → Slot2Coeff
- **核心开销**：大量 HMUL 和 HROT
- **Coeff2Slot/Slot2Coeff**：本质是密文矩阵向量乘法（MatVec）

### TFHE 方案

**三种密文**：
1. **LWE 密文**：`(a₀,...,a_{nlwe-1}, b) ∈ Z^{nlwe+1}_q`，标量消息
2. **RLWE 密文**：多项式对 `(B(x), A(x))`，维度 N
3. **RGSW 密文**：`(k+1)×(k+1)·l` 维矩阵，作为自举密钥

**TFHE Functional Bootstrapping（Algorithm 1）**：
```
输入：LWE密文 ct=(a,b)，自举密钥 BRK，测试多项式 testP
1. acc ← X^{-(b + N/β)} · testP        // 初始化累加器
2. for i = 0 to n-1:
       acc ← acc + acc ⊡ BRK_i · (X^{a_i} - 1)  // CMux门操作（外积）
3. ct = SampleExtract(acc)              // 提取常数项
输出：f(m) 的 RLWE 密文
```

关键步骤：Modulus Switching → **Blind Rotation（nlwe 次 CMux）** → Sample Extract

### 方案切换算法

**CKKS → TFHE**：
1. Slot2Coeff（CKKS HMUL/HROT密集）
2. Sample Extract → 每个槽对应一个独立 LWE 密文
3. Key Switching → 格式兼容

**TFHE → CKKS**：
1. Functional Bootstrapping（LUT evaluation，nlwe 次 CMux/槽）
2. **Repacking**：多个 LWE 密文 → 单个 CKKS RLWE 密文（核心：矩阵向量乘法）

---

## 三、核心技术方案

### 3.1 可扩展 NTT 加速设计

#### 问题根源
- 传统 GPU NTT：每级末尾需线程同步，最多 log₂N-1 次全局同步
- 同步停顿占总时间约 **35%**
- CKKS 大多项式（~2^16）和 TFHE 小多项式（~2^10）差异巨大，单一设计无法兼顾

#### 技术 1：蝴蝶分解（Butterfly Decomposition）——针对 TFHE 小多项式

**原理**：将完整 butterfly 拆为上半蝴蝶（upper triangle）和下半蝴蝶（lower triangle），重新分配使同一线程处理来自不同蝴蝶的同侧，从而消除级间同步。

```
传统：线程0 和 线程1 各处理完整butterfly → 线程间有数据依赖 → 需要同步
分解：线程0处理两个butterfly的上半 → 线程1处理两个butterfly的下半 → 同侧无依赖 → 融合两级
```

**效果**：16点NTT同步次数从3降至1，达到 **1.37×** 加速（N=8192）

**数据冲突处理**：引入多项式副本（polynomial replica）追踪中间结果，避免 RAW 冲突（线程 in-place 写后另一线程读的问题）。奇数级数时，剩余级使用 radix-2 实现。

**适用范围**：小多项式（TFHE，N≤8192）。大多项式（N=65536）时，thread divergence 增多（BD-NTT-3 有 51.48 次分支分歧 vs baseline 11.13 次），性能反而下降。

#### 技术 2：线程聚合（Thread Aggregation）——针对 CKKS 大多项式

**原理**：将多个线程的蝴蝶任务合并给单个线程，使一个线程处理多个 butterfly，从而融合多级，减少全局同步。

```
传统（4点NTT）：线程0/1 各处理一个butterfly/级 → 需同步
聚合（A=2）：将线程1的任务给线程0 → 线程数从N/2降至N/4 → 两级融合为一级
```

线程数 K 与每线程蝴蝶数 H 的关系：`K · H · log_A(N) = N/2 · log₂(N)`，其中 H = 2^A

引入"重定向索引"（redirection index）解决多蝴蝶映射问题。

**效果**：N=32768 时达到 **1.3×** 加速，但线程数过少会降低并行度（TA-NTT-3 效果不如 TA-NTT-2）。

**适用范围**：大多项式（CKKS，N≥32768）。

#### 技术 3：多项式系数洗牌（Polynomial Coefficient Shuffling）

**问题**：不同线程块中存在跨块数据依赖 → 需要全局同步（代价极高）。

**原理**：在执行前重排多项式系数的内存布局，使相互依赖的数据落在同一线程块的 shared memory 中，将全局同步降为块内局部同步。

```
传统：Thread block 0 处理 x[0]/x[4]，Thread block 1 处理 x[2]/x[6]
     → 下一级 Thread block 0 需要 x[2]（在block 1中）→ 全局同步
洗牌后：重排使 x[0]/x[2]/x[4]/x[6] 同在 Thread block 0 的 shared memory
     → 只需块内局部同步
```

**效果**：全局同步次数从最多 11 次（N=4096, radix-2）降至 **1 次**。
结合蝴蝶分解：N=4096 时 1.25× 加速；结合线程聚合：N=65536 时 1.2× 加速。

**限制**：shared memory 容量有限（无法容纳 N>8192 的 64-bit 多项式），因此对大多项式只能部分应用。

#### 技术 4：SM 感知最优同步切换点（OSSP）

**混合策略**：Phase 1（有系数洗牌的块内同步）→ Phase 2（无系数洗牌的块内同步）

**切换点选择问题**：切换太早 → 硬件利用率低；切换太晚 → 全局同步开销大。

**SM 感知切换触发器**：
```
给定 SM 数量 S，融合级数 F
探索所有同步组合 (I1, I2)，满足 I1 + I2 = log_F(N) - 1
寻找使 T = (2^F)^I1 最接近 S 的 I1
```

**效果**：
- A100（108 SMs）和 RTX3070（46 SMs）上，最优切换点均在数据块数 T=64 时（最接近各自 SM 数）
- 相比前人方案（切换过早），各多项式长度加速 **1.43×–2.78×**
- 全部优化组合（OSSP-PCS-TA-NTT-2）vs BASE-NTT：最高 **1.48×/1.58×**（RTX3070/A100）

### 3.2 CMux 门级并行化（LUT Evaluation 加速）

#### 传统方法（Slot-level Parallelism）的问题
- 每个 TFHE LWE 密文独立映射到 GPU 进行功能自举
- 每次 CMux 操作：Decompose + NTT/INTT + MAC，但小 TFHE 参数导致工作量不足，GPU 利用率低
- 需要 nslot × nlwe 次 CMux，每次单独 kernel launch，启动开销大
- 所有密文共享同一自举密钥，但传统方法重复访问

#### CMux 门级并行化方案

**核心洞察**：nlwe 个 TFHE 密文虽然数据不同，但共享完全相同的 CMux 操作和自举密钥。

**方法**：将所有密文在第 i 轮的 CMux 合并为一个批量 CMux 门：
```
第 i 轮 CMux：
  传统：对每个密文单独执行 CMux(acc_j, BRK_i)，j=0..nslot-1
  批量：一次性对所有密文执行 batched_CMux(all_acc, BRK_i)
```

**复杂度对比（Table I）**：

| 指标 | Slot-level | CMux-level（本文） |
|------|-----------|----------------|
| CMux 执行次数 | nslot × nlwe | nlwe |
| 密钥访问次数 | nslot × nlwe | nlwe |
| 每 CMux 处理密文数 | 1 | nslot |
| 核启动次数 | nslot × nlwe | nlwe |

减少因子 nslot 倍的 CMux 执行和密钥访问，同时数据量增大 nslot 倍提升 GPU 利用率。

### 3.3 无同态旋转 MatVec（Repacking 加速）

#### Repacking 的数学本质

将 nslot 个 TFHE LWE 密文合并为一个 CKKS RLWE 密文，核心步骤是线性变换（LT）：

```
LT：As + b    （矩阵向量乘法）
  A：nslot × nlwe 矩阵（每行是一个 LWE 密文）
  s：nlwe × 1 向量（LWE 密钥的 RLWE 加密）
```

注：通常用 Tiling 将 A 扩展为方阵（nslot = nlwe）。

#### 三种 MatVec 实现对比

**方法 1：对角线法（D-MatVec）**
- 将矩阵 A 转化为对角向量 ui
- 对密文 s 应用 Automorph，再与 ui 标量乘后累加
- 同态旋转次数：nslot - 1

**方法 2：BSGS 法（Pegasus，P-MatVec）**
- Baby step 预存旋转密文（代价：存储）
- Giant step 执行同态旋转（代价：实时旋转）
- 同态旋转次数：√nslot
- 仍需实时旋转，LT 耗时 ~35s，占 Repacking 89%

**方法 3：无旋转 BSGS（本文，HRF-MatVec）**⭐

**核心洞察**：
1. LWE 密钥已知 → 所有旋转密文可**全部预计算**
2. Automorph 天然支持线性操作（加法、乘法、automorph 本身）
3. Giant step 的 automorph 可与 Baby step 的线性操作合并

**实现**：
```
预计算阶段（离线）：
  对所有 Giant step 的 automorph 和对应的 Baby step 线性操作合并
  预计算所有 nslot 个旋转密文（对应合并后的操作）

实时阶段（在线）：
  构建明文矩阵（fat/thin → row/column tiling）
  对角向量与预计算旋转密文按 Giant/Baby step 分组
  执行标量乘法 + 累加（无同态旋转！）
  输出 CKKS RLWE 密文
```

**复杂度对比（Table II）**：

| 方法 | 同态旋转次数 | 标量乘次数 | 实时代价 |
|------|-----------|---------|---------|
| D-MatVec | nslot - 1 | nslot | 高 |
| P-MatVec（BSGS） | √nslot | nslot | 中 |
| **HRF-MatVec（本文）** | **0** | nslot | **极低** |

实时阶段只剩标量乘法，代价极低，Repacking 加速 **123.3×**（相对 CPU Pegasus）。

---

## 四、实验配置

### 硬件与软件环境

| 配置项 | 数值 |
|-------|------|
| CPU | Intel i9-10900K（10核，5.3GHz，128GB DRAM） |
| 主GPU | NVIDIA Tesla A100（108 SMs，6912 cores，1.41GHz） |
| 对比GPU | NVIDIA GeForce RTX 3070（46 SMs，5888 cores，1.5GHz） |
| 对比CPU | Intel Xeon Platinum 8592V（256核，3.9GHz，1TB内存）|
| 操作系统 | Ubuntu 18.04.5 |
| CUDA | 11.2 |
| GCC | 9.0.0 |
| Profiling | NVIDIA Nsight Compute |

### Benchmark 设置

| 参数 | 值 |
|-----|---|
| 多项式长度 | 4096 – 65536 |
| 模数 q | 56-bit（双字大小） |
| NTT 基线 | BASE-NTT（radix-2） |
| CKKS 参数 | (N, logQ, L, dnum) = (2^16, 2305, 44, 45) |
| Phantom 参数 | (2^16, 1700, 41, 42) |
| 批大小 | 128（与 TensorFHE 一致） |

### 对比对象

**CPU 库**：HElib, SEAL, OpenFHE（CKKS）；TFHE-rs（TFHE）；Pegasus（方案切换）

**GPU 实现**：Over100x, HE-Booster, Phantom, TensorFHE（CKKS）；cuFHE, nuFHE, CPU-GPU-TFHE（TFHE）；HEAPGPU（Repacking）

---

## 五、实验结果

### 结果 1：NTT 优化效果（单 NTT kernel）

#### BD-NTT（蝴蝶分解）vs BASE-NTT（Figure 11）
- N=8192（TFHE适用）：BD-NTT-2 加速 **1.34×–1.37×**（五种同步方法均值）
- N=65536（CKKS）：BD-NTT-2 **慢于** BASE-NTT（thread divergence 激增）
- BD-NTT-3 最差（过度分解，分歧分支从 11.13 增至 51.48）

#### TA-NTT（线程聚合）vs BASE-NTT（Figure 11）
- N=32768（CKKS适用）：TA-NTT-2 加速 **1.2×–1.32×**
- TA-NTT-3 无性能提升（线程数减半，并行度下降）

#### BD vs TA 对比
- 小多项式（N=8192）：BD-NTT-2 比 TA-NTT-2 快 **1.02×–1.22×**
- 大多项式（N=65536）：TA-NTT-2 比 BD-NTT-2 快 **1.26×–1.33×**

#### PCS（系数洗牌）效果（Figure 13）
- N=4096：PCS-BD-NTT-2 比 BD-NTT-2 快 **1.25×**
- N=65536：PCS-TA-NTT-2 比 TA-NTT-2 快 **1.2×**

#### OSSP 最优切换点（Table III）
OSSP-PCS-TA-NTT-2 vs BASE-NTT：最高 **1.48×/1.58×**（RTX3070/A100）

相比前人混合方案：各长度加速 **1.43×–2.78×**（如 N=4096 时 2.67×/2.78×）

### 结果 2：CKKS 原语 GPU vs CPU（Table IV）

参数：N=2^16, logQ=2305

| 操作 | vs SEAL | vs OpenFHE | vs HElib |
|------|---------|-----------|---------|
| HADD | 110.7× | 192.7× | 19.7× |
| HMUL | 115.9× | 76.6× | 73.1× |
| HROT | 86.8× | 75.8× | 112.2× |

### 结果 3：CKKS GPU vs 其他 GPU 实现（Table V，batch_size=128）

| 操作 | vs Over100x | vs HE-Booster | vs Phantom | vs TensorFHE |
|------|------------|-------------|----------|------------|
| HMUL | 3.12× | 1.72× | 1.58× | **1.23×** |
| HROT | 类似 | 类似 | 类似 | 类似 |
| BTS（bootstrapping） | 2.11× | 1.96× | - | **1.15×** |

硬件利用率：HMUL 和 HROT 均超过 90%，Chameleon 达 **98.2%**（多流并行流水线）。

### 结果 4：TFHE 功能自举（Table VII）

安全级别：80/110/128 bit

| 对比对象 | 加速比（128-bit） |
|---------|---------------|
| TFHE-rs（CPU） | 3.22× |
| cuFHE（GPU） | 1.47× |
| nuFHE（GPU） | 1.47× |
| CPU-GPU-TFHE | **1.51×** |

### 结果 5：方案切换完整流水线（Table VIII，vs CPU Pegasus）

| 阶段 | Chameleon vs Pegasus 加速比 |
|------|--------------------------|
| Sample Extract | 106.9× |
| **Repacking（LT）** | **123.3×** |
| Slot2Coeff | 60.5× |
| LUT Evaluation | 65.6× |
| **整体平均** | **67.3×** |

### 结果 6：Repacking 对比（Table IX，vs HEAPGPU）

Chameleon 在所有槽数下均优于 HEAPGPU。
原因：HEAP 用基于 RLWE automorphism 的 Repacking（旋转次数随槽数线性增长），而 Chameleon 用 LWE-BSGS（预计算旋转密文）。

### 结果 7：应用级 Benchmark

#### 非多项式函数（Table X，vs 单线程 Pegasus）

| 函数 | 加速比 |
|------|-------|
| Sigmoid | 50.1× |
| ReLU | ~55× |
| Sqrt | ~60× |
| Max-Pooling（2×2） | ~65× |
| Min-Index（256密文） | ~67× |
| Sorting（64密文） | 67.8× |

#### 私有决策树（Table XI，vs 16线程 Pegasus，UCI数据集）

| 数据集 | 加速比 |
|-------|-------|
| Iris | 3.9× |
| Housing | 3.9× |
| Spambase | 4.0× |

#### 安全 K-means 聚类（Table XII，vs 20线程 Pegasus）

| 阶段 | 加速比 |
|------|-------|
| Distance（CKKS） | 较高 |
| Repacking（CKKS） | 最高（HRF-MatVec） |
| Centroid（CKKS） | 较高 |
| Extract（TFHE） | 中等 |
| **Min-index & Recip（TFHE）** | **最高（CMux-level并行）** |
| **整体** | **3.8×–6.2×** |

---

## 六、相关工作梳理

### GPU NTT 加速工作

| 工作 | 主要方法 | 局限 |
|------|---------|------|
| Özerk et al. [30] | 混合 single/multi kernel，local/global sync | 未分析最优切换点，切换过早导致利用率低 |
| NTTFusion [19]（本文作者前作） | NTT 融合加速 | 本文在此基础上扩展 |
| 本文 | BD+TA+PCS+OSSP，最优切换点 | - |

### GPU FHE 加速工作

| 工作 | 方案 | 主要贡献 |
|------|------|---------|
| Over100x [25] | CKKS | 首个 GPU RNS-CKKS+Bootstrap |
| HE-Booster [8] | BGV/CKKS | 多项式算术框架 |
| Phantom [35] | BGV/BFV/CKKS | 统一框架 |
| TensorFHE [10] | CKKS | Tensor Core 加速 NTT |
| cuFHE/nuFHE | TFHE | GPU TFHE 库 |
| **Chameleon（本文）** | CKKS+TFHE | **首个 GPU 方案切换加速** |

### 硬件 FHE 加速工作

| 工作 | 平台 | 方案 | 特点 |
|------|------|------|------|
| HEAX [2] | FPGA | CKKS | CKKS 专用架构 |
| F1 [5] | ASIC | Word-wise | 可编程，高吞吐 |
| CraterLake [6] | ASIC | Word-wise | 无限乘法深度 |
| HEAP [40] | FPGA | 方案切换 | CKKS-side 并行自举，与本文互补 |
| **本文 vs HEAP** | | | HEAP 并行 CKKS Bootstrap，Chameleon 并行非线性操作；Repacking 方法不同（RLWE automorphism vs LWE-BSGS） |

---

## 七、对 RVV 密文 MatVec 加速工作的参考价值

### 最重要的算法参考：HRF-MatVec

本文 HRF-MatVec 是迄今最优的密文 MatVec 算法框架，对 RVV 实现有直接指导意义：

```
预计算阶段（与平台无关）：
  1. 将 Giant step 的 automorph 与 Baby step 线性操作合并
  2. 预计算所有 nslot 个旋转密文（存储开销：√nslot × 2 × p × 8 bytes）

实时阶段（RVV 实现目标）：
  对每个 Giant step j：
    对每个 Baby step i：
      明文对角向量 u[i,j] × 预计算旋转密文 rot_ct[combined(i,j)]  // RVV PMUL
    累加所有 Baby step 结果                                        // RVV HADD
  输出 CKKS 密文
```

**在 RVV 上的实现映射**：
- PMUL（明密文乘法）= NTT域逐元素乘法 = **RVV Shoup Reduction（TCHES Algorithm 6）**
- HADD（密文加法）= 逐元素模加 = **RVV vadd + 范围约减（TCHES Algorithm 5）**
- 预计算旋转密文的访问 = 大量连续/步幅内存读 = **RVV vle / vlse**

### MatVec 与 NTT 的融合机会

Chameleon 揭示 MatVec 的计算核心是 PMUL（NTT域乘法）和 HADD，若密文保持在 NTT 域（OpenFHE EVAL格式）：

```
传统：密文 → INTT → 时域乘法 → NTT → 结果（每次PMUL需完整NTT往返）
优化：密文保持NTT域 → 直接逐元素乘法（无NTT开销）→ 结果仍在NTT域
```

**CKKS Bootstrapping 的 Coeff2Slot/Slot2Coeff** 是 MatVec 的主要调用场景，直接收益明显。

### 内存规模估算（指导 RVV Cache 策略）

预计算旋转密文的存储量：
- 参数：nslot = 2^13, p = 2^14, 64-bit元素
- 存储量 = √nslot × 2 × p × 8 bytes = √8192 × 2 × 16384 × 8 ≈ **24 MB**
- 远超 TCHES 论文 gem5 的 L2（1MiB），需要 L3（32MiB）甚至主存

**启示**：需设计 cache 友好的 Baby/Giant step 遍历顺序，或考虑流式分块加载。

### 对 OpenFHE 的注入点

| 功能 | OpenFHE 函数 | MatVec 位置 |
|-----|------------|-----------|
| CKKS Bootstrapping | `EvalBootstrap` | Coeff2Slot/Slot2Coeff 内部 |
| 线性变换 | `EvalLinTransform` / `EvalSlotsToCoeffs` | 直接注入 |
| Repacking（若实现方案切换） | 自定义 | 完整 HRF-MatVec 流程 |

### 性能预期参考

Chameleon 数据：
- CPU Repacking 耗时约 35s（BSGS，nslot=1024，单线程 Pegasus）
- Chameleon GPU 123.3× → 约 280ms

TCHES 数据：
- NTT 加速 5–7×（VPU）
- MatVec 中 NTT 占比高 → MatVec 整体加速潜力显著

若 MatVec 的 NTT 部分占 50%，其余（标量乘+加法）也能 ~5× 加速（均为 RVV 友好操作），则 MatVec 总加速比有望达到 **5–7×**，进而将 bootstrapping 中 Coeff2Slot/Slot2Coeff 的贡献大幅提升。

---

## 八、符号约定

| 符号 | 含义 |
|------|------|
| nckks | CKKS 多项式次数 |
| nslot | CKKS 槽数（≤ nckks/2） |
| nlwe | LWE 密文维度 |
| nlut | LUT evaluation 中密文数量 |
| N | TFHE 多项式次数 |
| nslot | 功能自举槽数 |
| BRK | 自举密钥（Blind Rotation Key），RGSW 格式 |
| CMux | 密文多路选择器（Ciphertext Multiplexer） |
| BSGS | Baby-Step Giant-Step 算法 |
| HRF | Homomorphic Rotation-Free（无同态旋转） |
| MatVec | 矩阵向量乘法 |
| LT | 线性变换（Linear Transformation） |
| SM | Streaming Multiprocessor（GPU 流多处理器） |
| TB | Thread Block（线程块） |
| GMEM | Global Memory（GPU 全局内存） |
| SMEM | Shared Memory（GPU 共享内存） |
| CMEM | Constant Memory（GPU 常量内存） |
| PCS | Polynomial Coefficient Shuffling |
| BD-NTT | Butterfly Decomposition NTT |
| TA-NTT | Thread Aggregation NTT |
| OSSP | Optimal Synchronization Switching Point |