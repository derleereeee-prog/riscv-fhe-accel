# 文献参考：Accelerating NTT with RISC-V Vector Extension for Fully Homomorphic Encryption

## 元信息

| 字段 | 内容 |
|------|------|
| **标题** | Accelerating NTT with RISC-V Vector Extension for Fully Homomorphic Encryption |
| **作者** | Tiago B. Rodrigues, Alexandre Rodrigues, Manuel Goulão, Pedro Tomás, Leonel Sousa |
| **机构** | INESC-ID, Instituto Superior Técnico, Universidade de Lisboa, Portugal |
| **发表** | IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES), Vol. 2025, No. 4, pp. 711–736 |
| **DOI** | 10.46586/tches.v2025.i4.711-736 |
| **收稿/接收/发表** | 2025-04-15 / 2025-06-15 / 2025-09-05 |
| **代码仓库** | https://github.com/hpc-ulisboa/NTT-RVV-CHES2025/ |
| **关键词** | NTT, Vectorization, RVV, RISC-V, Homomorphic Encryption, OpenFHE |

---

## 一、研究背景与动机

### FHE 的计算开销问题
- FHE（全同态加密）支持对加密数据直接计算，无需解密，是隐私保护云计算的核心技术
- 相比明文计算，FHE 开销高达 **4个数量级**
- 最耗时的操作是**服务端自举（bootstrapping）**及相关的密钥生成，核心算子是 **NTT（数论变换）和 INTT（逆NTT）**

### RISC-V 与 RVV 的机遇
- RISC-V 是开源 ISA，正在从嵌入式到 HPC 快速普及，获得欧盟、美国、中国大量资助
- RVV（RISC-V Vector Extension）是标准向量扩展，具有：
  - **向量长度无关性**（Vector-Length Agnostic）：同一代码可在不同 VLEN 硬件上运行
  - **动态向量长度调整**：循环尾部处理无需特殊处理
  - **寄存器分组（LMUL）**：LMUL=2 时每条指令操作两个寄存器宽度，相当于加倍寄存器宽度
  - **掩码（Predicated）执行**
  - **Scatter-Gather 内存访问**
- 现有 RVV 硬件分为 v0.7.1（2019年草案）和 v1.0（2021年正式版）两个版本

### 文献空白
- Intel HEXL 利用 AVX-512 加速 NTT（x86 专用）
- 现有 RISC-V NTT 加速主要面向 PQC（参数较小），未针对 FHE
- **本文填补空白**：首个在标准 RISC-V 上通过 RVV 扩展加速 FHE 的工作

---

## 二、相关 FHE 方案与 NTT 算法

### 支持的 FHE 方案
| 方案 | 类型 | 特点 |
|------|------|------|
| **BGV** | 第二代，批处理 | 整数运算 |
| **BFV** | 第二代，批处理 | 整数运算 |
| **CKKS** | 第四代 | 近似实数/复数运算，适合 ML，**本文重点** |
| FHEW/TFHE | 第三代 | 布尔电路，不支持批处理 |

- 多项式乘法是 RLWE 问题的核心，复杂度 O(p²) → NTT 降为 **O(p log p)**
- FHE 实现将多项式存储在 NTT 域以提高效率
- 多项式系数通过 **RNS（剩余数系统）** 分解为多个小模数，本文针对 **64-bit 模数**

### 四种 NTT/INTT 算法对比

| 算法 | 类型 | 内存访问模式 | 是否原地 | 适合向量化 |
|------|------|------------|---------|----------|
| **Cooley-Tukey (CT)** | NTT | 每阶段变化（scatter） | ✅ | ❌ 效率低 |
| **Gentleman-Sande (GS)** | INTT | 每阶段变化（scatter） | ✅ | ❌ 效率低 |
| **Korn-Lambiotte (KL)** | NTT | 固定步幅，不随阶段变化 | ❌（需辅助数组） | ✅ **优选** |
| **Pease** | INTT | 固定步幅，不随阶段变化 | ❌（需辅助数组） | ✅ **优选** |

**关键区别**：
- CT/GS 每阶段 butterfly 输入地址不同 → scatter 访问 → 向量化效率差
- KL/Pease 输入地址固定（连续或固定步幅）→ 向量化友好 → **本文主力算法**
- KL/Pease 非原地，需大小为 p 的辅助数组，每阶段交换，最终可能需额外一次拷贝

### 模约减算法

**Barrett Reduction**：适合通用模约减，将除法转化为乘法+移位

**Shoup Reduction**（本文主要采用）：
- 针对常数模乘 `z = x × y mod Q`（y 为常数）
- 预计算 `b_inv = 2^k × y / Q`
- 向量化版本（Algorithm 6）：
  ```
  q ← vmulh(x, b_inv)    // 高位乘法
  y1 ← vmull(x, y)       // 低位乘法
  y2 ← vmull(q, Q)       // 低位乘法
  z ← y1 - y2
  z ← z - Q  if z >= Q   // 掩码减法
  ```
- OpenFHE 已预计算所需的辅助表，可直接复用

---

## 三、核心算法设计

### 3.1 向量化 Cooley-Tukey NTT（Algorithm 1）

分三段处理（以 p=16384 为例，VLEN=16384 bit → vector_length=256 个64-bit元素）：

**Stage A**（`t >= vector_length/2`）：直接 SIMD 化，向量寄存器利用率 100%

**Stage B**（`1 < t < vector_length/2`）：使用**掩码操作**
- 将两层循环融合，一次加载整个向量长度的数据
- 只对需要的位置执行 butterfly（掩码 store）
- 掩码和辅助单位根可预计算
- 上限保持在 p/2 个活跃元素
- **问题**：掩码变体通常比非掩码慢，且难以集成进 OpenFHE

**Stage C**（`t = 1`）：使用步幅为 2 的加载/存储（strided load/store），利用率恢复 100%

### 3.2 向量化 Korn-Lambiotte NTT（Algorithm 2）⭐主力

分三段：

**Stage A**（`m = 1`）：广播单位根（omegaTable[1]）到整个向量，调用 `KL_STANDARD_LOOP`

**Stage B**（`1 < m < vector_length`）：
- 需要的单位根数量 < 向量长度，利用 `vgather` 操作重排单位根（Algorithm 4）
- 索引生成：`index = (index_og & index_stage_aux) + m`，其中 `index_stage_aux` 每阶段左移+置位更新

**Stage C**（`m >= vector_length`）：
- 每次迭代加载 vector_length 个单位根，跳过已加载的（stride=m 的规律性）
- 调用 `KL_STANDARD_LOOP`

`KL_STANDARD_LOOP` 核心：
```
for i in 0 to p-1 step 2*vector_length:
    U ← x[i/2]                  // 连续加载
    V ← x[i/2 + p/2]            // 连续加载
    R1, R2 ← Butterfly(U, V, W)
    y[i :: 2] ← R1              // 步幅2存储
    y[i+1 :: 2] ← R2            // 步幅2存储
SWAP(x, y)
```

### 3.3 向量化 Pease INTT（Algorithm 3）⭐主力

与 KL NTT 对称，访问模式相反：
- 输入：步幅2加载（strided load）
- 输出：连续存储
- 最后一个循环需乘以常数 p⁻¹
- Butterfly 形式略有不同：`R1 = (U+V) × Aux mod Q`，`R2 = (U-V) × W × Aux mod Q`

### 3.4 单位根访问优化（Algorithm 4）

三种访问策略：
1. **首尾阶段**：只用 omegaTable[1]，broadcast 即可
2. **中间阶段**（m < vector_length）：所有需要的单位根可装入一个向量寄存器，用 `vgather` 按需取值
3. **大阶段**（m >= vector_length）：按步幅 m 加载，每个内层循环复用同一批单位根

### 3.5 向量化模约减（Algorithm 5）

针对范围 [0, 2Q) 的简单约减：
```
mask ← Q <= x       // 比较生成掩码
y ← x - Q  [mask]  // 掩码减法
```
预期加速比≈向量长度。

---

## 四、OpenFHE 集成

### 集成方式
- 编译完整 OpenFHE 库到 RISC-V 目标
- 由于编译器对 RVV intrinsics 的限制，无法直接内联
- 采用**辅助头文件 + 目标文件注入**方式：在 OpenFHE 的 NTT/INTT 函数中插入调用钩子
- 利用 OpenFHE 已有的内部数组和预计算表，减少额外预计算

### 支持的 FHE 库对比（Table 1）

| 库 | CKKS | CKKS+Bootstrap | BGV | BFV | TFHE | FHEW |
|---|---|---|---|---|---|---|
| **OpenFHE** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SEAL | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| HEAAN | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 五、实验平台

### 平台 1：VPU（FPGA 原型）

| 参数 | 数值 |
|------|------|
| 基础核 | 标量有序 RISC-V |
| RVV 版本 | v0.7.1 |
| VLEN | 16384 bit（最大），可配置 8Kibit / 4Kibit |
| 向量lanes | 8 |
| 每lane吞吐 | 1个64-bit指令/cycle |
| 总吞吐 | 512 bit/cycle（8×64-bit），FMA=16 个64-bit op/cycle |
| FPGA板 | Xilinx Virtex UltraScale+ HBM VCU128（VU37P） |
| 时钟频率 | 50 MHz |
| 内存带宽 | 64 bytes/cycle = 3.2 GB/s |
| 芯片工艺 | GlobalFoundries 22FDX FD-SOI（1.3 mm²） |
| 操作系统 | Ubuntu 22.04.1 LTS |

### 平台 2：gem5 模拟 OoO 处理器

| 参数 | 数值 |
|------|------|
| 核心类型 | 大规模超标量乱序 |
| RVV 版本 | v1.0 |
| 发射宽度 | 12条指令/cycle |
| 前端供给 | 8条指令/cycle |
| 缓存层次 | L1: 64KiB / L2: 1MiB / L3: 32MiB |
| 配置(i) | 1×512-bit SIMD单元，16个64-bit op/cycle |
| 配置(ii) | 2×512-bit SIMD单元，32个64-bit op/cycle |
| 配置(iii) | 1×1024-bit SIMD单元，32个64-bit op/cycle |
| 配置(iv) | 1×2048-bit SIMD单元，64个64-bit op/cycle |

**gem5 Bug 修复**：发现 gem5 对向量步幅加载指令（strided load）添加了假依赖（dummy source register `vtmp0`），导致与之前掩码指令的 RAW 冲突。修复方法：将 strided load 的 dummy source 改为 `vtmp1`，消除假依赖，恢复 ILP。

### 编译器配置
- 编译器：Clang（LLVM）
- VPU 平台：版本 12.0.0（支持 RVV 0.7.1）
- gem5 平台：版本 21.0.0git（支持 RVV 1.0）
- 标量代码：`-O2`（-O3 无额外收益）
- 向量代码（RVV intrinsics）：`-O3`

---

## 六、实验结果与Benchmark

### Benchmark 套件

| Benchmark | 内容 | 评估节 |
|-----------|------|--------|
| **Simple NTT** | 单独测 NTT/INTT 的向量加速比，按阶段分解 | §5.3 |
| **Simple HE Operations** | 加法、减法、Hadamard积、旋转等基础 HE 操作 | §5.4 |
| **Bootstrapping** | 序列乘法触发自举，评估复杂服务端操作加速 | §5.4 |
| **Fully-Connected NN** | 三层全连接网络 + ReLU（Chebyshev近似）+ Bootstrap | §5.4 |

所有 benchmark 在 BGV、BFV、CKKS 三种方案上测试，RNS 编码为 64-bit。

### 结果 1：NTT/INTT 阶段分析（VPU，p=16384）

**KL NTT vs CT NTT（Figure 7）**：
- CT NTT：第 8–13 阶段性能下降约 50%（只有一半元素活跃，其余被掩码）
- KL NTT：每阶段性能几乎恒定（单位根访问模式固定）
- **KL NTT 整体比 CT NTT 快约 50%**

### 结果 2：NTT/INTT 加速比 vs 多项式长度（VPU，Figure 8）

| log₂p | KL NTT 加速比 | Pease INTT 加速比 |
|--------|-------------|----------------|
| 10 | 5.1× | 5.0× |
| 11 | 5.8× | 5.9× |
| 12 | 6.7× | 6.7× |
| 13 | 6.8× | 6.9× |
| 14 | 7.0× | 7.0× |
| 15 | 7.0× | 7.1× |
| 16 | 7.4× | 7.4× |
| 17 | 7.7× | 7.3× |
| 18 | 7.9× | （同级别） |

参考范围 p=2^10 到 2^15（HE 安全标准），加速比 **5.0×–7.1×**，随 p 增大趋于提升。

### 结果 3：与 Intel HEXL 对比（Table 2）

硬件：Intel Xeon Gold 6312U，三个 AVX-512 单元，峰值 48个64-bit op/cycle（FMA IFMA52）

| 硬件配置 | 峰值(op/cycle) | p=1024 NTT | p=1024 INTT | p=16384 NTT | p=16384 INTT | p=65536 NTT | p=65536 INTT |
|---------|-------------|-----------|------------|------------|-------------|------------|-------------|
| Intel HEXL（Xeon 6312U） | 48 | 6.89× | 6.33× | 6.34× | 5.78× | 5.09× | 5.59× |
| VPU 16Kibit | 16 | 5.10× | 5.01× | 7.07× | 7.00× | 7.42× | 7.38× |
| VPU 8Kibit | 16 | 4.48× | 4.43× | 5.50× | 5.55× | 5.77× | 5.77× |
| VPU 4Kibit | 16 | 3.39× | 3.30× | 3.97× | 3.92× | 3.93× | 3.99× |
| gem5 1×512bit | 16 | 9.03× | 8.11× | 7.05× | 6.89× | 6.84× | 6.85× |
| gem5 2×512bit | 32 | 12.90× | 14.36× | 7.54× | 8.03× | 7.20× | 7.94× |
| gem5 1×1Kibit | 32 | 16.35× | 15.59× | 9.99× | 10.01× | 8.75× | 10.31× |
| gem5 1×2Kibit | 64 | **17.60×** | **27.05×** | 11.08× | 11.21× | 9.09× | 11.76× |

**关键观察**：
- VPU 在大 p 下超过 HEXL（峰值性能只有 HEXL 的 1/3）
- gem5 单512-bit 配置（等效 FPGA 吞吐）在所有 p 下超越 HEXL
- VLEN 越大，加速比越高（即使峰值吞吐不变）——更大向量减少指令数，提升 lane 利用率
- 随 p 增大，内存带宽成为瓶颈，加速比趋于饱和

### 结果 4：与更广泛 SoA 对比（Table 3）

| 方案类型 | 代表工作 | 面积 | p | Q | NTT加速 | 备注 |
|---------|---------|------|---|---|---------|------|
| 本文（VPU） | FHE | 1.3mm²@22nm | 16384 | 64-bit | 7.07× | 标准RISC-V |
| 本文（gem5 1Kibit） | FHE | - | 16384 | 64-bit | 9.99× | 标准RISC-V |
| Intel HEXL | FHE | x86平台 | 16384 | 64-bit | 6.34× | AVX-512专用 |
| [LMP22] | PQC | 93.2K+17.9K LUT | 256 | 32-bit | 35.3× | 定制指令 |
| [ZKS+23] | PQC | 161.9K+44.6K | 256 | 32-bit | 858× | 定制指令，仅p=256 |
| [RLPD20] HEAX | FHE FPGA | 600K+1746K | 16384 | 54-bit | 25.7× | FPGA专用 |
| [SFK+21] F1 | FHE ASIC | 151mm²@12nm | 16384 | 32-bit | 8838× | 面积大2个数量级 |
| [OATS23] GPU | FHE BGV | RTX 3060Ti | 8192 | 64-bit | 68.81× | 批量吞吐，非单次延迟 |
| [CDH+24] GPU | FHE CKKS | RTX 4090 | 8192 | 64-bit | 692× | 批量吞吐 |

**GPU 注意**：GPU 批量 NTT 吞吐高，但单次延迟大（CPU-GPU 数据迁移数微秒到数十微秒，相当于数千 CPU 周期）；向量化实现无数据迁移开销。

### 结果 5：基础 HE 操作加速比（Figure 9，VPU，p=16384）

| 操作 | CKKS | BFV | BGV |
|------|------|-----|-----|
| encrypt/decrypt | 1.20× | 1.27× | 1.26× |
| addition | 1.21× | 1.29× | 1.25× |
| subtraction | 1.21× | 1.29× | 1.25× |
| multiplication | 1.38× | 1.54× | 1.35× |
| rotation | 1.30× | 1.31× | 1.41× |

加速比受 Amdahl 定律限制（NTT 在各操作中占比不同）。

### 结果 6：Bootstrapping 加速比（Figure 10，VPU，p=4096）

| 乘法次数 | Bootstrapping 调用次数 | NTT 占比 | INTT 占比 | 最大理论加速 | 实测加速 |
|---------|---------------------|---------|---------|------------|---------|
| 5 | 1 | 49.4% | 12.7% | 2.39× | ~1.77× |
| 10 | 2 | 50.7% | 13.8% | 2.53× | ~1.82× |
| 15 | 3 | 50.7% | 14.0% | 2.54× | ~1.84× |
| 20 | 4 | 51.2% | 14.4% | 2.60× | ~1.88× |
| 40 | 5 | 51.7% | 15.0% | 2.67× | **1.94×** |

乘法次数增多 → NTT/INTT 占比上升 → 加速比提升，峰值 **1.94×**。

### 结果 7：全连接神经网络加速比（Figure 11，VPU，p=4096）

网络结构：3层全连接 + ReLU（Chebyshev 近似 [OS24]） + Bootstrapping 刷新密文

| 层大小 | 本文加速比 | Intel HEXL 加速比 |
|-------|----------|----------------|
| 8 | 2.02× | 1.22× |
| 16 | 1.99× | 1.22× |
| 32 | 1.96× | 1.32× |
| 64 | 1.94× | 1.23× |
| 128 | 1.86× | 1.21× |
| 256 | 1.84× | 1.26× |
| 512 | 1.78× | （同级别） |
| 1024 | **1.69×** | 1.21× |

随层大小增大，加速比略降（原因：全连接+Chebyshev 的乘法深度恒定，bootstrapping 调用次数不变，NTT/INTT 相对权重下降）。

---

## 七、主要结论

1. **KL NTT + Pease INTT 优于 CT NTT + GS INTT**：在 VPU 上加速比 5.01×–7.42×，在 gem5 OoO 上最高 9.03×（单512-bit），最高 27.05×（1×2Kibit INTT）
2. **超越 Intel HEXL**：gem5 单512-bit（峰值性能仅 HEXL 的 1/3）在所有 p 下超越 HEXL
3. **完整 FHE 加速**：bootstrapping 1.94×，神经网络 1.69×–2.02×
4. **可移植性**：无需定制指令或硬件扩展，使用标准 RVV

---

## 八、对当前工作（RVV 密文 MatVec）的直接参考价值

### 可直接复用的组件

| 组件 | 位置 | 用途 |
|------|------|------|
| 向量化 Shoup Reduction | Algorithm 6 | MatVec 中的模乘核心 |
| 向量化范围约减 [0,2Q) | Algorithm 5 | MatVec 中的加法后约减 |
| KL NTT / Pease INTT | Algorithm 2/3 | MatVec 内部的 NTT 变换调用 |
| OpenFHE 集成框架 | §4 | MatVec 注入点参考 |
| gem5 + VPU 评估框架 | §5.1 | MatVec 性能评估 |

### 关键设计洞察

- **NTT 域存储**：OpenFHE 支持 Evaluation Form（NTT 域），MatVec 中密文若保持 NTT 域，可省去 NTT/INTT 往返
- **三段式分块**（m < VL / m = VL / m > VL）：Algorithm 2 的分段策略可迁移到 MatVec 的 BSGS 分块
- **内存带宽是 VPU 的瓶颈**：MatVec 访问大量预计算密文，需优先考虑 cache 友好的访问顺序
- **gem5 bug 修复经验**：strided load 的假依赖问题在 MatVec 的 BSGS 实现中同样可能出现

### 性能上限参考

基于 Amdahl 定律，若 MatVec 在 bootstrapping 中的占比约为 NTT（~50%）的数倍（因为 Coeff2Slot/Slot2Coeff 内部含大量旋转，每次旋转=NTT×2+乘法），则 MatVec 优化的 bootstrapping 总加速比有望超过 2×。

---

## 九、符号约定

| 符号 | 含义 |
|------|------|
| p | 多项式次数（NTT 长度） |
| Q | RNS 模数（coefficient modulus） |
| [·]_Q | 模 Q 约减 |
| VLEN | 向量寄存器位宽 |
| LMUL | 向量寄存器分组倍数 |
| VPU | Vector Processing Unit（FPGA 上的向量加速器） |
| OoO | Out-of-Order（乱序处理器） |
| RNS | Residue Number System（剩余数系统） |
| ILP | Instruction Level Parallelism |
| RLWE | Ring Learning With Errors |