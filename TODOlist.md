### 阶段 0：基线测量与算法路径选型（1-2 周）

**目标**：在开始任何 RVV 优化之前，搞清楚目标场景下不同算法路径的真实代价分布。

#### TODO 0.1 — Profile OpenFHE 中 matrix-vector 的现状

- **任务**：在 RISC-V 平台（FPGA VPU 或 gem5）上跑通 OpenFHE 的标量版本，对一个 $n_{slot} = 256$ 到 $4096$ 的矩阵向量乘法做完整 profile
- **输入参数**：CKKS，$N = 2^{14}$ 或 $2^{15}$，logQ 中等深度
- **测量维度**：NTT/INTT 占比、automorph 占比、PMul + Add 占比、Keyswitch 占比、memory access 占比
- **预期结果**：得到类似 Chameleon Figure 2 的执行时间饼图。**关键判断点**：如果 automorph + Keyswitch 占比 < 30%，说明 HRF 思路即使能消除它们，端到端收益也有限；如果 > 60%，则 automorph 加速极有价值
- **交付物**：一张 breakdown 图 + 一份分析文档

#### TODO 0.2 — 决策：选择 matrix-vector 的目标算法路径

- **任务**：基于 0.1 的结果，从三条路径中选一条作为主线
  - **路径 A（HRF-MatVec，Chameleon 思路）**：预计算所有旋转密文，实时只做 PMul + ADD。RVV 适配最简单，但需要大量预计算存储
  - **路径 B（传统 BSGS）**：保留 Giant step 的 automorph，需要优化 automorph 在 RVV 上的实现
  - **路径 C（混合）**：小 $n_{slot}$ 用 HRF，大 $n_{slot}$ 用 BSGS。需要根据存储约束自动选择
- **决策依据**：FPGA VPU 内存 vs. gem5 cache size vs. 目标 $n_{slot}$ 范围下预计算表的大小
- **交付物**：一份"算法路径选型报告"，明确接下来主要做哪条

---

### 阶段 1：核心算子 RVV 实现（3-5 周）

#### TODO 1.1 — 实现 RVV PMul（NTT 域标量乘多项式）

- **任务**：把"一个常数标量乘多项式向量"用 RVV 实现，作为 HRF 路径的最核心算子
- **设计要点**：
  - 用 `vmul.vx`（向量乘标量）+ `vadd.vv` 累加
  - 模约简继续用 TCHES 验证过的 Shoup reduction 的 RVV 版本
  - 尝试 `LMUL=1, 2, 4, 8` 四种配置，找最优
  - 输入多项式按 NTT 域存储，与 OpenFHE 内部表示对齐
- **预期加速比**：vs. 标量 OpenFHE，应该在 5×~10× 范围（接近 TCHES NTT 的加速比）
- **测量**：单算子延迟、吞吐量、cache miss 率
- **交付物**：可独立测试的 RVV PMul 函数 + benchmark 报告

#### TODO 1.2 — 实现 RVV 多项式累加（Baby step 的核心）

- **任务**：实现 $\sum_i c_i \cdot \mathbf{v}_i$（多个向量按标量加权累加）
- **设计要点**：
  - 与 PMul 融合：`vmacc.vx` 一条指令完成"乘 + 累加"，比拆开做快
  - 注意累加器的模约简频率：每次都做还是积累到一定程度再做？对应 lazy reduction 优化
  - 这里有个潜在创新点：**RVV 的 vmacc 天然支持 fused multiply-add，但 64-bit 模运算下溢出风险大**，需要专门设计
- **预期加速比**：8×~12×（因为消除了中间 store/load）
- **交付物**：RVV `mac_accumulate` 函数 + 与朴素 PMul+ADD 的对比报告

#### TODO 1.3 — 实现 RVV automorph（基于之前讨论的 $R=32, C=2048$ 配置）

> **注意**：这一步只有在 0.2 选了路径 B 或 C 时才必做。如果选路径 A，automorph 仅作为 backup 实现，优先级降低。

- **任务**：按我们之前讨论的"逐列处理 + 隐式转置写回"方案实现
- **设计要点**：
  - 预计算 $\mathbf{rowmap}$（长度 32，进 register）、$\boldsymbol{\delta}$（长度 $C$，进内存）、$\mathbf{colmap}$（长度 $C$，进内存）
  - 主循环：strided load 一列 → vadd.vx 修正 + vrgather 重排 → strided store 写回
  - 关键 microbenchmark：**先单独测 `vlse64.v` 在 stride=$C \cdot 8$ 下的实际吞吐**，这是整个方案的性能上限
  - 与 naive vrgather 全长版本对比
- **预期加速比**：naive 标量 baseline 的 5×~15×，与 Poseidon 论文 HFAuto 10× 是同一量级
- **风险点**：strided load 如果太慢，需要 fallback 到"完整转置 + 顺序访问 + 再转置回来"的方案
- **交付物**：RVV automorph 函数 + 与 naive 版本对比 + 不同 $N, k, \text{VLEN}$ 下的扫描曲线

---

### 阶段 2：matrix-vector 整体集成（2-3 周）

#### TODO 2.1 — 实现 HRF-MatVec 的 RVV 版本（如果选路径 A 或 C）

- **任务**：基于 1.1 和 1.2 的算子，组装出完整的 HRF-MatVec
- **设计要点**：
  - 离线预计算：把 LWE secret key 的所有 rotation 密文（NTT 域）预先生成，存到内存
  - 在线计算：纯 vmacc.vx 循环
  - 实现 fat / thin 矩阵的自适应 tiling，参考 Chameleon Figure 4
  - 用 `vsetvli` 动态调整 vl 处理非 2 幂次的 $n_{slot}$
- **测量**：vs. 标量 OpenFHE repacking、vs. 仅 NTT 加速版本（TCHES 风格）、vs. Chameleon 的 GPU 实现（定性对比）
- **预期端到端加速比**：在 repacking 上达到 8×~20×
- **交付物**：完整可调用的 RVV repacking 函数

#### TODO 2.2 — 实现 BSGS-MatVec 的 RVV 版本（如果选路径 B 或 C）

- **任务**：保留 Giant step 的 automorph，组装 Baby step（PMul + ADD）+ Giant step（automorph + ADD）
- **设计要点**：
  - Baby step 用 1.1/1.2 的算子
  - Giant step 用 1.3 的 automorph
  - 注意两层循环的 cache 友好性：把"哪些密文常驻 cache"作为优化变量
- **测量**：与 HRF 路径对比，找出 cross-over 点（在什么 $n_{slot}$ 时哪种更快）
- **交付物**：BSGS-MatVec 函数 + cross-over 分析

#### TODO 2.3 — Cache 布局优化（参考 TCHES 对 NTT 内存访问的分析方法）

- **任务**：针对 2.1 或 2.2 的实现，分析旋转密文的访问模式
- **设计要点**：
  - 类似 TCHES Figure 5/6 那样画出 cache 访问模式
  - 尝试两种数据布局：(a) 旋转密文按行优先 (b) 按列优先（每个密文的对应系数相邻）
  - 测 L2/L3 miss 率，找到对当前硬件最优的布局
- **预期收益**：5%~20% 的额外提升
- **交付物**：cache 分析报告 + 最优布局的实现

---

### 阶段 3：端到端 benchmark（2-3 周）

#### TODO 3.1 — 复现 TCHES 的全连接 NN benchmark，加入 MatVec 加速

- **任务**：在 TCHES 已有的 3 层 FC NN 基础上，把 matrix-vector 部分换成你的 RVV 实现
- **设置**：与 TCHES 一致的输入维度（8 到 1024），Chebyshev 近似的 ReLU，至少 1 次 bootstrapping
- **对比基线**：
  - 标量 OpenFHE
  - 仅 NTT 加速（TCHES baseline）
  - NTT + MatVec 加速（你的工作）
- **预期端到端加速比**：在 TCHES 1.69×~2.02× 的基础上，再额外 1.5×~2×
- **交付物**：端到端 benchmark 报告

#### TODO 3.2 — 私有推理 demo（小型 CNN 或决策树）

- **任务**：参考 Chameleon 的 K-means / Decision Tree，但选一个**只用 CKKS 就能跑完**的简化版本（避开 TFHE 和 scheme switching，避免战线过长）
- **建议任务**：加密 logistic regression 推理 或 加密 2 层小型 NN
- **测量**：端到端延迟、内存占用、与论文中已有 baseline 的对比
- **交付物**：完整的 demo 程序 + 性能报告

#### TODO 3.3 — 多平台评测（参考 TCHES 双平台范式）

- **任务**：把所有 benchmark 在两个平台上跑一遍
  - 平台 A：FPGA VPU（VLEN=2048 或更高，长向量）
  - 平台 B：gem5 模拟器（OoO 核 + 多种 VLEN 配置）
- **分析维度**：peak performance vs. 实测、不同 VLEN 下的 scaling、cache size 敏感性
- **交付物**：跨平台对比表，类似 TCHES Table 2

---

### 阶段 4：对比与论文产出（2 周）

#### TODO 4.1 — 完成与现有工作的对比表

- **任务**：构建类似 TCHES Table 3 的横向对比
- **对比对象**：
  - CPU baseline（OpenFHE 标量）
  - GPU 方案（Chameleon、Phantom 等，定性对比 + 单密文延迟优势）
  - FPGA/ASIC 方案（Poseidon、HEAX 等，资源消耗对比）
  - RVV 同类工作（TCHES 仅 NTT，你 NTT + MatVec）
- **核心论点**：在 *low-latency single-ciphertext* 场景下，RVV 比 GPU 有 offload 优势；vs. ASIC 没有定制硬件代价；vs. 仅 NTT 加速有更完整的端到端覆盖
- **交付物**：对比表 + 论点支持文档

#### TODO 4.2 — 论文骨架

- 标题方向：*"RVV-Accelerated Matrix-Vector Multiplication for Fully Homomorphic Encryption"* 或类似
- 结构建议：
  1. Intro：FHE 加速现状 + RVV 优势 + 现有 RVV 工作仅做 NTT 的不足
  2. Background：CKKS、matrix-vector、BSGS、RVV 简介
  3. 方法：HRF 路径的 RVV 适配、automorph 优化（可选）、cache 布局
  4. 实验：双平台、单算子、端到端
  5. 相关工作 + 结论
- **交付物**：论文 draft 大纲

---

