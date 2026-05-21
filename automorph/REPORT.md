# Automorphism RVV 内核开发报告

## 一、背景与需求

OpenFHE 的 BSGS 矩阵-向量乘法每次调用会执行约 1023 次 `EvalRotate`。
性能分析（`m5out/stats.txt`）显示：

- EvalRotate 占总 BSGS 耗时 **69.2%**
- 其中 `AutomorphismTransform`（多项式下标置换）是最后一公里瓶颈

目标：用 RVV 向量指令加速 N=4096、64 位元素多项式的 Automorphism 变换，
使其正确性与标量实现一致，性能尽可能高。

---

## 二、新增文件列表

| 文件 | 说明 |
|------|------|
| `automorph/automorph.h` | 三个实现的统一头文件 + `rdcycle()` 内联汇编 |
| `automorph/automorph_scalar.c` | 标量参考实现（正确性基准） |
| `automorph/automorph_naive_rvv.c` | 朴素 RVV 实现（功能验证对比） |
| `automorph/automorph_rvv.c` | 2D HFAuto 分解 RVV 优化实现（本次重点） |
| `automorph/bench_automorph.c` | 基准测试驱动程序 |
| `automorph/Makefile` | 交叉编译 + gem5 运行规则 |

---

## 三、算法说明

### 3.1 Automorphism 变换定义

给定奇数 `k`，将 N 元素多项式 `src` 变换为 `dst`：

```
dst[(i * k) % N] = src[i]    对所有 i ∈ [0, N)
```

### 3.2 标量实现（`automorph_scalar.c`）

直接遍历，O(N) 时间：

```c
for (uint32_t i = 0; i < N; i++)
    dst[((uint64_t)i * k) % N] = src[i];
```

### 3.3 朴素 RVV（`automorph_naive_rvv.c`）

以目标索引分组，每 8 个目标元素一批：
1. 计算逆映射 `k_inv = k^{-1} mod N`
2. 对每批目标元素，用 `vmul + vand` 计算源字节偏移
3. `vloxei64`（索引聚集加载）从 src 取值
4. `vse64` 顺序写入 dst

**问题**：`mod_inverse` 暴力求解耗时；对大 k 乘法 + 散列开销大，性能波动剧烈。

### 3.4 优化 RVV（`automorph_rvv.c`）：2D HFAuto 分解

将 N 元素多项式视为 **R×C 矩阵**（R=8, C=N/8=512）：

```
元素 src[i*C + j]  →  矩阵位置（行 i，列 j）
目标位置（行 I，列 J）：
  J = (j * k) mod C              -- 列映射（仅与 j 有关）
  I = (i*k + floor(j*k/C)) mod R -- 行映射（与 i、j 均有关）
```

每列 j 独立处理（共 C=512 次迭代，每次操作 8 个元素）：

1. 对列 j 做**索引加载**（vloxei64），取出 8 个行元素
2. 加载预计算逆置换索引 `inv_idx_table[delta[j]]`（vle64）
3. `vrgather.vv` 做行内置换
4. 对目标列 colmap[j] 做**索引散射存储**（vsoxei64）

**预计算表**（仅在 k 或 N 变化时重新计算）：

| 表名 | 大小 | 内容 |
|------|------|------|
| `rowmap_table[8]` | 8 项 | `rowmap[i] = (i*k) mod 8` |
| `inv_idx_table[8][8]` | 64 项 | `inv_idx_table[d][i] = rowmap[(i-d+8) mod 8]` |
| `stride_offsets[8]` | 8 项 | `[0, C×8, 2C×8, …, 7C×8]` 字节偏移 |
| `delta_arr[C]` | 512 项 | `delta_arr[j] = floor(j*k/C) mod 8` |
| `colmap_arr[C]` | 512 项 | `colmap_arr[j] = (j*k) mod C` |

---

## 四、调试过程与问题定位

开发经历三次迭代，每次暴露一个不同的 bug。

### 第一版：方向错误（逆置换 vs 正置换）

**现象**：k=7、k=4095 通过，其余 k 失败  
**原因**：`vrgather` 需要**逆置换**索引 `I⁻¹[d]`，而非正置换 `I[i]`。
只有 k≡7(mod 8) 时正置换与逆置换恰好相等，所以这两个 k 通过。  
**修复**：预计算 `inv_idx_table[delta][row]`，用 `vle64` 加载。

### 第二版：`vid.v` 在 gem5 中输出错误值

**现象**：所有 k 全部失败，输出 UINT64_MAX  
**原因**：用 `__riscv_vid_v_u64m1()` 生成 `[0,1,...,7]` 作为基础索引向量，
而 gem5 v24.1.0.2 的 `vid.v` 实现存在已知 bug，输出错误值。  
**修复**：彻底弃用 `vid.v`，改为从静态数组 `vle64` 加载预计算的逆置换表。

### 第三版：`vsse64.v` 步长=4096 字节时存储失败

**现象**：k=1 通过，k=3/5/7/31/257 失败（got=UINT64_MAX），k=4095 通过  
**定位方法**：编译 N=64 的小尺寸版本（步长=64 字节），gem5 运行后所有 k 全部通过。
→ 确认是**步长=4096（恰好等于页大小）**触发了 gem5 的 bug，而非算法逻辑错误。  
**原因**：gem5 v24.1.0.2 中 `vsse64.v` 在步长等于页大小（4096 字节）时，
目标地址与源地址不重叠的情况下存储行为异常。  
**修复**：将 `vlse64`/`vsse64`（步长式加载/存储）全部替换为
`vloxei64`/`vsoxei64`（索引式加载/存储），使用预计算字节偏移向量。

```c
// 旧：步长式，有 gem5 bug
vuint64m1_t v_col = __riscv_vlse64_v_u64m1(src + j, stride, vl);
__riscv_vsse64_v_u64m1(dst + colmap_arr[j], stride, v_perm, vl);

// 新：索引式，绕过 gem5 bug
vuint64m1_t v_src_offs = __riscv_vadd_vx_u64m1(v_strides, j * 8, vl);
vuint64m1_t v_col = __riscv_vloxei64_v_u64m1(src, v_src_offs, vl);
vuint64m1_t v_dst_offs = __riscv_vadd_vx_u64m1(v_strides, colmap_arr[j] * 8, vl);
__riscv_vsoxei64_v_u64m1(dst, v_dst_offs, v_perm, vl);
```

---

## 五、测试方案

### 编译

```bash
cd automorph
make benchmark    # 交叉编译，生成 bin/bench_automorph
```

工具链：clang 18，`--target=riscv64-linux-gnu`，`-march=rv64gcv`，静态链接

### 运行

```bash
make run    # 等价于：~/gem5.opt .../main.py bin/bench_automorph
```

平台：gem5 v24.1.0.2，O3CPU，VLEN=512 bits，RVV 1.0

### 测试用例

测试 7 个奇数 k 值，覆盖边界和典型场景：

| k | 特点 |
|---|------|
| 1 | 恒等映射 |
| 3 | 小奇数 |
| 5 | 小奇数 |
| 7 | k≡7(mod 8)，历史上容易误判通过 |
| 31 | 中等，k≡7(mod 8) 但 delta 分布复杂 |
| 257 | 大值 |
| 4095 | N-1，最大有效 k |

每个 k 的验证逻辑：将 automorph_rvv 输出逐元素对比标量 `automorph_scalar` 的输出，
任何不匹配即报 FAIL 并打印首个错误位置和期望/实际值。

每个 k 重复运行 5 次，取最小周期数（best-of-5）以排除 gem5 的偶发抖动。

---

## 六、最终性能对比

测试参数：N=4096，VLEN=512，gem5 O3CPU

```
bench_automorph: N=4096 (R=8, C=512, VLEN=512)

k=1:
  scalar       : 86,031 cycles
  naive_rvv    :  2,106 cycles  [PASS]
  automorph_rvv:  2,617 cycles  [PASS]   加速比: 32.8×

k=3:
  scalar       : 86,031 cycles
  naive_rvv    : 32,166 cycles  [PASS]
  automorph_rvv:  2,629 cycles  [PASS]   加速比: 32.7×

k=5:
  scalar       : 86,031 cycles
  naive_rvv    : 38,171 cycles  [PASS]
  automorph_rvv:  2,638 cycles  [PASS]   加速比: 32.6×

k=7:
  scalar       : 86,031 cycles
  naive_rvv    : 40,753 cycles  [PASS]
  automorph_rvv:  2,664 cycles  [PASS]   加速比: 32.2×

k=31:
  scalar       : 86,031 cycles
  naive_rvv    : 35,556 cycles  [PASS]
  automorph_rvv:  2,612 cycles  [PASS]   加速比: 32.9×

k=257:
  scalar       : 86,031 cycles
  naive_rvv    : 44,372 cycles  [PASS]
  automorph_rvv:  2,619 cycles  [PASS]   加速比: 32.8×

k=4095:
  scalar       : 86,031 cycles
  naive_rvv    : 47,176 cycles  [PASS]
  automorph_rvv:  2,633 cycles  [PASS]   加速比: 32.6×
```

### 横向对比总结

| 实现 | k=1 周期 | k=4095 周期 | 性能稳定性 | 对 scalar 加速 |
|------|---------|------------|----------|--------------|
| scalar | 86,031 | 86,031 | 稳定 | 1× |
| naive_rvv | 2,106 | 47,176 | **极不稳定**（23× 波动） | 1.8×–40× |
| **automorph_rvv** | **2,617** | **2,633** | **极稳定（<2% 波动）** | **~32.7×** |

**automorph_rvv 相比 naive_rvv 的优势**：
- k=4095 时快 **18×**（47,176 vs 2,633）
- 性能与 k 值无关（2D 分解使每列操作量固定为 8 个元素）

---

## 七、当前状态与后续工作

**已完成**：automorph_rvv 内核，通过所有 k 值的正确性验证，性能 ~32.7× vs scalar

**待完成**：
1. 将此内核集成进 OpenFHE 的 `AutomorphismTransform` 替换路径
2. 运行端到端 BSGS 基准，测量实际 EvalRotate 加速效果
3. 通过 `analyze.py` 量化整体 CKKS 推理加速比
