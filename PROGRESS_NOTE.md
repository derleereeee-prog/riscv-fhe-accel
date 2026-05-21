# Barrett RVV 向量化进度记录

> 最后更新：2026-05-21（第三次会话）

---

## 当前状态一览

| 模块 | 状态 |
|------|------|
| `pmul/barrett_rvv.h` — 声明头文件 | ✅ 纯声明，无内联实现 |
| `pmul/barrett_rvv.c` — RVV 实现 | ✅ 独立 .c，编译为 `barrett-rvv.o` |
| `pmul/barrett_scalar.c` — 标量对照 | ✅ 独立 .c，编译为 `barrett-scalar.o` |
| `pmul/test_barrett_rvv.c` — 单元测试 | ✅ 全部通过 |
| OpenFHE `mubintvecnat.h` — patch | ✅ 调用声明，链接时注入 .o |
| `barrett.patch` — 可复现 patch | ✅ 从 clean checkout 可直接 apply |
| `example-ofhe-app/build/neural-net-small` | ✅ dcrtBits=50，RVV路径激活 |
| `example-ofhe-app/build-scalar/neural-net-small` | ✅ dcrtBits=50，标量对照 |
| gem5 全程基准测试 | ❌ **暂停** — 整个 neural-net 模拟太慢且反复崩溃 |
| Barrett micro-benchmark（独立） | ⏳ **下一步** |

---

## 第一次会话完成的工作（摘要）

### 1. 修复了 barrett_rvv 批量正确性问题

**根本原因**：`scalar_barrett_batch` 的 `__attribute__((optimize("O0")))` 是 GCC 扩展，**clang 静默忽略**。clang 以 `-O2 -march=rv64gcv` 编译时自动向量化了本应是"纯标量"的参考实现，导致 scalar 和 RVV 都出错，产生大量误报 mismatch（37%）。

**修复**：
- `scalar_barrett` 加 `__attribute__((noinline))`
- `scalar_barrett_batch` 改为 `__attribute__((optnone))`（clang 原生 O0 属性）

**结果（gem5 VLEN=512）**：ALL TESTS PASSED (0 errors)

---

### 2. OpenFHE 集成（已完成）

`ModMulNoCheckEq`（`mubintvecnat.h`）加入 RVV 快速路径：

```cpp
#if defined(__riscv_vector)
    if constexpr (std::is_same_v<typename IntegerType::Integer, uint64_t> &&
                  sizeof(IntegerType) == sizeof(uint64_t)) {
        int64_t n{mv.GetMSB() - 2};
        if (n + 7 < 64) {          // ← 激活条件：模数 ≤ 58-bit
            uint64_t*       ap = &m_data[0].m_value;
            const uint64_t* bp = &b.m_data[0].m_value;
            barrett_mul_rvv_u64m1(ap, bp, ap, size,
                                  mv.m_value, mu.m_value,
                                  (int)n, (int)(n + 7),
                                  (int)(64 - n), (int)(57 - n));
            return *this;
        }
    }
#endif
```

已修改并重编译：
- `openfhe-development/src/core/include/math/hal/intnat/mubintvecnat.h`
- `openfhe-install/include/openfhe/core/math/hal/intnat/mubintvecnat.h`
- OpenFHE 静态库（`openfhe-development/build` 下 `make` + `make install`）
- `example-ofhe-app/build/neural-net`（静态链接）

---

## 第二次会话（本次）

### 问题诊断：gem5 崩溃

**现象**（上一次运行输出）：
```
CKKS scheme is using ring dimension 1024
bootstrap evaluated
Memory Usage: 2768556 KBytes        ← 约 2.7 GiB
Program aborted at tick 264925512000
```

**根本原因**：`gem5-model/main.py` 第 73 行只分配了 2 GiB 模拟内存，程序用了 ~2.7 GiB → 越界崩溃。**和我们写的 RVV 代码无关。**

**修复**（已做）：
```python
# gem5-model/main.py 第 73 行
system.mem_ranges = [AddrRange("8GiB")]   # 原来是 2GiB
```

### 当前：gem5 正在重新运行

```bash
~/gem5.opt ~/NTT-RVV-project/gem5-model/main.py \
           ~/NTT-RVV-project/example-ofhe-app/build/neural-net
```

**重要提醒 — RVV 路径激活条件**：

| 参数 | 值 | n = MSB-2 | n+7 | n+7 < 64？ | RVV 激活？ |
|------|-----|-----------|-----|-----------|-----------|
| `dcrtBits=59`（当前） | ~59-bit 素数 | 57 | 64 | ❌ | 否，走 scalar |
| `dcrtBits=50`（可测试） | ~50-bit 素数 | 48 | 55 | ✅ | 是，走 RVV |
| `dcrtBits=58` | ~58-bit 素数 | 56 | 63 | ✅ | 是，走 RVV |

**结论**：neural-net 当前用 `dcrtBits=59`，RVV 路径不会被激活。本次 gem5 运行给出的是纯标量基准。若要测 RVV 加速，需要用 ≤ 58-bit 模数的参数重新编译 neural-net。

---

## 第二次会话 gem5 崩溃修复记录

### 崩溃 1：内存不足（已修复）
- **现象**：`Program aborted`，Memory Usage 2768556 KBytes（~2.7 GiB）
- **原因**：`gem5-model/main.py` 行 73：`AddrRange("2GiB")`
- **修复**：改为 `AddrRange("8GiB")`

### 崩溃 2：缺少 SimdDiv 功能单元（已修复）
- **现象**：`panic: CPU cannot execute op_class: 23 but did not trigger a fault`
- **原因**：op_class 23 = `SimdDiv`（向量整数除法），`gem5-model/func_units.py` 的 `CustomSIMD` 里没有这个 OpDesc
- **修复**：在 `func_units.py` 的 `CustomSIMD.opList` 中加入：
  ```python
  OpDesc(opClass="SimdDiv", opLat=24, pipelined=False),
  ```
- **当前状态**：⏳ gem5 已重新启动（后台运行中）

---

## 第三次会话（2026-05-21）

### 架构重构：Barrett 从 inline 改为链接时注入（已完成）

**动机**：和 TCHES 论文对 NTT 的处理方式一致——RVV 实现编译为独立 `.o`，patch 只改调用点和 include，切换 RVV/scalar 只需换一个文件，无需重编译 OpenFHE。

**改动内容**：

| 文件 | 变化 |
|------|------|
| `pmul/barrett_rvv.h` | 从完整实现改为纯声明（extern "C"） |
| `pmul/barrett_rvv.c` | 新建：RVV 实现，编译为 `barrett-rvv.o` |
| `pmul/barrett_scalar.c` | 新建：标量实现，同接口，编译为 `barrett-scalar.o` |
| `pmul/Makefile` | 增加两个 .o 目标 |
| `openfhe-development/.../mubintvecnat.h` | 去掉 `#ifdef __riscv_vector`，改为无条件 include 声明头 |
| `openfhe-install/.../mubintvecnat.h` | 同上，OpenFHE 已重新编译 |
| `example-ofhe-app/CMakeLists.txt` | 增加 `-DBARRETT_SCALAR=ON` 选项切换 |
| `barrett.patch` | 新建：从 clean OpenFHE checkout 可直接 apply |

**切换方式（无需重编译 OpenFHE）**：
```bash
# RVV 版本（默认）
cmake .. -DBUILD_STATIC=ON
# 标量对照版本
cmake .. -DBUILD_STATIC=ON -DBARRETT_SCALAR=ON
```

### neural-net-small 已编译（dcrtBits=50，RVV路径激活）

- `example-ofhe-app/neural-net-small.cpp`：dcrtBits=50, firstMod=51, N=1024
- n = 48, n+7 = 55 < 64 → RVV Barrett 路径被激活 ✅
- `build/neural-net-small`：链接 barrett-rvv.o
- `build-scalar/neural-net-small`：链接 barrett-scalar.o

### gem5 全程测试：反复崩溃，已暂停

**崩溃类型**：`panic: Dependency graph 19 (vector) (flat: 531) not empty!`
（`src/cpu/o3/inst_queue.cc:1415`）

- 两个进程（rvv 和 scalar）都在相同位置崩溃
- 崩溃发生在 `bootstrap evaluated` 之后（tick ≈ 776B），说明不是启动问题
- 根本原因：gem5 O3 CPU 的向量寄存器依赖图（dependency graph）未能正确清理某条 RVV 指令的寄存器状态
- 由于 rvv 和 scalar 版本都崩，问题来自 NTT RVV 代码，而非 Barrett

**已尝试修复**：
- 尝试将 `numPhysVecRegs` 从 384 增大到 768（未执行，模拟太慢放弃）

**决策**：暂停全程 neural-net gem5 测试，改用独立 Barrett micro-benchmark。

---

## 下一步

### 计划：独立 Barrett micro-benchmark

在 `pmul/` 下写 `bench_barrett.c`：
- 直接循环调用 `barrett_mul_rvv_u64m1`（不跑 OpenFHE）
- 用 `rdcycle` CSR 指令测量 cycles
- 编译成 RISC-V static ELF，gem5 只需模拟几千次乘法（几分钟 vs 几小时）
- 参考 OpenPEGASUS 的 benchmark 结构（待查看）

---

## 待做事项

- [ ] 查看 OpenPEGASUS 的 benchmark 实现方式（用户提供 GitHub 链接）
- [ ] 写 `pmul/bench_barrett.c`（独立 micro-benchmark）
- [ ] 编译并在 gem5 上跑（RVV vs scalar 两个版本）
- [ ] 记录 cycles/element 对比数据

---

## 关键文件索引

```
gem5-model/main.py               ← 已改 AddrRange("8GiB")
example-ofhe-app/neural-net.cpp  ← dcrtBits=59，RVV 路径未激活

pmul/
  barrett_rvv.h                  ← RVV 实现（核心）
  test_barrett_rvv.c             ← 正确性+性能测试（全部 PASS）
  test_mulhu.c                   ← 低级单元测试

openfhe-development/src/core/include/math/hal/intnat/
  barrett_rvv.h                  ← 同 pmul/barrett_rvv.h
  mubintvecnat.h                 ← 已修改，含 RVV 快速路径

openfhe-install/include/openfhe/core/math/hal/intnat/
  barrett_rvv.h                  ← 同上
  mubintvecnat.h                 ← 已修改，库已重编译
```

---

## 已确认的技术事实

- gem5 v24.1.0.2：`vmulhu_vx`（64-bit SEW）有 bug，本实现全用 `.vv` + `vmv_v_x` 广播
- gem5 中 `__uint128_t %` 有 bug；`__uint128_t /` 正常
- clang 的 `optimize("O0")` 无效，应用 `__attribute__((optnone))`
- VLEN=512，m1=8 元素/向量，barrett_mul_rvv_u64m1 每批约 36 cycles（4.5 cyc/elem）
- RVV 激活条件：`n + 7 < 64`，即模数 MSB ≤ 58（模数 < 2^58）
