# 项目架构与工程方案说明

> 写给自己看的完整技术梳理，解答"我到底在做什么、怎么做的、为什么这样做"

---

## 一、整个项目的分层结构

```
neural-net（应用层）
    ↓ 调用
OpenFHE（FHE 库）
    EvalMult / EvalBootstrap / EvalChebyshev
        ↓ 内部调用
    Key Switching（最热路径）
        ↓
    NativeVectorT::ModMulNoCheckEq   ← 模乘热点
        ↓
    NTT / INTT                       ← NTT 热点
        ↓
    底层：uint64_t 数组上的向量运算
```

TCHES 论文和本项目都在最底层动手，只是针对的热点不同：

| 热点 | 谁来加速 | 状态 |
|------|----------|------|
| NTT / INTT + Automorphism | TCHES 论文（`ntt-rvv.o`） | ✅ **已链接进当前 binary** |
| 向量模乘（ModMulNoCheckEq） | **本项目**（Barrett RVV） | ✅ 已编译进库，dcrtBits≤58 时激活 |

两者互补，**当前运行的 neural-net binary 同时包含两者**：
- `ntt-rvv.o` 通过 CMakeLists.txt 的 `link_libraries(... ../../ntt/bin/ntt-rvv.o)` 链接进来
- Barrett RVV 编译进了 OpenFHE 静态库（dcrtBits=59 时走标量 fallback）

---

## 二、TCHES 的做法 vs 本项目的做法

### TCHES 做法（NTT 加速）

```
openfhe.patch
  └─ 修改 OpenFHE 的 .cpp 源文件
       把内部 NTT() 调用改为 ntt_rvv()

ntt/ntt-rvv.h    ← RVV intrinsic 实现（头文件声明+实现）
ntt/ntt-rvv.o    ← 单独编译成目标文件（用 EPI clang 编译）

链接时：
  OpenFHE_STATIC_LIBRARIES + ntt-rvv.o → neural-net
```

**关键特点**：
- RVV 代码在独立 `.o` 里，和 OpenFHE 代码物理分离
- OpenFHE 修改量极小（只改了函数调用点），类结构、内存管理全部保留
- 需要重编译 OpenFHE（因为 patch 改了源码）

### 本项目做法（模乘加速）

```
mubintvecnat.h（OpenFHE 模板头文件）中修改：

  NativeVectorT::ModMulNoCheckEq() {
      ...
  #ifdef __riscv_vector
      if (n + 7 < 64) {
          barrett_mul_rvv_u64m1(...);  // RVV 快速路径
          return *this;
      }
  #endif
      // 原始标量路径（fallback）
      for (...) m_data[i].ModMulFastEq(...);
  }

barrett_rvv.h    ← RVV intrinsic 实现（头文件 inline）

重编译 OpenFHE → RVV 代码被编译进 .a 静态库
```

**关键特点**：
- RVV 代码 inline 在头文件里，编译进 OpenFHE 的 `.a` 库
- 修改量也极小（只在一个函数里加了 `#ifdef` 块）
- 同样需要重编译 OpenFHE

### 对 benchmark 说服力的影响

两种做法对 benchmark 说服力的影响**几乎相同**，理由如下：

- 两者都修改了 OpenFHE（TCHES 改 `.cpp`，本项目改 `.h`），只是修改位置不同
- 两者都只改了最小的必要部分，OpenFHE 的调度逻辑、密码学正确性全部保留
- 两者都用同一个 gem5 模型、同一套编译工具链

**关键区别：切换 RVV/标量的成本不同**

| | TCHES | 本项目 |
|---|---|---|
| 切换方式 | 链接时换 `.o` 文件（秒级） | 重编译 OpenFHE 库（~10分钟） |
| 得到"有 RVV"版 | 链接 `ntt-rvv.o` | 库用 `-march=rv64gcv` 编译（已完成） |
| 得到"无 RVV"版 | 链接标量 `ntt-scalar.o` | 注释掉 `#ifdef __riscv_vector` 块，重编译 |

这让 TCHES 在做 A/B 对比实验时更方便，但**对 benchmark 的公信力没有影响**，公信力来自对比是否公平，而不是 `.o` 在哪里。

---

## 三、为什么之前的测试没有内存问题？

### 之前跑的测试

```
pmul/test_barrett_rvv.c
pmul/test_mulhu.c
```

这两个测试**完全不用 OpenFHE**，也不用 CKKS。它们只是：

```c
uint64_t a[4096], b[4096], out[4096];
// 直接调用 barrett_mul_rvv_u64m1(...)
// 验证结果是否正确，测量 cycles
```

内存使用：3 个数组 × 4096 元素 × 8 字节 = **96 KB**，和内存限制完全无关。

### 现在跑的测试（neural-net）

这个测试使用了 **CKKS Bootstrapping**，bootstrapping 在运行前需要预生成大量密钥：

```
EvalBootstrapKeyGen() 生成的密钥包括：
  ├── EvalMult key（重线性化密钥）
  ├── EvalSum key
  └── 大量 Rotation key（每个旋转方向各一把）
       bootstrapping 的 CoeffsToSlots / SlotsToCoeffs 需要
       几十个旋转方向 → 几十把 key

每把 key 的大小（HYBRID key switch，numLargeDigits=3）：
  ≈ 3 × (L+1) × N × 8 bytes × 2
  ≈ 3 × 31 × 1024 × 8 × 2 ≈ 1.5 MB

总计（含 eval key、cipher 本身等）：
  N=1024 → ~8 GB
  N=4096 → ~32 GB（无法在 gem5 里跑）
```

这是 CKKS bootstrapping 的固有特性，和 RVV 代码完全无关。

---

## 四、RVV 激活条件

本项目的 RVV 代码有一个运行时 guard：

```cpp
int64_t n = mv.GetMSB() - 2;   // mv = 模数
if (n + 7 < 64) {               // 等价于：模数 < 2^58
    // 走 RVV 路径
}
```

| dcrtBits（素数位数） | n | n+7 | 满足条件？ | 走哪条路 |
|---|---|---|---|---|
| 59（当前 neural-net） | 57 | 64 | ❌ 64 < 64 假 | 标量 |
| 58 | 56 | 63 | ✅ | **RVV** |
| 50（单元测试用的量级） | 48 | 55 | ✅ | **RVV** |

**当前 neural-net 用的 dcrtBits=59，RVV 路径不会被激活。** 跑出来的 cycle 数是纯标量 baseline。

---

## 五、完整 benchmark 计划

要得到有效的加速比对比，需要跑两次 gem5：

### Run A（Baseline，标量）
- 修改 `mubintvecnat.h`：把 `#ifdef __riscv_vector` 块注释掉
- 重编译 OpenFHE + neural-net
- 用 `dcrtBits=50`（或 ≤58 均可）跑 gem5
- 记录 `Cycles: X_scalar`

### Run B（RVV，本项目）
- 恢复 `mubintvecnat.h` 的 `#ifdef` 块
- 重编译 OpenFHE + neural-net
- 同样 `dcrtBits=50` 跑 gem5
- 记录 `Cycles: X_rvv`

加速比 = X_scalar / X_rvv

### 现在正在跑的
- dcrtBits=59，标量路径，作为参考（但因为 RVV 根本没激活，这次和 Run A 的对比意义不大）
- 主要目的是确认程序能跑完，确认 gem5 配置没问题

---

## 六、当前文件状态

```
pmul/barrett_rvv.h              ← RVV Barrett 实现（核心）✅
pmul/test_barrett_rvv.c         ← 单元测试，ALL PASS，4.5 cyc/elem ✅
pmul/test_mulhu.c               ← 低级单元测试 ✅

openfhe-development/src/core/include/math/hal/intnat/
  barrett_rvv.h                 ← 同上（拷贝）✅
  mubintvecnat.h                ← 已修改（含 RVV 快速路径）✅

openfhe-install/                ← 重编译后的安装目录 ✅

example-ofhe-app/
  neural-net.cpp                ← dcrtBits=59，当前 benchmark（RVV 未激活）
  neural-net-small.cpp          ← dcrtBits=50，待用（RVV 会激活）⏳

gem5-model/main.py              ← 内存已改为 8GiB，SimdDiv FU 已加 ✅
```
