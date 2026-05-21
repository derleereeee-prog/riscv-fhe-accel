# 工具链问题分析与解决方案

## 目标
用标准 clang 18 + lld（不用 EPI toolchain）编译 OpenFHE + example-ofhe-app，产出 RISC-V 目标文件，能在 gem5 上运行。

---

## 已确认可行的事

1. **NTT 单独编译和链接完全没问题**（`ntt/Makefile`）：
   ```
   clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d -fuse-ld=lld --sysroot=/usr/riscv64-linux-gnu
   ```
   编译目标文件 + 静态链接 test 二进制都正常。

2. **手动一行命令交叉编译 C 程序没问题**：
   ```
   clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d -fuse-ld=lld simple.c -o output
   ```
   产出合法 RISC-V ELF。

---

## 遇到的问题

### 问题 A：cmake 检测架构失败
**现象**：cmake 报 "Architecture is x86_64"，`Check size of __int128 - failed`，最终 `Cannot support NATIVE_SIZE == 64`。

**原因**：cmake 的 `check_type_size(__int128)` 等测试需要编译 + **运行**目标程序。RISC-V 二进制在 x86 宿主上无法运行，所以所有 try_run 类型的测试都失败。cmake 用这些结果来推断目标架构。

**已试方案**：
- `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY`：让 cmake 只编译不链接不运行。但这样 cmake 没法检测目标 ABI，就回退到宿主架构 x86_64，导致 `__int128` 之类的判断走错分支。

### 问题 B：lld + Ubuntu RISC-V 包的 libc.so 链接脚本冲突
**现象**：`ld.lld: cannot find /usr/riscv64-linux-gnu/lib/libc.so.6 inside /usr/riscv64-linux-gnu`

**原因**：Ubuntu 的 `/usr/riscv64-linux-gnu/lib/libc.so` 是 GNU 链接脚本，里面写的是绝对路径：
```
GROUP ( /usr/riscv64-linux-gnu/lib/libc.so.6 ... )
```
lld 用 `--sysroot=/usr/riscv64-linux-gnu` 时，会把这个绝对路径再加前缀，变成：
```
/usr/riscv64-linux-gnu/usr/riscv64-linux-gnu/lib/libc.so.6  <- 不存在
```
Ubuntu 这套包本来就是设计给不用 sysroot 的场景的（直接把 sysroot 内容装在 `/usr/riscv64-linux-gnu/`）。

**已试方案**：去掉 `--sysroot`，改用 `-I/usr/riscv64-linux-gnu/include` + `-L/usr/riscv64-linux-gnu/lib`。但 cmake 的 check_include_file 测试因为同样的原因（不知道去哪找头文件）失败了。

### 问题 C（EPI 方案下）：ISA 属性不兼容
**现象**（仅 EPI 方案）：用 EPI clang 编译 OpenFHE，用标准 clang 编译 ntt-rvv.o，再用 EPI 的 GNU ld 链接：
```
riscv64-unknown-linux-gnu-ld: unknown z ISA extension `zve'
```
**原因**：EPI GNU binutils 是 2021 年版本，不认识 RVV 1.0 正式规范的 `zve*` / `zvl*` sub-extension 名称（这些名称是后来才标准化的）。

---

## 候选解决方案

### 方案 1（推荐）：创建正确的 sysroot 结构（symlink）
**思路**：Ubuntu RISC-V 包装好的东西在 `/usr/riscv64-linux-gnu/`，linker script 里写的路径前缀也是 `/usr/riscv64-linux-gnu/lib/`。如果我们把 sysroot 设为 `~/riscv-sysroot`，然后创建：
```bash
mkdir -p ~/riscv-sysroot/usr
ln -s /usr/riscv64-linux-gnu ~/riscv-sysroot/usr/riscv64-linux-gnu
```
那么 lld 用 `--sysroot=~/riscv-sysroot` 时：
- 找 `-lc`：查 `<sysroot>/usr/riscv64-linux-gnu/lib/libc.so` → symlink → 真实文件 ✓
- linker script 里 `/usr/riscv64-linux-gnu/lib/libc.so.6` → `<sysroot>/usr/riscv64-linux-gnu/lib/libc.so.6` → symlink → 真实文件 ✓

对 cmake 的好处：有了完整的 sysroot，cmake 的 `check_include_file` 和 `check_type_size` 就能在编译时找到头文件，不需要 try_run 就能通过。

**还需要解决**：cmake 的 `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` + `HAVE_INT128=1` 等变量覆盖，让 cmake 知道目标是 64 位且支持 `__int128`。

### 方案 2：只用 clang 18，不用 cmake 的交叉编译模式
**思路**：不设 `CMAKE_SYSTEM_NAME`，让 cmake 认为这是本机编译，传入正确的编译器和标志。cmake 的测试会通过（因为它认为是 x86），然后用实际的 RISC-V 标志编译。
**缺点**：cmake 会生成错误的平台检测结果（x86 的 `__int128` 大小等），可能导致 OpenFHE 配置错误。实际上是"骗"cmake。对于 64 位 RISC-V 和 64 位 x86 来说，大部分类型大小是一样的，可能凑巧能工作。

### 方案 3：手动 Makefile 编译 OpenFHE（不用 cmake）
**思路**：完全跳过 cmake，手写 Makefile 或 shell 脚本直接调用 clang 编译 OpenFHE 的所有 .cpp 文件，再 llvm-ar 打包成静态库。
**缺点**：OpenFHE 有大量条件编译和依赖，cmake 处理很多细节，手写 Makefile 容易出错。

---

## 结论

**推荐方案 1**，关键步骤：
1. 创建 sysroot symlink（2行命令）
2. 更新 toolchain 文件用 `--sysroot=~/riscv-sysroot`
3. cmake 配置加 `-DHAVE_INT128=1`，也许还需要几个 override
4. 重新跑 cmake + make

需要你决定：继续尝试方案 1，还是用其他方案？
