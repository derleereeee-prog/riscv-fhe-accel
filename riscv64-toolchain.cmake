set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_C_COMPILER   clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET   riscv64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-linux-gnu)

set(RISCV_SYSROOT /usr/riscv64-linux-gnu)

# No --sysroot: Ubuntu's riscv64-linux-gnu package uses absolute paths in
# linker scripts that resolve correctly only without a sysroot override.
# Pass include/library paths explicitly instead.
set(CMAKE_C_FLAGS_INIT
    "-march=rv64gcv -mabi=lp64d -I${RISCV_SYSROOT}/include")
set(CMAKE_CXX_FLAGS_INIT
    "-march=rv64gcv -mabi=lp64d -I${RISCV_SYSROOT}/include")
# -fuse-ld=lld is a linker driver flag; pass only at link time
set(CMAKE_EXE_LINKER_FLAGS_INIT    "-fuse-ld=lld -L${RISCV_SYSROOT}/lib -Wl,-rpath-link,${RISCV_SYSROOT}/lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld -L${RISCV_SYSROOT}/lib -Wl,-rpath-link,${RISCV_SYSROOT}/lib")

# Skip try_run and try to compile as static library to avoid
# executing RISC-V binaries on the x86 host during cmake checks.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH ${RISCV_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
