SYSROOT := /usr/riscv64-linux-gnu
CC      := clang
CFLAGS  := -O2 \
            --target=riscv64-linux-gnu \
            -march=rv64gcv \
            -mabi=lp64d \
            -fuse-ld=lld \
            --sysroot=$(SYSROOT) \
            -std=c11

# Scalar build: same flags but no vector extension
CFLAGS_SCALAR := -O2 \
                  --target=riscv64-linux-gnu \
                  -march=rv64gc \
                  -mabi=lp64d \
                  --sysroot=$(SYSROOT) \
                  -std=c11

GEM5    := $(HOME)/gem5.opt
GEM5CFG := $(HOME)/NTT-RVV-project/gem5-model/main.py

BIN         := bin/test_barrett_rvv
BIN_MULHU   := bin/test_mulhu

# Linkable object files for OpenFHE injection
OBJ_RVV    := bin/barrett-rvv.o
OBJ_SCALAR := bin/barrett-scalar.o

.PHONY: all run run_mulhu clean

all: $(BIN) $(BIN_MULHU) $(OBJ_RVV) $(OBJ_SCALAR)

$(OBJ_RVV): barrett_rvv.c barrett_rvv.h
	mkdir -p bin
	$(CC) $(CFLAGS) -c barrett_rvv.c -o $(OBJ_RVV)

$(OBJ_SCALAR): barrett_scalar.c barrett_rvv.h
	mkdir -p bin
	$(CC) $(CFLAGS_SCALAR) -c barrett_scalar.c -o $(OBJ_SCALAR)

$(BIN): test_barrett_rvv.c barrett_rvv.h $(OBJ_RVV)
	mkdir -p bin
	$(CC) $(CFLAGS) -static test_barrett_rvv.c $(OBJ_RVV) -o $(BIN)

$(BIN_MULHU): test_mulhu.c
	mkdir -p bin
	$(CC) $(CFLAGS) -static test_mulhu.c -o $(BIN_MULHU)

run: $(BIN)
	$(GEM5) $(GEM5CFG) $(BIN)

run_mulhu: $(BIN_MULHU)
	$(GEM5) $(GEM5CFG) $(BIN_MULHU)

clean:
	rm -rf bin m5out
