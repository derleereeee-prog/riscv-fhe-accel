SYSROOT  := /usr/riscv64-linux-gnu
LIBSTDCXX :=  # not needed: pure C

CC      := clang
CFLAGS  := -O2 \
            --target=riscv64-linux-gnu \
            -march=rv64gcv \
            -mabi=lp64d \
            -fuse-ld=lld \
            --sysroot=$(SYSROOT) \
            -std=c11

GEM5    := $(HOME)/gem5.opt
GEM5CFG := $(HOME)/NTT-RVV-project/gem5-model/main.py

SRCS_LIB := automorph_scalar.c automorph_naive_rvv.c automorph_rvv.c
OBJS_LIB := $(SRCS_LIB:%.c=bin/%.o)
BIN      := bin/bench_automorph

.PHONY: all benchmark run clean

all: benchmark

benchmark: $(BIN)

bin/%.o: %.c automorph.h
	mkdir -p bin
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJS_LIB) bench_automorph.c automorph.h
	mkdir -p bin
	$(CC) $(CFLAGS) -static \
	    $(OBJS_LIB) bench_automorph.c \
	    -o $(BIN) -lm

run: $(BIN)
	$(GEM5) $(GEM5CFG) $(BIN)

clean:
	rm -rf bin
