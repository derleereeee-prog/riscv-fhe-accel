CC = clang

SYSROOT = /usr/riscv64-linux-gnu
C_FLAGS = -O2 \
          --target=riscv64-linux-gnu \
          -march=rv64gcv \
          -mabi=lp64d \
          -fuse-ld=lld \
          --sysroot=$(SYSROOT)

SRC = ntt-rvv.cpp

.PHONY: all
all: bin/ntt-rvv.o bin/test

bin/ntt-rvv.o: $(SRC)
	mkdir -p bin
	$(CC) $(C_FLAGS) $^ -o $@ -c

bin/test: test.cpp bin/ntt-rvv.o
	mkdir -p bin
	$(CC) $(C_FLAGS) -static $^ -o $@ -lm

.PHONY: clean
clean:
	rm -rf bin
