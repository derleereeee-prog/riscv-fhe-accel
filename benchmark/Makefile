OPENFHE  := $(HOME)/NTT-RVV-project/openfhe-install-clang18
NTT_OBJ  := $(HOME)/NTT-RVV-project/ntt/bin/ntt-rvv.o
LIBSTDCXX := /usr/lib/gcc-cross/riscv64-linux-gnu/13

CXX := clang++
CXXFLAGS := \
    --target=riscv64-linux-gnu \
    -march=rv64gcv -mabi=lp64d \
    -fuse-ld=lld \
    --sysroot=/usr/riscv64-linux-gnu \
    -std=c++17 -O3 \
    -DMATHBACKEND=4 -DOPENFHE_VERSION=1.3.1 \
    -I$(OPENFHE)/include/openfhe \
    -I$(OPENFHE)/include/openfhe/core \
    -I$(OPENFHE)/include/openfhe/pke \
    -I$(OPENFHE)/include/openfhe/binfhe \
    -I$(OPENFHE)/include/openfhe/cereal

# small: quick workflow test (N=4096, batch=64)
SMALL_FLAGS := -DRING_DIM=4096 -DBATCH_SIZE=64

# full: paper-scale parameters (N=16384, batch=1024)
FULL_FLAGS  := -DRING_DIM=16384 -DBATCH_SIZE=1024

LDFLAGS := \
    -L$(OPENFHE)/lib \
    -L$(LIBSTDCXX) \
    -Wl,--start-group \
      -lOPENFHEpke_static -lOPENFHEbinfhe_static -lOPENFHEcore_static \
    -Wl,--end-group \
    -static -lstdc++ -lm

SRCS := bench_rotate.cpp bench_pmul.cpp bench_add.cpp
BINS_SMALL := $(SRCS:%.cpp=bin/small/%)
BINS_FULL  := $(SRCS:%.cpp=bin/full/%)

.PHONY: all small full clean

all: small

small: $(BINS_SMALL)
full:  $(BINS_FULL)

bin/small/%: %.cpp common.h $(NTT_OBJ)
	mkdir -p bin/small
	$(CXX) $(CXXFLAGS) $(SMALL_FLAGS) $< $(NTT_OBJ) $(LDFLAGS) -o $@

bin/full/%: %.cpp common.h $(NTT_OBJ)
	mkdir -p bin/full
	$(CXX) $(CXXFLAGS) $(FULL_FLAGS) $< $(NTT_OBJ) $(LDFLAGS) -o $@

clean:
	rm -rf bin
