#pragma once
#include "openfhe.h"

// Parameterised at compile time; override with -DRING_DIM=... -DBATCH_SIZE=...
#ifndef RING_DIM
#define RING_DIM 4096
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 64
#endif
#ifndef MULT_DEPTH
#define MULT_DEPTH 5
#endif
#ifndef SCALE_MOD
#define SCALE_MOD 50
#endif

static inline uint64_t rdcycle() {
    uint64_t v;
    asm volatile("rdcycle %0" : "=r"(v));
    return v;
}

// Build a CKKS crypto context with the compile-time parameters.
// Returns the context and keypair via out-params.
inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly>
make_context(lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys)
{
    using namespace lbcrypto;
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(MULT_DEPTH);
    p.SetScalingModSize(SCALE_MOD);
    p.SetBatchSize(BATCH_SIZE);
    p.SetRingDim(RING_DIM);
    p.SetSecurityLevel(HEStd_NotSet);  // disable security check for benchmarking

    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "GenCryptoContext done" << std::endl;
    keys = cc->KeyGen();
    std::cout << "KeyGen done" << std::endl;
    return cc;
}
