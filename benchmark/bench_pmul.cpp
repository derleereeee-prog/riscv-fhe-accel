// Measures one EvalMult(ciphertext, plaintext) at the configured params.
#include "common.h"
#include <iostream>

int main() {
    using namespace lbcrypto;

    std::cout << "bench_pmul: RING_DIM=" << RING_DIM
              << " BATCH_SIZE=" << BATCH_SIZE << std::endl;

    KeyPair<DCRTPoly> keys;
    auto cc = make_context(keys);

    cc->EvalMultKeyGen(keys.secretKey);
    std::cout << "EvalMultKeyGen done" << std::endl;

    std::vector<double> vec(BATCH_SIZE, 1.0);
    std::vector<double> diag(BATCH_SIZE, 0.5);
    auto pt   = cc->MakeCKKSPackedPlaintext(vec);
    auto ptd  = cc->MakeCKKSPackedPlaintext(diag);
    auto ct   = cc->Encrypt(keys.publicKey, pt);

    // Warmup
    auto r = cc->EvalMult(ct, ptd);

    // Measure
    const int N = 5;
    uint64_t start = rdcycle();
    for (int i = 0; i < N; i++)
        r = cc->EvalMult(ct, ptd);
    uint64_t end = rdcycle();

    std::cout << "BENCH EvalMult(ct,pt) cycles: " << (end - start) / N << std::endl;

    try {
        Plaintext dec;
        cc->Decrypt(keys.secretKey, r, &dec);
    } catch (...) {}
    return 0;
}
