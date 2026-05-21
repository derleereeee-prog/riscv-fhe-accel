// Measures one EvalRotate (rotation + key switch) at the configured params.
#include "common.h"
#include <iostream>
#include <cmath>

int main() {
    using namespace lbcrypto;

    std::cout << "bench_rotate: RING_DIM=" << RING_DIM
              << " BATCH_SIZE=" << BATCH_SIZE << std::endl;

    KeyPair<DCRTPoly> keys;
    auto cc = make_context(keys);

    cc->EvalMultKeyGen(keys.secretKey);
    std::cout << "EvalMultKeyGen done" << std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, {1});
    std::cout << "EvalRotateKeyGen done" << std::endl;

    std::vector<double> vec(BATCH_SIZE, 1.0);
    auto pt = cc->MakeCKKSPackedPlaintext(vec);
    auto ct = cc->Encrypt(keys.publicKey, pt);

    // Warmup
    auto r = cc->EvalRotate(ct, 1);

    // Measure
    const int N = 5;
    uint64_t start = rdcycle();
    for (int i = 0; i < N; i++)
        r = cc->EvalRotate(ct, 1);
    uint64_t end = rdcycle();

    // Use result to prevent dead-code elimination
    std::cout << "BENCH EvalRotate cycles: " << (end - start) / N << std::endl;

    try {
        Plaintext dec;
        cc->Decrypt(keys.secretKey, r, &dec);
    } catch (...) {}
    return 0;
}
