// Measures one EvalAdd(ciphertext, ciphertext) at the configured params.
#include "common.h"
#include <iostream>

int main() {
    using namespace lbcrypto;

    std::cout << "bench_add: RING_DIM=" << RING_DIM
              << " BATCH_SIZE=" << BATCH_SIZE << std::endl;

    KeyPair<DCRTPoly> keys;
    auto cc = make_context(keys);

    std::vector<double> v1(BATCH_SIZE, 1.0), v2(BATCH_SIZE, 2.0);
    auto ct1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v1));
    auto ct2 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v2));

    // Warmup
    auto r = cc->EvalAdd(ct1, ct2);

    // Measure
    const int N = 10;
    uint64_t start = rdcycle();
    for (int i = 0; i < N; i++)
        r = cc->EvalAdd(ct1, ct2);
    uint64_t end = rdcycle();

    std::cout << "BENCH EvalAdd cycles: " << (end - start) / N << std::endl;

    // Prevent dead-code elimination; ignore precision errors at small params
    try {
        Plaintext dec;
        cc->Decrypt(keys.secretKey, r, &dec);
    } catch (...) {}
    return 0;
}
