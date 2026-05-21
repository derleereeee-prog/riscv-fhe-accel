#ifndef NTT_RVV
#define NTT_RVV

#include <stdint.h>

void ntt_korn_lambiote_vector(
    const uint32_t p, uint32_t t, uint32_t logt, const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityTable, const uint64_t *preconRootOfUnityTable);

void intt_pease_vector_mulh(
    const uint32_t p, const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityInverseTable,
    const uint64_t *preconRootOfUnityInverseTable,
    const uint64_t cycloOrderInv, const uint64_t preconCycloOrderInv);

void free_ntts_mem();

#endif
