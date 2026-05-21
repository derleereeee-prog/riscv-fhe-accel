#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "ntt-rvv.h"

// ============================================================
// 类型约定（标准 RVV 1.0）：
//   vint64m1_t  ← 所有数据向量（vmulh 要求有符号）
//   vuint64m1_t ← 所有索引向量（vrgather/vsll/vsrl/vand/vor）
//   vbool64_t   ← mask（LMUL=1，64位元素，每元素1个mask bit）
//   size_t      ← vl（替代原来的 uint32_t gvl/max_gvl）
//
// 有符号/无符号转换用 vreinterpret，不改变 bit 表示：
//   __riscv_vreinterpret_v_i64m1_u64m1(v)  ← signed→unsigned
//   __riscv_vreinterpret_v_u64m1_i64m1(v)  ← unsigned→signed
// ============================================================

uint8_t initialized_mem = 1;
uint64_t *aux_increment = NULL;
uint64_t *auxiliary_array = NULL;

void free_ntts_mem()
{
    if (!initialized_mem)
    {
        free(aux_increment);
        free(auxiliary_array);
    }
    initialized_mem = 1;
}

void ntt_korn_lambiote_vector(
    const uint32_t p, uint32_t t, uint32_t logt, const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityTable, const uint64_t *preconRootOfUnityTable)
{
    uint8_t element_or_aux = 0;
    uint32_t stopvalue_sft1 = p >> 1;

    // 标准 RVV：vsetvl 返回 size_t，原来的 gvl 和 max_gvl 合并为 vl
    size_t vl = __riscv_vsetvl_e64m1(p);

    uint64_t *input_stage_array  = element;
    uint64_t *output_stage_array = NULL;
    uint64_t *swap_pointer_aux   = NULL;

    uint32_t log_of_vl = (uint32_t)log2((double)vl);

    // 广播常数：所有数据向量用 vint64m1_t
    vint64m1_t v_coef_mod = __riscv_vmv_v_x_i64m1((int64_t)modulus, vl);
    vint64m1_t v_U, v_V;

    if (initialized_mem)
    {
        initialized_mem = 0;
        aux_increment   = (uint64_t *)malloc(vl * sizeof(uint64_t));
        auxiliary_array = (uint64_t *)malloc(p * sizeof(uint64_t));
        for (size_t i = 0; i < vl; i++)
            aux_increment[i] = i;
    }

    output_stage_array = auxiliary_array;

    // 索引向量用 vuint64m1_t（供 Stage B 的 vrgather/vsll/vsrl/vand/vor 使用）
    vuint64m1_t v_index_1         = __riscv_vmv_v_x_u64m1(1, vl);
    vuint64m1_t v_index_original  = __riscv_vle64_v_u64m1(&aux_increment[0], vl);
    vuint64m1_t v_index_stage_aux = __riscv_vmv_v_x_u64m1(1, vl);
    vuint64m1_t v_m               = __riscv_vmv_v_x_u64m1(2, vl);

    uint32_t m = 1;

    // Stage A 广播单个旋转因子
    vint64m1_t v_S           = __riscv_vmv_v_x_i64m1((int64_t)rootOfUnityTable[1], vl);
    vint64m1_t v_precomp_aux = __riscv_vmv_v_x_i64m1((int64_t)preconRootOfUnityTable[1], vl);

    // --------------------------------------------------------
    // Stage A：First Loop（m=1，Broadcast）
    // U 来自前半段连续内存，V 来自后半段连续内存
    // 输出以交错格式写入 auxiliary_array（stride=2元素=16字节）
    // --------------------------------------------------------
    for (uint32_t i = 0; i < p; i += (uint32_t)(vl << 1))
    {
        // 连续 vload（vle64）
        v_U = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[(i >> 1)]), vl);
        v_V = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[(i >> 1) + stopvalue_sft1]), vl);

        // Shoup 模乘：v_V = [W * v_V] mod Q（6条指令）
        // Step1: vmulh 取高64位估商
        vint64m1_t v_q    = __riscv_vmulh_vv_i64m1(v_V, v_precomp_aux, vl);
        // Step2a: vmul 算 W*V 低64位
        vint64m1_t v_mult = __riscv_vmul_vv_i64m1(v_V, v_S, vl);
        // Step2b: vmul 算 q*Q 低64位
        vint64m1_t v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
        // Step2c: vsub 相减得近似余数
        v_V = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
        // Step3: 条件修正（vmsleu 需要无符号，reinterpret 后比较）
        vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_V), vl);
        v_V = __riscv_vsub_vv_i64m1_mu(mask, v_V, v_V, v_coef_mod, vl);

        // 蝴蝶加法：R0 = U + [W*V] mod Q，结果在 [0,2Q)，条件减法修正
        vint64m1_t v_result_0 = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
        v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);
        // strided store（vsse64）：stride=16字节=2个uint64_t，R0写偶数位
        __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[i]), 16, v_result_0, vl);

        // 蝴蝶减法：R1 = U - [W*V] mod Q，下溢时条件加法修正
        vint64m1_t v_result_1 = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
        v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);
        // R1 写奇数位，和 R0 交错
        __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[i + 1]), 16, v_result_1, vl);
    }

    m <<= 1;
    element_or_aux ^= 1;

    // 加载 omegaTable 前 vl 个元素到向量寄存器，供 Stage B 的 vrgather 使用
    vint64m1_t v_omega_og   = __riscv_vle64_v_i64m1((const int64_t*)&rootOfUnityTable[0], vl);
    vint64m1_t v_precomp_og = __riscv_vle64_v_i64m1((const int64_t*)&preconRootOfUnityTable[0], vl);

    swap_pointer_aux   = output_stage_array;
    output_stage_array = input_stage_array;
    input_stage_array  = swap_pointer_aux;

    // --------------------------------------------------------
    // Stage B：Intermediate Loops（1 < m < vl，vrgather）
    // 用 Algorithm 4 构造索引，vrgather 取本 stage 每个槽所需的 W
    // --------------------------------------------------------
    for (; m < (uint32_t)vl; m <<= 1, element_or_aux ^= 1)
    {
        // Algorithm 4：构造 vrgather 索引
        // vand 取 index_og 的低 log(m) 位，vadd 加偏移量 m
        vuint64m1_t v_index = __riscv_vand_vv_u64m1(v_index_original, v_index_stage_aux, vl);
        v_index = __riscv_vadd_vv_u64m1(v_index, v_m, vl);

        // vrgather：source 为 vint64m1_t，需 reinterpret 为 vuint64m1_t 后再 gather，结果再转回
        v_S = __riscv_vreinterpret_v_u64m1_i64m1(
            __riscv_vrgather_vv_u64m1(
                __riscv_vreinterpret_v_i64m1_u64m1(v_omega_og), v_index, vl));
        v_precomp_aux = __riscv_vreinterpret_v_u64m1_i64m1(
            __riscv_vrgather_vv_u64m1(
                __riscv_vreinterpret_v_i64m1_u64m1(v_precomp_og), v_index, vl));

        // 更新索引向量为下一个 stage 做准备：(x << 1) | 1
        v_index_stage_aux = __riscv_vsll_vv_u64m1(v_index_stage_aux, v_index_1, vl);
        v_index_stage_aux = __riscv_vor_vv_u64m1(v_index_stage_aux, v_index_1, vl);
        v_m = __riscv_vsll_vv_u64m1(v_m, v_index_1, vl);

        // 内层蝴蝶计算，与 Stage A 完全相同，W 已由 vrgather 准备好
        for (uint32_t i = 0; i < p; i += (uint32_t)(vl << 1))
        {
            v_U = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[(i >> 1)]), vl);
            v_V = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[(i >> 1) + stopvalue_sft1]), vl);

            // Shoup 模乘
            vint64m1_t v_q    = __riscv_vmulh_vv_i64m1(v_V, v_precomp_aux, vl);
            vint64m1_t v_mult = __riscv_vmul_vv_i64m1(v_V, v_S, vl);
            vint64m1_t v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
            v_V = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_V), vl);
            v_V = __riscv_vsub_vv_i64m1_mu(mask, v_V, v_V, v_coef_mod, vl);

            vint64m1_t v_result_0 = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
            v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);

            vint64m1_t v_result_1 = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
            v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

            __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[i]), 16, v_result_0, vl);
            __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[i + 1]), 16, v_result_1, vl);
        }

        swap_pointer_aux   = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array  = swap_pointer_aux;
    }

    // --------------------------------------------------------
    // Stage C：Final Loops（m >= vl，顺序 vload）
    // 中层循环每段加载一次 W，内层循环复用
    // --------------------------------------------------------
    for (; m < p; m <<= 1, element_or_aux ^= 1)
    {
        for (uint32_t i = 0; i < (m >> log_of_vl); i++)
        {
            uint64_t index = (uint64_t)i << log_of_vl;

            // 顺序 vload 加载本段旋转因子，内层循环全程复用
            v_S          = __riscv_vle64_v_i64m1((const int64_t*)&rootOfUnityTable[index + m], vl);
            v_precomp_aux = __riscv_vle64_v_i64m1((const int64_t*)&preconRootOfUnityTable[index + m], vl);

            for (uint32_t j = 0; j < (stopvalue_sft1 / m); j++)
            {
                v_U = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[index + j * m]), vl);
                v_V = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[index + j * m + stopvalue_sft1]), vl);

                // Shoup 模乘
                vint64m1_t v_q    = __riscv_vmulh_vv_i64m1(v_V, v_precomp_aux, vl);
                vint64m1_t v_mult = __riscv_vmul_vv_i64m1(v_V, v_S, vl);
                vint64m1_t v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
                v_V = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
                vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_V), vl);
                v_V = __riscv_vsub_vv_i64m1_mu(mask, v_V, v_V, v_coef_mod, vl);

                vint64m1_t v_result_0 = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
                mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
                v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);

                vint64m1_t v_result_1 = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
                mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
                v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

                // 输出地址 << 1：交错格式每对蝴蝶占两个槽
                __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[((index + j * m) << 1)]), 16, v_result_0, vl);
                __riscv_vsse64_v_i64m1((int64_t*)&(output_stage_array[((index + j * m) << 1) + 1]), 16, v_result_1, vl);
            }
        }

        swap_pointer_aux   = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array  = swap_pointer_aux;
    }

    // 收尾：奇数 stage 数时结果在辅助数组，拷贝回 element
    if (element_or_aux)
    {
        for (uint32_t i = 0; i < p; i += (uint32_t)vl)
        {
            vint64m1_t v_auxarr = __riscv_vle64_v_i64m1((const int64_t*)&(input_stage_array[i]), vl);
            __riscv_vse64_v_i64m1((int64_t*)&element[i], v_auxarr, vl);
        }
    }
}

void intt_pease_vector_mulh(
    const uint32_t p, const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityInverseTable,
    const uint64_t *preconRootOfUnityInverseTable,
    const uint64_t cycloOrderInv, const uint64_t preconCycloOrderInv)
{
    uint8_t element_or_aux = 0;
    uint32_t stopvalue_sft1 = p >> 1;
    size_t vl = __riscv_vsetvl_e64m1(p);

    vint64m1_t v_coef_mod            = __riscv_vmv_v_x_i64m1((int64_t)modulus, vl);
    // INTT 特有：p^{-1} 和其 Shoup 预计算值
    vint64m1_t v_preconCycloOrderInv = __riscv_vmv_v_x_i64m1((int64_t)preconCycloOrderInv, vl);
    vint64m1_t v_cycloOrderInv       = __riscv_vmv_v_x_i64m1((int64_t)cycloOrderInv, vl);

    uint64_t *input_stage_array  = element;
    uint64_t *output_stage_array = NULL;
    uint64_t *swap_pointer_aux   = NULL;
    uint32_t log_of_vl = (uint32_t)log2((double)vl);

    if (initialized_mem)
    {
        initialized_mem = 0;
        aux_increment   = (uint64_t *)malloc(vl * sizeof(uint64_t));
        auxiliary_array = (uint64_t *)malloc(p * sizeof(uint64_t));
        for (size_t i = 0; i < vl; i++)
            aux_increment[i] = i;
    }

    output_stage_array = auxiliary_array;

    // INTT 从大 m 到小 m，索引初始值和 NTT 相反
    vuint64m1_t v_index_1         = __riscv_vmv_v_x_u64m1(1, vl);
    vuint64m1_t v_index_original  = __riscv_vle64_v_u64m1(&aux_increment[0], vl);
    vuint64m1_t v_index_stage_aux = __riscv_vmv_v_x_u64m1((vl >> 1) - 1, vl);
    vuint64m1_t v_m               = __riscv_vmv_v_x_u64m1(vl >> 1, vl);

    uint32_t m = p >> 1;

    // --------------------------------------------------------
    // Stage A（INTT）：第一个 stage，m=p/2
    // 输入是 NTT 输出的交错格式，用 strided load（vlse64）
    // 每个元素额外乘 p^{-1}（cycloOrderInv），两次 Shoup
    // 输出连续 store：R0 前半段，R1 后半段
    // --------------------------------------------------------
    for (uint32_t i = 0; i < (m >> log_of_vl); i++)
    {
        uint64_t index = (uint64_t)i << log_of_vl;

        vint64m1_t v_S           = __riscv_vle64_v_i64m1((const int64_t*)&rootOfUnityInverseTable[index + m], vl);
        vint64m1_t v_precomp_aux = __riscv_vle64_v_i64m1((const int64_t*)&preconRootOfUnityInverseTable[index + m], vl);

        // strided load（vlse64）：偶数位→U，奇数位→V，stride=16字节
        vint64m1_t v_U = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index) << 1)]), 16, vl);
        vint64m1_t v_V = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index) << 1) + 1]), 16, vl);

        // R0 = (U + V) mod Q
        vint64m1_t v_result_0 = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
        vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
        v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);

        // R0 *= p^{-1}（Shoup，用 v_V 估商）
        vint64m1_t v_q    = __riscv_vmulh_vv_i64m1(v_V, v_preconCycloOrderInv, vl);
        vint64m1_t v_mult = __riscv_vmul_vv_i64m1(v_result_0, v_cycloOrderInv, vl);
        vint64m1_t v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
        v_result_0 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
        v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);

        // R1 = (U - V) mod Q
        vint64m1_t v_result_1 = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
        v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

        // R1 *= W^{-1}（第一次 Shoup）
        v_q    = __riscv_vmulh_vv_i64m1(v_result_1, v_precomp_aux, vl);
        v_mult = __riscv_vmul_vv_i64m1(v_result_1, v_S, vl);
        v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
        v_result_1 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
        v_result_1 = __riscv_vsub_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

        // R1 *= p^{-1}（第二次 Shoup）
        v_q    = __riscv_vmulh_vv_i64m1(v_result_1, v_preconCycloOrderInv, vl);
        v_mult = __riscv_vmul_vv_i64m1(v_result_1, v_cycloOrderInv, vl);
        v_aux  = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
        v_result_1 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
        mask = __riscv_vmsleu_vv_u64m1_b64(
            __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
            __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
        v_result_1 = __riscv_vsub_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

        // 连续 store：R0 前半段，R1 后半段
        __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index]), v_result_0, vl);
        __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index + stopvalue_sft1]), v_result_1, vl);
    }

    swap_pointer_aux   = output_stage_array;
    output_stage_array = input_stage_array;
    input_stage_array  = swap_pointer_aux;
    m >>= 1;
    element_or_aux ^= 1;

    // --------------------------------------------------------
    // Stage B（INTT）：m >= vl，无 p^{-1}，j 步长为 2（循环展开）
    // --------------------------------------------------------
    for (; m >= (uint32_t)vl; m >>= 1, element_or_aux ^= 1)
    {
        for (uint32_t i = 0; i < (m >> log_of_vl); i++)
        {
            uint64_t index = (uint64_t)i << log_of_vl;

            vint64m1_t v_S           = __riscv_vle64_v_i64m1((const int64_t*)&rootOfUnityInverseTable[index + m], vl);
            vint64m1_t v_precomp_aux = __riscv_vle64_v_i64m1((const int64_t*)&preconRootOfUnityInverseTable[index + m], vl);

            // j += 2：同时处理两对蝴蝶，减少循环控制开销
            for (uint32_t j = 0; j < (stopvalue_sft1 / m); j += 2)
            {
                // 两对蝴蝶的 strided load
                vint64m1_t v_U  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index + j * m) << 1)]), 16, vl);
                vint64m1_t v_V  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index + j * m) << 1) + 1]), 16, vl);
                vint64m1_t _v_U = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index + (j + 1) * m) << 1)]), 16, vl);
                vint64m1_t _v_V = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[((index + (j + 1) * m) << 1) + 1]), 16, vl);

                vint64m1_t v_result_0  = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
                vint64m1_t _v_result_0 = __riscv_vadd_vv_i64m1(_v_U, _v_V, vl);

                vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
                v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);
                vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(_v_result_0), vl);
                // 注：原始代码此处用 mask 而非 _mask，保持原有行为
                _v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, _v_result_0, _v_result_0, v_coef_mod, vl);

                vint64m1_t v_result_1  = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
                mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
                v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);
                vint64m1_t _v_result_1 = __riscv_vsub_vv_i64m1(_v_U, _v_V, vl);
                _mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
                _v_result_1 = __riscv_vadd_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

                vint64m1_t v_q  = __riscv_vmulh_vv_i64m1(v_result_1, v_precomp_aux, vl);
                vint64m1_t _v_q = __riscv_vmulh_vv_i64m1(_v_result_1, v_precomp_aux, vl);

                vint64m1_t v_mult  = __riscv_vmul_vv_i64m1(v_result_1, v_S, vl);
                vint64m1_t v_aux   = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
                v_result_1 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
                mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
                v_result_1 = __riscv_vsub_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

                vint64m1_t _v_mult = __riscv_vmul_vv_i64m1(_v_result_1, v_S, vl);
                vint64m1_t _v_aux  = __riscv_vmul_vv_i64m1(_v_q, v_coef_mod, vl);
                _v_result_1 = __riscv_vsub_vv_i64m1(_v_mult, _v_aux, vl);
                _mask = __riscv_vmsleu_vv_u64m1_b64(
                    __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                    __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
                _v_result_1 = __riscv_vsub_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

                __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index + j * m]), v_result_0, vl);
                __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index + j * m + stopvalue_sft1]), v_result_1, vl);
                __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index + (j + 1) * m]), _v_result_0, vl);
                __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[index + (j + 1) * m + stopvalue_sft1]), _v_result_1, vl);
            }
        }

        swap_pointer_aux   = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array  = swap_pointer_aux;
    }

    // 加载逆旋转因子前 vl 个，供 Stage C 的 vrgather 使用
    vint64m1_t v_omega_og   = __riscv_vle64_v_i64m1((const int64_t*)&rootOfUnityInverseTable[0], vl);
    vint64m1_t v_precomp_og = __riscv_vle64_v_i64m1((const int64_t*)&preconRootOfUnityInverseTable[0], vl);

    // --------------------------------------------------------
    // Stage C（INTT）：vrgather 阶段，m < vl
    // 与 NTT Stage B 镜像，索引更新用 vsrl（右移，m 从大到小）
    // --------------------------------------------------------
    for (; m >= 2; m >>= 1, element_or_aux ^= 1)
    {
        // vrgather 索引构造（方向与 NTT 相反）
        vuint64m1_t v_index = __riscv_vand_vv_u64m1(v_index_original, v_index_stage_aux, vl);
        v_index = __riscv_vadd_vv_u64m1(v_index, v_m, vl);

        vint64m1_t v_S = __riscv_vreinterpret_v_u64m1_i64m1(
            __riscv_vrgather_vv_u64m1(
                __riscv_vreinterpret_v_i64m1_u64m1(v_omega_og), v_index, vl));
        vint64m1_t v_precomp_aux = __riscv_vreinterpret_v_u64m1_i64m1(
            __riscv_vrgather_vv_u64m1(
                __riscv_vreinterpret_v_i64m1_u64m1(v_precomp_og), v_index, vl));

        // 右移更新（NTT 是左移）
        v_index_stage_aux = __riscv_vsrl_vv_u64m1(v_index_stage_aux, v_index_1, vl);
        v_m               = __riscv_vsrl_vv_u64m1(v_m, v_index_1, vl);

        // 内层：同时处理两组（i 和 ii = i + 2*vl）
        for (uint32_t i = 0; i < p; i += (uint32_t)(vl << 2))
        {
            uint32_t ii = i + (uint32_t)(vl << 1);

            vint64m1_t v_U  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[i]), 16, vl);
            vint64m1_t v_V  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[i + 1]), 16, vl);
            vint64m1_t _v_U = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[ii]), 16, vl);
            vint64m1_t _v_V = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[ii + 1]), 16, vl);

            vint64m1_t v_result_0  = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
            vint64m1_t _v_result_0 = __riscv_vadd_vv_i64m1(_v_U, _v_V, vl);

            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
            v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);
            vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_0), vl);
            _v_result_0 = __riscv_vsub_vv_i64m1_mu(_mask, _v_result_0, _v_result_0, v_coef_mod, vl);

            vint64m1_t v_result_1  = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
            v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);
            vint64m1_t _v_result_1 = __riscv_vsub_vv_i64m1(_v_U, _v_V, vl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
            _v_result_1 = __riscv_vadd_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

            vint64m1_t v_q  = __riscv_vmulh_vv_i64m1(v_result_1, v_precomp_aux, vl);
            vint64m1_t _v_q = __riscv_vmulh_vv_i64m1(_v_result_1, v_precomp_aux, vl);

            vint64m1_t v_mult  = __riscv_vmul_vv_i64m1(v_result_1, v_S, vl);
            vint64m1_t v_aux   = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
            v_result_1 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
            v_result_1 = __riscv_vsub_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

            vint64m1_t _v_mult = __riscv_vmul_vv_i64m1(_v_result_1, v_S, vl);
            vint64m1_t _v_aux  = __riscv_vmul_vv_i64m1(_v_q, v_coef_mod, vl);
            _v_result_1 = __riscv_vsub_vv_i64m1(_v_mult, _v_aux, vl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
            _v_result_1 = __riscv_vsub_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

            // 输出地址 >> 1（对应 NTT Stage C 的 << 1）
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[i >> 1]), v_result_0, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[(i >> 1) + stopvalue_sft1]), v_result_1, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[ii >> 1]), _v_result_0, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[(ii >> 1) + stopvalue_sft1]), _v_result_1, vl);
        }

        swap_pointer_aux   = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array  = swap_pointer_aux;
    }

    // --------------------------------------------------------
    // Stage D（INTT）：最后一个 stage，m=1，broadcast
    // --------------------------------------------------------
    if (m != 0)
    {
        v_omega_og   = __riscv_vmv_v_x_i64m1((int64_t)rootOfUnityInverseTable[1], vl);
        v_precomp_og = __riscv_vmv_v_x_i64m1((int64_t)preconRootOfUnityInverseTable[1], vl);

        for (uint32_t i = 0; i < p; i += (uint32_t)(vl << 2))
        {
            uint32_t ii = i + (uint32_t)(vl << 1);

            vint64m1_t v_U  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[i]), 16, vl);
            vint64m1_t v_V  = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[i + 1]), 16, vl);
            vint64m1_t _v_U = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[ii]), 16, vl);
            vint64m1_t _v_V = __riscv_vlse64_v_i64m1((const int64_t*)&(input_stage_array[ii + 1]), 16, vl);

            vint64m1_t v_result_0  = __riscv_vadd_vv_i64m1(v_U, v_V, vl);
            vint64m1_t _v_result_0 = __riscv_vadd_vv_i64m1(_v_U, _v_V, vl);

            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_0), vl);
            v_result_0 = __riscv_vsub_vv_i64m1_mu(mask, v_result_0, v_result_0, v_coef_mod, vl);
            vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_0), vl);
            _v_result_0 = __riscv_vsub_vv_i64m1_mu(_mask, _v_result_0, _v_result_0, v_coef_mod, vl);

            vint64m1_t v_result_1  = __riscv_vsub_vv_i64m1(v_U, v_V, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
            v_result_1 = __riscv_vadd_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);
            vint64m1_t _v_result_1 = __riscv_vsub_vv_i64m1(_v_U, _v_V, vl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
            _v_result_1 = __riscv_vadd_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

            // 用广播的逆旋转因子（v_omega_og/v_precomp_og）
            vint64m1_t v_q  = __riscv_vmulh_vv_i64m1(v_result_1, v_precomp_og, vl);
            vint64m1_t _v_q = __riscv_vmulh_vv_i64m1(_v_result_1, v_precomp_og, vl);

            vint64m1_t v_mult  = __riscv_vmul_vv_i64m1(v_result_1, v_omega_og, vl);
            vint64m1_t v_aux   = __riscv_vmul_vv_i64m1(v_q, v_coef_mod, vl);
            v_result_1 = __riscv_vsub_vv_i64m1(v_mult, v_aux, vl);
            mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(v_result_1), vl);
            v_result_1 = __riscv_vsub_vv_i64m1_mu(mask, v_result_1, v_result_1, v_coef_mod, vl);

            vint64m1_t _v_mult = __riscv_vmul_vv_i64m1(_v_result_1, v_omega_og, vl);
            vint64m1_t _v_aux  = __riscv_vmul_vv_i64m1(_v_q, v_coef_mod, vl);
            _v_result_1 = __riscv_vsub_vv_i64m1(_v_mult, _v_aux, vl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(
                __riscv_vreinterpret_v_i64m1_u64m1(v_coef_mod),
                __riscv_vreinterpret_v_i64m1_u64m1(_v_result_1), vl);
            _v_result_1 = __riscv_vsub_vv_i64m1_mu(_mask, _v_result_1, _v_result_1, v_coef_mod, vl);

            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[i >> 1]), v_result_0, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[(i >> 1) + stopvalue_sft1]), v_result_1, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[ii >> 1]), _v_result_0, vl);
            __riscv_vse64_v_i64m1((int64_t*)&(output_stage_array[(ii >> 1) + stopvalue_sft1]), _v_result_1, vl);
        }

        element_or_aux ^= 1;
    }

    if (element_or_aux)
    {
        for (uint32_t i = 0; i < p; i += (uint32_t)vl)
        {
            vint64m1_t v_auxarr = __riscv_vle64_v_i64m1((const int64_t*)&(output_stage_array[i]), vl);
            __riscv_vse64_v_i64m1((int64_t*)&element[i], v_auxarr, vl);
        }
    }
}
