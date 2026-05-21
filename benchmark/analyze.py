#!/usr/bin/env python3
"""
方案 A breakdown 计算。
从各 benchmark 的输出中读取 cycle 数，按 BSGS 调用次数算占比。

用法：
  python analyze.py bench_rotate.log bench_pmul.log bench_add.log

每个 log 文件是对应 benchmark 在 gem5 上的 stdout，
包含形如 "BENCH EvalRotate cycles: 123456" 的行。

也可以直接在命令行输入数字：
  python analyze.py --rotate 123456 --pmul 78900 --add 12300
"""

import sys, re, argparse, math

BATCH_SIZE = 1024  # batchSize for the target BSGS (not the benchmark params)
sqrt_n = int(math.isqrt(BATCH_SIZE))

# BSGS operation counts (from benchmark_matvec.cpp skeleton analysis)
# Baby-step rotations:  31 per giant step × 32 giant steps = 992
# Giant-step rotations: 31
N_ROTATE = (sqrt_n - 1) * sqrt_n + (sqrt_n - 1)   # 1023
N_PMUL   = sqrt_n * sqrt_n                          # 1024
N_ADD    = (sqrt_n - 1) * sqrt_n + (sqrt_n - 1)    # 1023

def extract(path, pattern):
    text = open(path).read()
    m = re.search(pattern + r':\s*(\d+)', text)
    if not m:
        raise ValueError(f"Pattern '{pattern}' not found in {path}")
    return int(m.group(1))

def report(rotate_c, pmul_c, add_c):
    t_rotate = N_ROTATE * rotate_c
    t_pmul   = N_PMUL   * pmul_c
    t_add    = N_ADD    * add_c
    total    = t_rotate + t_pmul + t_add

    print(f"\n=== BSGS Breakdown  (batchSize={BATCH_SIZE}, sqrtN={sqrt_n}) ===")
    print(f"{'Operation':<20} {'count':>6} {'cycles/op':>12} {'total cycles':>15} {'%':>6}")
    print("-" * 65)
    for name, n, c, t in [
        ("EvalRotate",  N_ROTATE, rotate_c, t_rotate),
        ("EvalMult(ct,pt)", N_PMUL, pmul_c, t_pmul),
        ("EvalAdd",     N_ADD,   add_c,   t_add),
    ]:
        print(f"{name:<20} {n:>6} {c:>12,} {t:>15,} {100*t/total:>5.1f}%")
    print("-" * 65)
    print(f"{'Total':<20} {'':>6} {'':>12} {total:>15,} {'100.0':>5}%")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rotate', type=int)
    p.add_argument('--pmul',   type=int)
    p.add_argument('--add',    type=int)
    p.add_argument('logs', nargs='*')
    args = p.parse_args()

    if args.rotate and args.pmul and args.add:
        report(args.rotate, args.pmul, args.add)
        return

    if len(args.logs) == 3:
        rotate_c = extract(args.logs[0], 'BENCH EvalRotate cycles')
        pmul_c   = extract(args.logs[1], r'BENCH EvalMult\(ct,pt\) cycles')
        add_c    = extract(args.logs[2], 'BENCH EvalAdd cycles')
        report(rotate_c, pmul_c, add_c)
        return

    # Interactive
    rotate_c = int(input("EvalRotate cycles/op: "))
    pmul_c   = int(input("EvalMult(ct,pt) cycles/op: "))
    add_c    = int(input("EvalAdd cycles/op: "))
    report(rotate_c, pmul_c, add_c)

if __name__ == '__main__':
    main()
