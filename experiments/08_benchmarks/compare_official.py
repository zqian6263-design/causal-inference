# -*- coding: utf-8 -*-
"""
experiments/08_benchmarks/compare_official.py — 与官方 TestPC.py 基准对照（gap 5）
==================================================================================
causal-learn 官方 tests/TestData/bnlearn_discrete_10000/benchmark_returned_results/
存有官方 PC+chisq 的输出图矩阵（`<ds>_pc_chisq_0.05_stable_0_-1.txt`，0/1/-1 编码）。
本脚本重跑 PC+chisq，把输出 `G.graph` 矩阵与官方文件**逐位比对**（MD5 级一致性），
13 个数据集全量核对——官方「完美复现」的最强实证（对应 08_benchmarks README 的
「与官方 TestPC.py 基准 MD5 对照」TODO）。

用法: python experiments/08_benchmarks/compare_official.py [bnlearn数据目录]
产出: results/metrics/compare_official.json（每集 match + 差异格数）
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

DATASETS = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley",
            "child", "insurance", "water", "hailfinder", "hepar2", "win95pts"]
OUT = os.path.join("results", "metrics", "compare_official.json")
DEFAULT_SUB = os.path.join("..", "..", "..", "causal-learn", "tests", "TestData",
                           "bnlearn_discrete_10000")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), DEFAULT_SUB))
    if not os.path.isdir(os.path.join(data_dir, "data")):
        sys.exit(f"找不到 bnlearn 数据目录: {data_dir}")
    bmr = os.path.join(data_dir, "benchmark_returned_results")

    out = {"note": "PC+chisq 输出图矩阵 vs 官方 benchmark_returned_results 逐位对照；"
                    "PC 调用与官方 TestPC.py 完全一致: pc(data, 0.05, chisq, True, 0, -1)"
                    "（stable=True, uc_rule=0, uc_priority=-1）",
           "results": {}}
    n_ok = 0
    for ds in DATASETS:
        off = os.path.join(bmr, f"{ds}_pc_chisq_0.05_stable_0_-1.txt")
        if not os.path.exists(off):
            out["results"][ds] = {"error": "官方基准文件缺失"}
            print(f"{ds:<12} 官方基准文件缺失", flush=True)
            continue
        data = np.loadtxt(os.path.join(data_dir, "data", f"{ds}.txt"), skiprows=1, dtype=int)
        t0 = time.time()
        # 与官方 TestPC.py 逐位对齐: stable=True, uc_rule=0, uc_priority=-1（默认是 2，会漂移）
        cg = pc(data, 0.05, chisq, True, 0, -1, show_progress=False)
        gm = np.asarray(cg.G.graph, dtype=int)
        official = np.loadtxt(off, dtype=int)
        dt = round(time.time() - t0, 2)
        match = bool(np.array_equal(gm, official))
        n_diff = int(np.sum(gm != official)) if gm.shape == official.shape else -1
        out["results"][ds] = {"match": match, "n_diff_cells": n_diff,
                              "nodes": gm.shape[0], "time_s": dt}
        n_ok += int(match)
        print(f"{ds:<12} {'[MATCH]' if match else '[DIFF]'} 差异格={n_diff:<4} "
              f"{data.shape[0]}×{data.shape[1]}  {dt}s", flush=True)

    out["summary"] = {"matched": n_ok, "total": len(DATASETS),
                      "all_match": n_ok == len(DATASETS)}
    print(f"\n=== 官方基准对照: {n_ok}/{len(DATASETS)} 数据集逐位一致 ===")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"指标已落盘: {OUT}")


if __name__ == "__main__":
    main()