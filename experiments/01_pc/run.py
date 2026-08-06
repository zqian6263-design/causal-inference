# -*- coding: utf-8 -*-
"""
experiments/01_pc/run.py — PC 专属实验
=======================================
PC（Peter-Clark）约束型因果发现，两组对照：
  A. PC + fisherz   5 节点线性高斯（TestPC 基准图，n=3000, seed=42）
     —— 线性高斯 + 参数检验的黄金组合，预期接近完美（SHD=0）
  B. PC + kci       同真值图、小样本（n=600）
     —— KCI 非参数但 O(n³) 极慢、样本需求大（小样本检验力不足，实测 SHD 高）

真值 DAG -> CPDAG 对齐后算 SHD/P/R（scripts.evaluate）；CPDAG 图落 results/figs。
单 seed（PC 无内部随机，数据 seed=42 固定）。

用法: cd causal-lab && python experiments/01_pc/run.py
产出:
  - results/metrics/01_pc.json               指标
  - results/figs/01_pc_fisherz.png           估计 CPDAG（A）
  - results/figs/01_pc_kci.png               估计 CPDAG（B）
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import simulate_linear_gaussian
from scripts.evaluate import evaluate_graph
from scripts.plotting import plot_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, kci

N = 3000
SEED = 42
N_KCI = 600
OUT = os.path.join("results", "metrics", "01_pc.json")
FIG_DIR = os.path.join("results", "figs")


def run_pc(name, data, truth, indep_test, fig_path, **kw):
    t0 = time.time()
    cg = pc(data, 0.05, indep_test, show_progress=False, **kw)
    m = evaluate_graph(truth, cg)
    m["time_s"] = round(time.time() - t0, 3)
    plot_graph(cg.G, fig_path, title=f"{name} (CPDAG)")
    print(f"  {name:<14} SHD={m['SHD']:<3} adjP/R={m['adj_precision']}/{m['adj_recall']}  "
          f"arrP/R={m['arrow_precision']}/{m['arrow_recall']}  {m['time_s']}s")
    return m


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    data, truth = simulate_linear_gaussian(n=N, seed=SEED)
    print(f"基准: {N} 样本, 5 节点 7 边（TestPC 基准图）, seed={SEED}\n")

    results = {"data": "linear_gaussian (5 nodes, 7 edges)", "n_samples": N, "seed": SEED}
    print("A. PC + fisherz（线性高斯, n=3000）")
    results["PC+fisherz"] = run_pc(
        "PC+fisherz", data, truth, fisherz,
        os.path.join(FIG_DIR, "01_pc_fisherz.png"))

    print(f"\nB. PC + kci（同真值图, 小样本 n={N_KCI}, O(n^3) 慢）")
    # cache 文件名内嵌样本量：KCI 缓存按数据 hash 校验，换 n 会 "Data hash mismatch"
    cache_path = os.path.join("results", "metrics", f"cache_pc_kci_n{N_KCI}.json")
    results["PC+kci_n600"] = run_pc(
        f"PC+kci(n{N_KCI})", data[:N_KCI], truth, kci,
        os.path.join(FIG_DIR, "01_pc_kci.png"), cache_path=cache_path)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标落盘: {OUT}")


if __name__ == "__main__":
    main()
