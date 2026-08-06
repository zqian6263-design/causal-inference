# -*- coding: utf-8 -*-
"""
experiments/06_grasp_boss/run.py — GRaSP/BOSS 专属实验（多 seed 稳定性对比）
============================================================================
排列型因果发现（permutation-based）稳定性展示：
  - 数据集：线性高斯 / 线性非高斯（5 节点 7 边 TestPC 基准图, n=3000）
  - 每个方法跑 5 个 seed（42/1/7/2024/999），数据逐 seed 重生成
  - GRaSP/BOSS 内部用全局 random.shuffle（无 API seed 参数），按 (方法, seed)
    固定 random.seed(seed) 保证可复现（批次 B 确立的纪律）
  - SHD 报 mean±std；分组柱状图直观对比两方法稳定性

用法: cd causal-lab && python experiments/06_grasp_boss/run.py
产出:
  - results/metrics/06_grasp_boss.json    指标（含 per-seed SHD）
  - results/figs/06_grasp_boss.png        SHD mean±std 分组柱状图
"""
import sys, os, json, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import simulate_linear_gaussian, simulate_linear_nongaussian
from scripts.evaluate import evaluate_graph
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss

N_SAMPLES = 3000
SEEDS = [42, 1, 7, 2024, 999]
OUT = os.path.join("results", "metrics", "06_grasp_boss.json")
FIG = os.path.join("results", "figs", "06_grasp_boss.png")


def _run(method_fn, data, truth, seed):
    """单 seed：固定 random.seed(seed) 后跑排列搜索并评估。"""
    random.seed(seed)  # GRaSP/BOSS 内部随机固定，保证 (方法, seed) 可复现
    t0 = time.time()
    est = method_fn(data, score_func="local_score_BIC_from_cov", verbose=False)
    m = evaluate_graph(truth, est)
    m["time_s"] = round(time.time() - t0, 3)
    return m


def _aggregate(per_seed):
    shd_list = [m["SHD"] for m in per_seed]
    return {
        "SHD_mean": round(float(np.mean(shd_list)), 4),
        "SHD_std": round(float(np.std(shd_list)), 4),
        "SHD_per_seed": [int(s) for s in shd_list],
        "adj_precision": round(float(np.mean([m["adj_precision"] for m in per_seed])), 4),
        "adj_recall": round(float(np.mean([m["adj_recall"] for m in per_seed])), 4),
        "time_s_mean": round(float(np.mean([m["time_s"] for m in per_seed])), 3),
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    datasets = {
        "linear_gaussian": simulate_linear_gaussian,
        "linear_nongaussian": simulate_linear_nongaussian,
    }
    matrix = {}
    for dname, dgen in datasets.items():
        matrix[dname] = {}
        print(f"\n===== {dname} ({N_SAMPLES} 样本, 5 节点 7 边) =====")
        for mname, fn in [("GRaSP", grasp), ("BOSS", boss)]:
            per_seed = []
            for seed in SEEDS:
                data, truth = dgen(n=N_SAMPLES, seed=seed)
                m = _run(fn, data, truth, seed)
                per_seed.append(m)
                print(f"  {mname:<5} seed={seed:<5} SHD={m['SHD']}  {m['time_s']}s")
            matrix[dname][mname] = _aggregate(per_seed)
            a = matrix[dname][mname]
            print(f"  -> {mname}: SHD={a['SHD_mean']}±{a['SHD_std']}  "
                  f"adjP/R={a['adj_precision']}/{a['adj_recall']}")

    # 分组柱状图: GRaSP vs BOSS × 两数据集, 误差棒 = std（稳定性对比）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dnames = list(matrix.keys())
    methods = ["GRaSP", "BOSS"]
    colors = ["#2b6cb0", "#dd6b20"]
    x = np.arange(len(dnames))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, mname in enumerate(methods):
        means = [matrix[d][mname]["SHD_mean"] for d in dnames]
        stds = [matrix[d][mname]["SHD_std"] for d in dnames]
        ax.bar(x + (k - 0.5) * width, means, width, yerr=stds, capsize=4,
               label=mname, color=colors[k])
        for xi, mn, sd in zip(x + (k - 0.5) * width, means, stds):
            ax.text(xi, mn + sd + 0.15, f"{mn:.1f}±{sd:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(dnames)
    ax.set_ylabel("SHD (5-seed mean +/- std)")
    ax.set_title("GRaSP vs BOSS stability over 5 seeds")
    ax.set_ylim(0, max([matrix[d][m]["SHD_mean"] + matrix[d][m]["SHD_std"]
                        for d in dnames for m in methods]) * 1.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    plt.close(fig)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_samples": N_SAMPLES, "seeds": SEEDS, "matrix": matrix}, f,
                  ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标落盘: {OUT}\n图落盘: {FIG}")


if __name__ == "__main__":
    main()
