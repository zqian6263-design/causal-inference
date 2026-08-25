# -*- coding: utf-8 -*-
"""
experiments/08_benchmarks/alpha_adaptive.py — 大图 PC alpha 自适应（F2 轮，E 轮遗留治理）
============================================================================================
问题：固定 alpha=0.05 + 10000 样本在 50+ 变量上做 chisq 多次检验，累积误报。
      实测（bnlearn_all.json）：hailfinder SHD=96、adjP/R≈0.18/0.11；hepar2 SHD=92；
      win95pts SHD=57。本脚本在三个大图上对比固定 alpha vs 自适应校正。

校正选型（Bonferroni，非 BH-FDR）：
  causal-learn 的 pc() 内部按「骨架阶段逐条件集」顺序消费 p 值，**不暴露全部检验
  p 值**，BH-FDR 需要完整 p 值集合排序，无法在不重写 learn_skeleton 的前提下直接实现。
  Bonferroni 只需修正显著性阈值（FWER 控制）：
      alpha_bonf = alpha / C(n,2) = alpha / (n*(n-1)/2)
  以「任意两变量之间的独立性检验」总数上界作多重性修正——这是高维 PC 的常用保守做法。
  另加 alpha=0.01 作解释性探针（观察效果是否随 alpha 单调，判断 Bonferroni 是否过度保守）。

评估：真值 DAG → CPDAG 对齐 → evaluate.py（SHD / adjP-R / arrP-R / 时间）。
数据：causal-learn TestData bnlearn_discrete_10000（同 run_all_bnlearn.py，可 <数据目录> 覆盖）。

用法:
  python experiments/08_benchmarks/alpha_adaptive.py [bnlearn数据目录]
产出:
  results/metrics/alpha_adaptive.json   三图 × [0.05, 0.01, alpha_bonf] 全指标
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 导入同目录 run_all_bnlearn
import numpy as np
from run_all_bnlearn import parse_truth_graph

from scripts.evaluate import evaluate_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

DATASETS = [("hailfinder", 56), ("hepar2", 70), ("win95pts", 76)]
ALPHA_FIXED = 0.05
ALPHA_MID = 0.01                    # 解释性探针：观察效果是否单调
ALPHA_BASE = 0.05                   # Bonferroni 的基准（与 ALPHA_FIXED 一致）
OUT = os.path.join("results", "metrics", "alpha_adaptive.json")
DEFAULT_SUB = os.path.join("..", "..", "..", "causal-learn", "tests", "TestData",
                           "bnlearn_discrete_10000")


def bonferroni_alpha(n):
    """Bonferroni 修正：alpha / C(n,2)。"""
    n_pairs = n * (n - 1) // 2
    return round(ALPHA_BASE / n_pairs, 10)


def run_pc(data, truth, alpha, label):
    t0 = time.time()
    cg = pc(data, alpha, chisq, show_progress=False)
    m = evaluate_graph(truth, cg)
    m["time_s"] = round(time.time() - t0, 3)
    m["alpha"] = alpha
    m["alpha_label"] = label
    print(f"    {label:<14} alpha={alpha:<12.3g} SHD={m['SHD']:<4} "
          f"adjP/R={m['adj_precision']}/{m['adj_recall']}  {m['time_s']}s")
    return m


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), DEFAULT_SUB))
    if not os.path.isdir(os.path.join(data_dir, "data")):
        sys.exit(f"找不到 bnlearn 数据目录: {data_dir}\n"
                 f"请先获取 causal-learn tests/TestData/bnlearn_discrete_10000/，或用参数指定")

    out = {"method": "PC+chisq, fixed-vs-Bonferroni", "n_samples": 10000,
           "alpha_fixed": ALPHA_FIXED, "alpha_mid_probe": ALPHA_MID,
           "alpha_bonf_formula": "0.05 / C(n,2)", "results": {}}

    for ds, n in DATASETS:
        print(f"\n[数据集] {ds}（{n} 节点）" + "=" * 40)
        data = np.loadtxt(os.path.join(data_dir, "data", f"{ds}.txt"), skiprows=1, dtype=int)
        truth = parse_truth_graph(os.path.join(data_dir, "truth_dag_graph",
                                               f"{ds}.graph.txt"), data.shape[1])
        ab = bonferroni_alpha(n)
        print(f"  n={n}, 真值边 {truth.sum()}, Bonferroni alpha={ab:.4g}")
        out["results"][ds] = {"n_nodes": n, "truth_edges": int(truth.sum()), "alpha_bonf": ab}
        for alpha, label, key in [(ALPHA_FIXED, "fixed=0.05", "fixed_0.05"),
                                  (ALPHA_MID, "mid=0.01", "mid_0.01"),
                                  (ab, f"bonf={ab:.3g}", "bonf")]:
            out["results"][ds][key] = run_pc(data, truth, alpha, label)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False, 无 NaN 字面量)")


if __name__ == "__main__":
    main()