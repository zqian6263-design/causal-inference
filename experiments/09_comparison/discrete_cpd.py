# -*- coding: utf-8 -*-
"""
experiments/09_comparison/discrete_cpd.py — 真实离散 CPD 复核（IMPROVEMENTS #1）
===============================================================================
simulate_discrete() 是「连续高斯潜变量分位数分箱」，不是真实离散生成模型。
本脚本用 simulate_discrete_cpd()（多项 logit CPD，非高斯分箱）复核离散选型结论，
看「BOSS 离散最佳」在真实离散 CPD 下是否仍然成立。

方法（与 run_all.py 离散口径一致）:
  - BOSS + BIC(BIC_from_cov) / GRaSP + BIC / PC + chisq / GES + BDeu
  - 5 个 seed（42/1/7/2024/999），数据逐 seed 重生成，SHD 报 mean±std
  - 真值 DAG -> CPDAG 对齐 -> evaluate_graph（scripts.evaluate）

产出:
  - results/metrics/discrete_cpd.json   复核指标（含高斯分箱对照列）
"""
import sys, os, json, time, warnings, random
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import simulate_discrete, simulate_discrete_cpd, DEFAULT_DAG
from scripts.evaluate import evaluate_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss

N_SAMPLES = 3000
SEEDS = [42, 1, 7, 2024, 999]
N_NODES, N_EDGES = 5, len(DEFAULT_DAG)
OUT = os.path.join("results", "metrics", "discrete_cpd.json")


def _mean(xs):
    return round(float(np.mean(xs)), 4)


def _std(xs):
    return round(float(np.std(xs)), 4) if len(xs) > 1 else 0.0


def run_method(name, data, truth, runner, seed):
    """单 seed: runner(data, seed) -> est 图对象，返回单 seed 指标。"""
    random.seed(seed)  # GRaSP/BOSS 内部全局随机，须固定
    t0 = time.time()
    try:
        est = runner(data, seed)
        m = evaluate_graph(truth, est)
        m["time_s"] = round(time.time() - t0, 3)
        print(f"    {name:<12} seed={seed:<5} SHD={m['SHD']:<3} "
              f"adjP/R={m['adj_precision']}/{m['adj_recall']}  {m['time_s']}s")
        return m
    except Exception as e:
        print(f"    {name:<12} seed={seed:<5} FAILED: {str(e)[:80]}")
        return {"SHD": None, "error": str(e)[:120], "time_s": round(time.time() - t0, 3)}


def _aggregate(per_seed):
    ok = [m for m in per_seed if m.get("SHD") is not None]
    agg = {
        "SHD_mean": _mean([m["SHD"] for m in ok]),
        "SHD_std": _std([m["SHD"] for m in ok]),
        "SHD_per_seed": [int(m["SHD"]) for m in ok],
        "adj_precision": _mean([m["adj_precision"] for m in ok]),
        "adj_recall": _mean([m["adj_recall"] for m in ok]),
        "arrow_precision": _mean([m["arrow_precision"] for m in ok]),
        "arrow_recall": _mean([m["arrow_recall"] for m in ok]),
        "time_s_mean": round(float(np.mean([m["time_s"] for m in ok])), 3),
    }
    return agg


def run_dataset(dgen, dlabel):
    matrix = {}
    methods = [
        ("PC+chisq", lambda d, s: pc(d, 0.05, chisq, show_progress=False)),
        ("GES+BDeu", lambda d, s: ges(d, score_func="local_score_BDeu")["G"]),
        ("GRaSP+BIC", lambda d, s: grasp(d, score_func="local_score_BIC_from_cov", verbose=False)),
        ("BOSS+BIC", lambda d, s: boss(d, score_func="local_score_BIC_from_cov", verbose=False)),
    ]
    for mname, runner in methods:
        per_seed = []
        for seed in SEEDS:
            data, truth = dgen(n=N_SAMPLES, seed=seed)
            per_seed.append(run_method(mname, data, truth, runner, seed))
        matrix[mname] = _aggregate(per_seed)
        a = matrix[mname]
        print(f"    -> {mname:<12} SHD={a['SHD_mean']}±{a['SHD_std']}  "
              f"per_seed={a['SHD_per_seed']}  time={a['time_s_mean']}s")
    return matrix


def main():
    print(f"真实离散 CPD 复核（多项 logit, {N_SAMPLES} 样本, {N_NODES} 节点 {N_EDGES} 边, "
          f"seeds={SEEDS}）\n" + "=" * 72)
    out = {
        "n_samples": N_SAMPLES, "seeds": SEEDS, "n_nodes": N_NODES, "n_edges": N_EDGES,
        "description": "离散选型复核: GSM(高斯分箱,现有 simulate_discrete) vs 真实CPD(多项logit, simulate_discrete_cpd)",
    }
    print("\n[列 1] 真实离散 CPD（多项 logit）")
    out["discrete_cpd_logit"] = run_dataset(simulate_discrete_cpd, "discrete_cpd_logit")
    print("\n[对照] 高斯分箱离散（现有口径）")
    out["discrete_gaussian_binned"] = run_dataset(simulate_discrete, "discrete_gaussian_binned")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False, 无 NaN 字面量)")


if __name__ == "__main__":
    main()
