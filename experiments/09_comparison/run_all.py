# -*- coding: utf-8 -*-
"""
Phase 2/3 统一对比实验: 4 类数据 × 全方法 × 5 seeds（批次 B 改造版）
=====================================================================
用法: cd causal-lab && python experiments/09_comparison/run_all.py

批次 B 改造（修实证证据矛盾）:
  ① 每个方法跑 5 个 seed（42/1/7/2024/999），SHD 报 mean±std（数据逐 seed 重生成）
  ② GRaSP/BOSS 在离散数据上显式对比 score_func: BIC_from_cov(默认) vs BDeu
     —— 修正「BOSS 碾压归因 BDeu」的错误结论（此前未显式传参、实际用的是默认 BIC_from_cov）
  ③ 修 ArrowConfusion/AdjacencyConfusion 无边 0/0: precision/recall 置 0
     （根因在 scripts/evaluate.py，已在该处统一修复；本脚本 JSON 落盘 allow_nan=False 兜底）
  ④ ICA-LiNGAM 纳入统一矩阵: adjacency_matrix_ 阈值 0.1 -> 邻接掩码 -> graph_from_adj
     -> dag2cpdag（与 PC/GES/GRaSP/BOSS 同为 CPDAG 口径）-> evaluate_graph 算 SHD
  ⑤ 输出合法 JSON（无 NaN 字面量）+ 更新 comparison_report.md

产出:
  - results/metrics/comparison.json   全指标矩阵（多 seed 聚合）
  - results/comparison_report.md      可读报告（mean±std）
"""
import sys, os, json, time, warnings, random
warnings.filterwarnings("ignore", category=FutureWarning)  # BDeu 内部 pandas 分组弃用提示，与实验无关
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import (simulate_linear_gaussian, simulate_linear_nongaussian,
                              simulate_nonlinear_anm, simulate_discrete, DEFAULT_DAG)
from scripts.evaluate import evaluate_graph, graph_from_adj
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.FCMBased import lingam
from causallearn.utils.DAG2CPDAG import dag2cpdag

N_SAMPLES = 3000
SEEDS = [42, 1, 7, 2024, 999]
N_NODES, N_EDGES = 5, len(DEFAULT_DAG)
LINGAM_THRESH = 0.1          # ICA-LiNGAM adjacency_matrix_ 系数阈值
OUT_METRICS = os.path.join("results", "metrics", "comparison.json")
OUT_REPORT = os.path.join("results", "comparison_report.md")


def _mean(xs):
    return round(float(np.mean(xs)), 4)


def _std(xs):
    return round(float(np.std(xs)), 4) if len(xs) > 1 else 0.0


def run_method(name, data, truth, runner, seed):
    """跑一个 seed: runner(data, seed) -> est 图对象，返回单 seed 指标 dict。

    random.seed(seed)：GRaSP/BOSS 内部用全局 random.shuffle（无 API 参数），
    不固定则每次运行结果都变、证据不可复现；按 (方法, seed) 固定后整次运行可复现。
    """
    random.seed(seed)
    t0 = time.time()
    try:
        est = runner(data, seed)
        m = evaluate_graph(truth, est)
        m["time_s"] = round(time.time() - t0, 3)
        print(f"    {name:<14} seed={seed:<5} SHD={m['SHD']:<3} "
              f"adjP/R={m['adj_precision']}/{m['adj_recall']}  "
              f"arrP/R={m['arrow_precision']}/{m['arrow_recall']}  {m['time_s']}s")
        return m
    except Exception as e:
        print(f"    {name:<14} seed={seed:<5} FAILED: {str(e)[:80]}")
        return {"SHD": None, "error": str(e)[:120], "time_s": round(time.time() - t0, 3)}


def _run_lingam(data, seed):
    """ICA-LiNGAM -> CPDAG 图（与打分/约束方法同口径）。

    lingam.adjacency_matrix_[i,j] 是 X_j 在 X_i 方程中的系数 -> 边 j→i，
    即相对本仓库 adj[i,j]=边 i→j 约定是转置的，必须先 .T（已实测校验 mask.T==真值）。
    阈值 0.1 -> 邻接掩码 -> graph_from_adj 得完整 DAG，再 dag2cpdag 对齐到 CPDAG
    ——LiNGAM 输出完整 DAG，直接对比真值 CPDAG 会因「比等价类更具体」而虚高 SHD，
    其余方法均按 CPDAG 评估，需同口径。
    """
    m = lingam.ICALiNGAM(random_state=seed)
    m.fit(data)
    mask = np.abs(m.adjacency_matrix_.T) > LINGAM_THRESH
    return dag2cpdag(graph_from_adj(mask))


def _aggregate(per_seed):
    """5 个单 seed 指标 -> 聚合 dict（SHD mean±std + P/R 平均）。"""
    ok = [m for m in per_seed if m.get("SHD") is not None]
    if not ok:
        return {"SHD_mean": None,
                "error": "; ".join(m.get("error", "?")[:80] for m in per_seed if m)}
    agg = {
        "SHD_mean": _mean([m["SHD"] for m in ok]),
        "SHD_std": _std([m["SHD"] for m in ok]),
        "SHD_per_seed": [int(m["SHD"]) for m in ok],
        "adj_precision": _mean([m["adj_precision"] for m in ok]),
        "adj_recall": _mean([m["adj_recall"] for m in ok]),
        "arrow_precision": _mean([m["arrow_precision"] for m in ok]),
        "arrow_recall": _mean([m["arrow_recall"] for m in ok]),
        "time_s_mean": round(float(np.mean([m["time_s"] for m in ok])), 3),
        "time_s_per_seed": [round(float(m["time_s"]), 3) for m in ok],
    }
    n_fail = len(per_seed) - len(ok)
    if n_fail:
        agg["n_failed"] = n_fail
        agg["errors"] = [m.get("error", "?")[:80] for m in per_seed if m.get("SHD") is None]
    return agg


def main():
    generators = {
        "linear_gaussian": simulate_linear_gaussian,
        "linear_nongaussian": simulate_linear_nongaussian,
        "nonlinear_anm": simulate_nonlinear_anm,
        "discrete": simulate_discrete,
    }
    matrix = {}
    for dname, dgen in generators.items():
        print(f"\n{'=' * 72}\n数据集: {dname}  ({N_SAMPLES} 样本, {N_NODES} 节点 {N_EDGES} 边, seeds={SEEDS})\n{'=' * 72}")
        matrix[dname] = {}
        is_discrete = dname == "discrete"
        indep = chisq if is_discrete else fisherz

        methods = []  # (显示名, runner(data, seed) -> est 图对象)
        methods.append((f"PC+{'chisq' if is_discrete else 'fisherz'}",
                        lambda d, s, it=indep: pc(d, 0.05, it, show_progress=False)))
        methods.append(("GES+BDeu" if is_discrete else "GES+BIC",
                        lambda d, s: ges(d, score_func="local_score_BDeu")["G"] if is_discrete else ges(d)["G"]))
        methods.append(("GRaSP+BIC", lambda d, s: grasp(d, score_func="local_score_BIC_from_cov", verbose=False)))
        methods.append(("BOSS+BIC", lambda d, s: boss(d, score_func="local_score_BIC_from_cov", verbose=False)))
        if is_discrete:  # 显式对比 score_func: BIC_from_cov(默认) vs BDeu
            methods.append(("GRaSP+BDeu", lambda d, s: grasp(d, score_func="local_score_BDeu", verbose=False)))
            methods.append(("BOSS+BDeu", lambda d, s: boss(d, score_func="local_score_BDeu", verbose=False)))
        if dname == "linear_nongaussian":  # LiNGAM 前提：线性非高斯
            methods.append(("ICA-LiNGAM", _run_lingam))

        for mname, runner in methods:
            per_seed = []
            for seed in SEEDS:
                data, truth = dgen(n=N_SAMPLES, seed=seed)
                per_seed.append(run_method(mname, data, truth, runner, seed))
            matrix[dname][mname] = _aggregate(per_seed)
            a = matrix[dname][mname]
            if a.get("SHD_mean") is not None:
                print(f"    -> {mname:<14} SHD={a['SHD_mean']}±{a['SHD_std']}  "
                      f"arrP/R={a['arrow_precision']}/{a['arrow_recall']}  time={a['time_s_mean']}s")
            else:
                print(f"    -> {mname:<14} ALL FAILED: {a.get('error')}")

    # ---- 落盘 JSON（合法 JSON，无 NaN 字面量）----
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    payload = {"n_samples": N_SAMPLES, "seeds": SEEDS, "n_nodes": N_NODES, "n_edges": N_EDGES,
               "matrix": matrix}
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT_METRICS}  (allow_nan=False, 无 NaN 字面量)")

    # ---- Markdown 报告 ----
    lines = [
        "# 方法 × 数据 对比矩阵（多 seed 口径，Phase 3 输入）\n",
        f"- 数据: {N_NODES} 节点 {N_EDGES} 边 DAG（TestPC 基准图），{N_SAMPLES} 样本",
        f"- 每个方法跑 5 个 seed（{SEEDS}），数据逐 seed 重生成；SHD 报 mean±std（越小越好），P/R 与时间为 5 seed 平均",
        "- GRaSP/BOSS 在离散数据显式对比 score_func：BIC_from_cov（默认）vs BDeu（此前误把默认当 BDeu）",
        "- ICA-LiNGAM 纳入统一矩阵：系数阈值 0.1 → 邻接掩码 → dag2cpdag（与其余方法同 CPDAG 口径）→ evaluate_graph",
        "- Arrow/Adjacency Confusion 无边时 precision 置 0（evaluate.py 根因修复）\n",
        "| 数据 | 方法 | SHD(mean±std) | adjP | adjR | arrP | arrR | 时间(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dname, methods in matrix.items():
        for mname, m in methods.items():
            if m.get("SHD_mean") is None:
                lines.append(f"| {dname} | {mname} | 失败 | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {dname} | {mname} | {m['SHD_mean']:.2f}±{m['SHD_std']:.2f} | "
                    f"{m['adj_precision']:.2f} | {m['adj_recall']:.2f} | "
                    f"{m['arrow_precision']:.2f} | {m['arrow_recall']:.2f} | {m['time_s_mean']:.3f} |")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已落盘: {OUT_REPORT}")


if __name__ == "__main__":
    main()
