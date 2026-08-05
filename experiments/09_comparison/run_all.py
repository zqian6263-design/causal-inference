# -*- coding: utf-8 -*-
"""
Phase 2/3 统一对比实验: 4 类数据 × 全方法
============================================
用法: cd D:/win/causal-lab && PYTHONPATH= D:/Anaconda/envs/pytorch/python.exe experiments/09_comparison/run_all.py

产出:
  - results/metrics/comparison.json   全指标矩阵
  - results/comparison_report.md      可读报告（速查表升级版）
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import (simulate_linear_gaussian, simulate_linear_nongaussian,
                              simulate_nonlinear_anm, simulate_discrete)
from scripts.evaluate import evaluate_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.FCMBased import lingam

N_SAMPLES = 3000
SEED = 42
OUT_METRICS = os.path.join("results", "metrics", "comparison.json")
OUT_REPORT = os.path.join("results", "comparison_report.md")


def run_method(name, data, truth, fn, **kw):
    t0 = time.time()
    try:
        est = fn(data, **kw)
        m = evaluate_graph(truth, est)
        m["time_s"] = round(time.time() - t0, 3)
        print(f"    {name:<18} SHD={m['SHD']:<3} adjP/R={m['adj_precision']}/{m['adj_recall']}  "
              f"arrowP/R={m['arrow_precision']}/{m['arrow_recall']}  {m['time_s']}s")
        return m
    except Exception as e:
        print(f"    {name:<18} FAILED: {str(e)[:80]}")
        return {"SHD": None, "error": str(e)[:100]}


def main():
    datasets = {
        "linear_gaussian": simulate_linear_gaussian(n=N_SAMPLES, seed=SEED),
        "linear_nongaussian": simulate_linear_nongaussian(n=N_SAMPLES, seed=SEED),
        "nonlinear_anm": simulate_nonlinear_anm(n=N_SAMPLES, seed=SEED),
        "discrete": simulate_discrete(n=N_SAMPLES, seed=SEED),
    }

    matrix = {}
    for dname, (data, truth) in datasets.items():
        print(f"\n{'='*70}\n数据集: {dname}  (3000 样本, 5 节点 7 边)\n{'='*70}")
        matrix[dname] = {}
        is_discrete = dname == "discrete"

        # 约束型: PC + fisherz（离散用 chisq）
        indep = chisq if is_discrete else fisherz
        matrix[dname]["PC+fisherz" if not is_discrete else "PC+chisq"] = run_method(
            "PC", data, truth, lambda d, it=indep: pc(d, 0.05, it, show_progress=False))

        # 打分型: GES + BIC（离散用 BDeu）
        if is_discrete:
            matrix[dname]["GES+BDeu"] = run_method(
                "GES+BDeu", data, truth, lambda d: ges(d, score_func="local_score_BDeu")["G"])
        else:
            matrix[dname]["GES+BIC"] = run_method(
                "GES+BIC", data, truth, lambda d: ges(d)["G"])

        # 排列型
        matrix[dname]["GRaSP"] = run_method("GRaSP", data, truth, grasp)
        matrix[dname]["BOSS"] = run_method("BOSS", data, truth, boss)

        # 函数模型: 仅连续非高斯数据用 LiNGAM
        if dname == "linear_nongaussian":
            def run_lingam(d):
                m = lingam.ICALiNGAM(random_state=SEED)
                m.fit(d)
                return np.abs(m.adjacency_matrix_) > 0.1  # 邻接掩码 → 当 GeneralGraph 用
            matrix[dname]["ICA-LiNGAM"] = {"SHD": None, "note": "输出为系数矩阵,单独解读"}
            t0 = time.time()
            try:
                m = lingam.ICALiNGAM(random_state=SEED)
                m.fit(data)
                order = m.causal_order_
                nz = np.count_nonzero(np.abs(m.adjacency_matrix_) > 0.1)
                print(f"    ICA-LiNGAM           order={order} 非零系数={nz}  "
                      f"{(time.time()-t0):.2f}s")
                matrix[dname]["ICA-LiNGAM"] = {"causal_order": list(order), "n_edges": int(nz),
                                               "time_s": round(time.time() - t0, 3)}
            except Exception as e:
                matrix[dname]["ICA-LiNGAM"]["error"] = str(e)[:100]

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump({"n_samples": N_SAMPLES, "seed": SEED, "matrix": matrix}, f,
                  ensure_ascii=False, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n指标已落盘: {OUT_METRICS}")

    # Markdown 报告
    lines = ["# 方法 × 数据 对比矩阵（Phase 3 输入）\n",
             f"- 数据: 5 节点 7 边 DAG（TestPC 基准图）, {N_SAMPLES} 样本, seed={SEED}",
             "- 指标: SHD（越小越好）/ Adjacency P/R / Arrow P/R / 时间\n",
             "| 数据 | 方法 | SHD | adjP | adjR | arrP | arrR | 时间(s) |", "|---|---|---|---|---|---|---|---|"]
    for dname, methods in matrix.items():
        for mname, m in methods.items():
            if m.get("SHD") is None:
                lines.append(f"| {dname} | {mname} | - | - | - | - | - | {m.get('time_s','-')} |")
            else:
                lines.append(f"| {dname} | {mname} | {m['SHD']} | {m['adj_precision']} | {m['adj_recall']} | "
                             f"{m['arrow_precision']} | {m['arrow_recall']} | {m['time_s']} |")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已落盘: {OUT_REPORT}")


if __name__ == "__main__":
    main()
