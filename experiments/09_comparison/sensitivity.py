# -*- coding: utf-8 -*-
"""
experiments/09_comparison/sensitivity.py — 样本量敏感性扫描（IMPROVEMENTS #4）
===============================================================================
批次 B 暴露 3000 样本下 PC/GES 线性高斯不稳（4.8±2.5）。本脚本按样本量梯度
[1000, 3000, 10000, 30000] 扫描方法稳定性，产出「SHD 随样本量变化」证据表，
供 knowledge/08 按样本量分档建议。

方法（线性高斯 / 线性非高斯两行数据）:
  - PC + fisherz / GES + BIC / GRaSP + BIC / BOSS + BIC
  - ICA-LiNGAM 仅非高斯行（前提吻合）
  - 每格 3 个 seed（42/1/7），数据逐 seed 重生成，SHD 报 mean±std

产出:
  - results/metrics/sensitivity.json          全指标（合法 JSON，无 NaN）
  - results/sensitivity_report.md             可读表（SHD 随样本量变化）
"""
import sys, os, json, time, warnings, random
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import simulate_linear_gaussian, simulate_linear_nongaussian, DEFAULT_DAG
from scripts.evaluate import evaluate_graph, graph_from_adj
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.FCMBased import lingam
from causallearn.utils.DAG2CPDAG import dag2cpdag

N_SAMPLES_GRID = [1000, 3000, 10000, 30000]
SEEDS = [42, 1, 7]
N_NODES, N_EDGES = 5, len(DEFAULT_DAG)
LINGAM_THRESH = 0.1
OUT_METRICS = os.path.join("results", "metrics", "sensitivity.json")
OUT_REPORT = os.path.join("results", "sensitivity_report.md")


def _mean(xs):
    return round(float(np.mean(xs)), 4)


def _std(xs):
    return round(float(np.std(xs)), 4) if len(xs) > 1 else 0.0


def run_method(name, data, truth, runner, seed):
    random.seed(seed)  # GRaSP/BOSS 内部全局随机
    t0 = time.time()
    try:
        est = runner(data, seed)
        m = evaluate_graph(truth, est)
        m["time_s"] = round(time.time() - t0, 3)
        return m
    except Exception as e:
        return {"SHD": None, "error": str(e)[:120], "time_s": round(time.time() - t0, 3)}


def _run_lingam(data, seed):
    """ICA-LiNGAM -> CPDAG（同 run_all.py 口径，系数阈值 0.1 -> 邻接掩码 .T -> dag2cpdag）。"""
    m = lingam.ICALiNGAM(random_state=seed)
    m.fit(data)
    mask = np.abs(m.adjacency_matrix_.T) > LINGAM_THRESH
    return dag2cpdag(graph_from_adj(mask))


def _aggregate(per_seed):
    ok = [m for m in per_seed if m.get("SHD") is not None]
    agg = {
        "SHD_mean": _mean([m["SHD"] for m in ok]),
        "SHD_std": _std([m["SHD"] for m in ok]),
        "SHD_per_seed": [int(m["SHD"]) for m in ok],
        "time_s_mean": round(float(np.mean([m["time_s"] for m in ok])), 3),
    }
    if len(ok) < len(per_seed):
        agg["n_failed"] = len(per_seed) - len(ok)
        agg["errors"] = [m.get("error", "?")[:120] for m in per_seed if m.get("SHD") is None]
    return agg


def main():
    datasets = {
        "linear_gaussian": {
            "gen": simulate_linear_gaussian,
            "methods": [
                ("PC+fisherz", lambda d, s: pc(d, 0.05, fisherz, show_progress=False)),
                ("GES+BIC", lambda d, s: ges(d)["G"]),
                ("GRaSP+BIC", lambda d, s: grasp(d, score_func="local_score_BIC_from_cov", verbose=False)),
                ("BOSS+BIC", lambda d, s: boss(d, score_func="local_score_BIC_from_cov", verbose=False)),
            ],
        },
        "linear_nongaussian": {
            "gen": simulate_linear_nongaussian,
            "methods": [
                ("PC+fisherz", lambda d, s: pc(d, 0.05, fisherz, show_progress=False)),
                ("GES+BIC", lambda d, s: ges(d)["G"]),
                ("GRaSP+BIC", lambda d, s: grasp(d, score_func="local_score_BIC_from_cov", verbose=False)),
                ("BOSS+BIC", lambda d, s: boss(d, score_func="local_score_BIC_from_cov", verbose=False)),
                ("ICA-LiNGAM", _run_lingam),
            ],
        },
    }
    out = {"datasets": {}, "n_samples_grid": N_SAMPLES_GRID, "seeds": SEEDS,
           "n_nodes": N_NODES, "n_edges": N_EDGES}

    for dname, cfg in datasets.items():
        print(f"\n{'=' * 70}\n数据: {dname}  samples={N_SAMPLES_GRID}  seeds={SEEDS}\n{'=' * 70}")
        out["datasets"][dname] = {}
        for ns in N_SAMPLES_GRID:
            out["datasets"][dname][str(ns)] = {}
            print(f"  n_samples = {ns}")
            for mname, runner in cfg["methods"]:
                per_seed = []
                for seed in SEEDS:
                    data, truth = cfg["gen"](n=ns, seed=seed)
                    per_seed.append(run_method(mname, data, truth, runner, seed))
                a = _aggregate(per_seed)
                out["datasets"][dname][str(ns)][mname] = a
                v = "失败" if a["SHD_mean"] is None else f"{a['SHD_mean']}±{a['SHD_std']}"
                print(f"    {mname:<12} SHD={v}  {a.get('time_s_mean')}s")

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT_METRICS}  (allow_nan=False, 无 NaN 字面量)")

    # ---- Markdown 报告（SHD 随样本量变化表）----
    lines = [
        "# 样本量敏感性扫描（SHD 随样本量变化）\n",
        f"- 数据: {N_NODES} 节点 {N_EDGES} 边 DAG（TestPC 基准图），样本量梯度 {N_SAMPLES_GRID}",
        f"- 每格 3 个 seed（{SEEDS}），数据逐 seed 重生成；SHD 报 mean±std（越小越好），时间为平均",
        "- 方法: PC+fisherz / GES+BIC / GRaSP+BIC / BOSS+BIC（两行数据）；ICA-LiNGAM 仅线性非高斯（前提吻合）\n",
    ]
    for dname, ds in out["datasets"].items():
        lines.append(f"## {dname}\n")
        # 表头: 方法 | SHD@1000 | SHD@3000 | SHD@10000 | SHD@30000
        mnames = []
        for ns in N_SAMPLES_GRID:
            for k in ds[str(ns)]:
                if k not in mnames:
                    mnames.append(k)
        lines.append("| 方法 | " + " | ".join(f"SHD@{ns}" for ns in N_SAMPLES_GRID) + " |")
        lines.append("|---" * (len(N_SAMPLES_GRID) + 1) + "|")
        for m in mnames:
            cells = []
            for ns in N_SAMPLES_GRID:
                a = ds[str(ns)].get(m)
                if a is None or a["SHD_mean"] is None:
                    cells.append("失败")
                else:
                    cells.append(f"{a['SHD_mean']:.1f}±{a['SHD_std']:.1f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已落盘: {OUT_REPORT}")


if __name__ == "__main__":
    main()