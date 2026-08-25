# -*- coding: utf-8 -*-
"""
experiments/09_comparison/kci_large.py — KCI 大样本补充（IMP #2，F3 轮）
========================================================================
现状锚点（results/metrics/kci_supplement.json，n=1200、5 节点 7 边 TestPC 图）：
  PC+KCI 非线性 ANM SHD=7 / 75.2s，线性高斯 SHD=6 / 104.7s——小样本下 KCI 漏边/定向失败。
本脚本在更大样本量下复测：样本量增大能否让 KCI 的非参数检验力恢复。

用法:
  python experiments/09_comparison/kci_large.py                  # 默认 --samples 1500 冒烟
  python experiments/09_comparison/kci_large.py --samples 3000   # 3000（约 5-15 分钟/数据集）
  python experiments/09_comparison/kci_large.py --samples 5000   # 5000（Hermes 后台扩跑）

约定:
  - KCI 缓存按数据 hash 校验，缓存文件名内嵌样本量（cache_kci_large_<数据>_<n>.json），
    换 n 直接出现 "Data hash mismatch"（见 01_pc/run.py 既有约定）
  - 评估与其余方法同尺子：真值 DAG → CPDAG 对齐 → evaluate.py
产出:
  results/metrics/kci_large.json   各数据集 × 样本量 KCI 指标（metadata 标注留跑样本量）
"""
import sys, os, json, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import simulate_nonlinear_anm, simulate_linear_gaussian, DEFAULT_DAG
from scripts.evaluate import evaluate_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci

N_NODES, N_EDGES = 5, len(DEFAULT_DAG)
OUT = os.path.join("results", "metrics", "kci_large.json")
CACHE_DIR = os.path.join("results", "metrics")


def run_pc_kci(dataset, n, seed=42):
    """生成 (非线性 ANM / 线性高斯) 数据 → PC+kci → 评估。返回指标 dict。"""
    if dataset == "nonlinear_anm":
        data, truth = simulate_nonlinear_anm(n=n, seed=seed)
    elif dataset == "linear_gaussian":
        data, truth = simulate_linear_gaussian(n=n, seed=seed)
    else:
        raise ValueError(dataset)
    cache_path = os.path.join(CACHE_DIR, f"cache_kci_large_{dataset}_n{n}.json")
    t0 = time.time()
    cg = pc(data, 0.05, kci, show_progress=False, cache_path=cache_path)
    m = evaluate_graph(truth, cg)
    m["time_s"] = round(time.time() - t0, 3)
    m["dataset"] = dataset
    m["n_samples"] = int(n)
    m["seed"] = seed
    print(f"  {dataset:<16} n={n:<5} SHD={m['SHD']:<3} "
          f"adjP/R={m['adj_precision']}/{m['adj_recall']}  {m['time_s']}s", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1500, help="样本量（1500 冒烟；3000/5000 后台重算）")
    ap.add_argument("--datasets", nargs="+", default=["nonlinear_anm", "linear_gaussian"])
    args = ap.parse_args()

    print(f"KCI 大样本复测: n={args.samples}, 数据集={args.datasets}, "
          f"{N_NODES} 节点 {N_EDGES} 边（TestPC 图）\n" + "=" * 64)

    # 累积合并：已有结果不覆盖，新样本量写进 results["<n>"]（Hermes 扩跑 3000/5000 可增量入库）
    out = {"n_nodes": N_NODES, "n_edges": N_EDGES, "alpha": 0.05,
           "note": "KCI 大样本补充；按样本量累积（1500 本机冒烟，3000/5000 由 Hermes 后台增量入库）",
           "results": {}}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
        out["results"] = prev.get("results", {})

    key = str(args.samples)
    out["results"][key] = {}
    for ds in args.datasets:
        out["results"][key][ds] = run_pc_kci(ds, args.samples)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False；现有 {sorted(out['results'])} 个样本量累积)")


if __name__ == "__main__":
    main()