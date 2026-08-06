# -*- coding: utf-8 -*-
"""
experiments/08_benchmarks/smoke_asia.py — bnlearn asia 冒烟（PC + chisq）
=========================================================================
asia: 8 节点离散贝叶斯网络基准（真值图 8 条边, 见 truth_dag_graph/asia.graph.txt）。
数据源: causal-learn 官方仓库 tests/TestData/bnlearn_discrete_10000/（见本目录 README.md）。

用法:
  python experiments/08_benchmarks/smoke_asia.py               # 默认找仓库同级 causal-learn/
  python experiments/08_benchmarks/smoke_asia.py <bnlearn数据目录>

产出:
  - results/metrics/08_benchmarks_asia.json   指标（真值 DAG -> CPDAG 对齐）
  - results/figs/08_benchmarks_asia.png       PC+chisq 估计 CPDAG
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.evaluate import evaluate_graph
from scripts.plotting import plot_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

# 默认: 仓库同级目录的 causal-learn 官方克隆（D:\win\causal-learn 的通用写法）
DEFAULT_SUB = os.path.join("..", "..", "..", "causal-learn", "tests", "TestData",
                           "bnlearn_discrete_10000")
OUT = os.path.join("results", "metrics", "08_benchmarks_asia.json")
FIG = os.path.join("results", "figs", "08_benchmarks_asia.png")


def parse_truth_graph(path, n):
    """解析 truth_dag_graph/*.graph.txt（形如 '1. X1 --> X2'）-> 邻接矩阵。"""
    adj = np.zeros((n, n), dtype=int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "-->" in line:
                arrow = line.split("-->")
                u = arrow[0].strip().split()[-1]   # 如 '1. X1' -> 'X1'
                v = arrow[1].strip().split()[0]    # 如 'X2'
                adj[int(u[1:]) - 1, int(v[1:]) - 1] = 1
    return adj


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), DEFAULT_SUB))
    data_path = os.path.join(data_dir, "data", "asia.txt")
    truth_path = os.path.join(data_dir, "truth_dag_graph", "asia.graph.txt")
    if not os.path.exists(data_path):
        sys.exit(f"找不到 asia 数据: {data_path}\n"
                 f"请按本目录 README.md 获取 bnlearn 数据（causal-learn tests/TestData/），"
                 f"或用参数指定数据目录")

    data = np.loadtxt(data_path, skiprows=1, dtype=int)
    n_nodes = data.shape[1]
    truth = parse_truth_graph(truth_path, n_nodes)
    print(f"asia 数据: {data.shape[0]} 样本 x {n_nodes} 节点（离散 1/2）; 真值 {truth.sum()} 条有向边")

    t0 = time.time()
    cg = pc(data, 0.05, chisq, show_progress=False)
    elapsed = round(time.time() - t0, 3)
    m = evaluate_graph(truth, cg)
    m["time_s"] = elapsed
    print(f"PC+chisq: SHD={m['SHD']}  adjP/R={m['adj_precision']}/{m['adj_recall']}  "
          f"arrP/R={m['arrow_precision']}/{m['arrow_recall']}  {elapsed}s")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"dataset": "asia (8 nodes, discrete bnlearn)", "n_samples": int(data.shape[0]),
                   "method": "PC+chisq", **m}, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plot_graph(cg.G, FIG, title="asia benchmark: PC+chisq CPDAG")
    print(f"指标落盘: {OUT}\n图落盘: {FIG}")


if __name__ == "__main__":
    main()
