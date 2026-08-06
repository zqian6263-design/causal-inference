# -*- coding: utf-8 -*-
"""
experiments/02_fci/cdnod_example.py — CD-NOD 示例（时变数据 + c_indx）
======================================================================
CD-NOD（Causal Discovery from Nonstationary/Heterogeneous Data）把「时间/域索引」
当作虚拟节点 c_indx 拼进数据，跑增广 PC——机制在变化的变量会与 c_indx 连边。

本示例构造**机制分段切换**的时变数据（knowledge/02 示例，逐字一致）：
    x2 = 0.9*x1 + e（前段, 强因果）;  x2 = 0.1*x1 + e（后段, 弱因果）
期望：CD-NOD 既恢复 X1 -> X2，又用 C -> X2 指出「X2 的机制在变」——这是它相对
普通 PC 的增量价值。

用法: cd causal-lab && python experiments/02_fci/cdnod_example.py
产出:
  - results/metrics/02_fci_cdnod.json
  - results/figs/02_fci_cdnod.png
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.plotting import plot_graph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci

T = 600
SEED = 42
OUT = os.path.join("results", "metrics", "02_fci_cdnod.json")
FIG_DIR = os.path.join("results", "figs")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- 机制分段切换的时变数据（前段强因果 / 后段弱因果）----
    rng = np.random.RandomState(SEED)
    x1 = rng.randn(T)
    x2 = np.concatenate([0.9 * x1[:T // 2] + 0.5 * rng.randn(T // 2),   # 前段: 强因果
                         0.1 * x1[T // 2:] + 0.5 * rng.randn(T - T // 2)])  # 后段: 弱因果
    data = np.column_stack([x1, x2])
    c_indx = np.arange(T).reshape(-1, 1)     # 时间索引 (n,1)
    print(f"时变数据: {T} 个时间点, X1->X2 机制在 T/2 处分段切换; c_indx = 时间索引")

    # ---- CD-NOD（时间索引连续值, 官方建议 kci）----
    import io, contextlib
    t0 = time.time()
    # cdnod 内部进度条写 stderr 会污染输出, 调用时静默捕获；出错则回放后抛出
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_buf):
            cg = cdnod(data, c_indx, 0.05, kci, True, 0, -1, show_progress=False)
    except Exception:
        err_buf.seek(0)
        sys.stderr.write(err_buf.read())
        raise
    elapsed = round(time.time() - t0, 3)

    print(f"\nCD-NOD 耗时: {elapsed}s")
    print("图矩阵（graph[i,j]; 最后一列 = c_indx 虚拟节点）:")
    print(cg.G.graph)

    # 解码：c_indx 是最后一个节点（此处名为 X3），统一改标为 C 便于理解
    nodes = cg.G.get_nodes()
    def node_label(node):
        return "C" if node is nodes[-1] else node.get_name()
    directed = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j or not cg.G.is_adjacent_to(a, b):
                continue
            ep_a = str(cg.G.get_endpoint(b, a)).split(".")[-1]  # a 端
            ep_b = str(cg.G.get_endpoint(a, b)).split(".")[-1]  # b 端
            if (ep_a, ep_b) == ("TAIL", "ARROW"):
                directed.append(f"{node_label(a)} -> {node_label(b)}")
    c_to_x2 = "C -> X2" in directed
    print("有向边:", directed)
    print(f"关键验证: C -> X2 已识别（X2 机制在变）= {c_to_x2}")

    results = {
        "data": "time-varying: x2 mechanism switches at T/2 (0.9 -> 0.1 coefficient)",
        "n_time": T, "indep_test": "kci", "time_s": elapsed,
        "directed_edges": directed,
        "C->X2_detected": c_to_x2,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n指标落盘: {OUT}")

    plot_graph(cg.G, os.path.join(FIG_DIR, "02_fci_cdnod.png"),
               title="CD-NOD: X1->X2 with changing mechanism (C = time index)")


if __name__ == "__main__":
    main()
