# -*- coding: utf-8 -*-
"""
causal-lab 统一评估器（scripts/evaluate.py）
=============================================
在合成数据上评估因果发现方法: 真值 DAG -> CPDAG/PAG 对齐后计算
SHD + Adjacency precision/recall + Arrow precision/recall。

用法示例:
    from scripts.evaluate import evaluate_graph, graph_from_adj, truth_adj_to_cpdag
    metrics = evaluate_graph(truth_adj, est_graph)   # est_graph 为 causallearn 图对象

约定:
    - truth_adj: numpy 邻接矩阵 (i,j)=1 表示 i->j
    - est: 可以是 CausalGraph（PC/cdnod 返回）、GeneralGraph（GES/FCI/BOSS/GRaSP 返回）
    - 对齐语义: 约束型输出 CPDAG/PAG, 打分型输出 CPDAG, 函数模型输出 DAG
    - 无边/无箭头时 precision 置 0（causal-learn 的 Confusion 会 0/0 抛错或返回 nan）
"""
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.SHD import SHD
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion


def graph_from_adj(adj, node_names=None):
    """邻接矩阵 -> causallearn GeneralGraph（有向边）。"""
    n = len(adj)
    nodes = [GraphNode(node_names[i] if node_names else f"X{i + 1}") for i in range(n)]
    g = GeneralGraph(nodes)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                g.add_directed_edge(nodes[i], nodes[j])
    return g


def truth_adj_to_cpdag(truth_adj):
    """真值 DAG 邻接 -> CPDAG（评估对齐用）。"""
    return dag2cpdag(graph_from_adj(truth_adj))


def _to_general_graph(est):
    """统一取 GeneralGraph。"""
    return est.G if hasattr(est, "G") else est


def _safe_ratio(tp, denom):
    """tp/denom；0/0 或非有限时返回 0.0。

    无边/无真值时 precision/recall 语义缺失（causal-learn 的
    ArrowConfusion/AdjacencyConfusion 会抛 ZeroDivisionError 或返回 nan），
    统一置 0 以产出合法数值。批次 B 修复点。
    """
    if denom == 0:
        return 0.0
    v = float(tp) / float(denom)
    return round(v, 4) if np.isfinite(v) else 0.0


def evaluate_graph(truth_adj, est, truth_is_cpdag=False, verbose=False):
    """评估: 返回指标 dict。

    Args:
        truth_adj: 真值 DAG 邻接矩阵
        est: causallearn 图对象（CausalGraph 或 GeneralGraph）
        truth_is_cpdag: 若 True 则 truth_adj 已是 CPDAG 邻接（跳过转换）
    """
    est_g = _to_general_graph(est)
    truth_g = graph_from_adj(truth_adj)
    truth_cpdag = truth_g if truth_is_cpdag else dag2cpdag(truth_g)

    shd = SHD(truth_cpdag, est_g).get_shd()
    adj = AdjacencyConfusion(truth_cpdag, est_g)
    arr = ArrowConfusion(truth_cpdag, est_g)

    # 批次 B: 用计数 + _safe_ratio 计算 P/R，避免无边时 0/0 抛错/nan
    adj_tp, adj_fp, adj_fn = adj.get_adj_tp(), adj.get_adj_fp(), adj.get_adj_fn()
    arr_tp, arr_fp, arr_fn = arr.get_arrows_tp(), arr.get_arrows_fp(), arr.get_arrows_fn()
    m = {
        "SHD": int(shd),
        "adj_precision": _safe_ratio(adj_tp, adj_tp + adj_fp),
        "adj_recall": _safe_ratio(adj_tp, adj_tp + adj_fn),
        "adj_tp": int(adj_tp),
        "adj_fp": int(adj_fp),
        "adj_fn": int(adj_fn),
        "arrow_precision": _safe_ratio(arr_tp, arr_tp + arr_fp),
        "arrow_recall": _safe_ratio(arr_tp, arr_tp + arr_fn),
    }
    if verbose:
        print(f"SHD={m['SHD']}  adj P/R={m['adj_precision']}/{m['adj_recall']}  "
              f"arrow P/R={m['arrow_precision']}/{m['arrow_recall']}")
    return m


def run_and_evaluate(data, truth_adj, method_fn, verbose=True, **kwargs):
    """便捷包装: 跑方法 + 评估 + 计时。method_fn(data, **kwargs) -> est 图对象。"""
    import time
    t0 = time.time()
    est = method_fn(data, **kwargs)
    elapsed = time.time() - t0
    m = evaluate_graph(truth_adj, est)
    m["time_s"] = round(elapsed, 3)
    if verbose:
        print(f"  [{method_fn.__name__}] {elapsed:.2f}s  {m}")
    return m


if __name__ == "__main__":
    # 自测: 完美恢复的场景（PC + fisherz 在 5 节点线性高斯上应接近 0 SHD）
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.data_gen import simulate_linear_gaussian, DEFAULT_DAG
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    data, truth = simulate_linear_gaussian(n=30000, seed=42)
    cg = pc(data, 0.05, fisherz, show_progress=False)
    m = evaluate_graph(truth, cg, verbose=True)
    assert m["SHD"] == 0, f"完美场景 SHD 应为 0, 实际 {m['SHD']}"
    print("evaluate 自测通过 ✓")
