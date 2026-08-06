# -*- coding: utf-8 -*-
"""
experiments/02_fci/run.py — FCI 专属实验（允许隐变量，输出 PAG）
================================================================
FCI（Fast Causal Inference）放宽 PC 的「因果充分性」假设，允许未观测共同原因，
输出 PAG（部分祖先图）——用圆圈端点表达「可能存在隐变量/方向未定」。

真值结构：隐变量 L 混淆观测 X1、X2（L→X1, L→X2）；另 X1→X3, X2→X4, X3→X5, X4→X5。
观测数据只含 X1..X5（n=3000, seed=42），L 不出现在数据里。

评估（PAG 语义，重点）：
  - FCI 输出 PAG，真值必须用 dag2pag 对齐（causallearn.utils.DAG2PAG.dag2pag），
    **不能 dag2cpdag**——CPDAG 表达不了圆圈端点，语义错位会让 SHD 虚高。
  - SHD 在两 PAG 间计数「端点不匹配」（含圆圈 vs 箭头），比 CPDAG SHD 更严格：
    保守的圆圈是「不确定」而非错误，故同时给 Adjacency P/R 作结构主指标。

用法: cd causal-lab && python experiments/02_fci/run.py
产出:
  - results/metrics/02_fci.json              PAG 对齐指标 + 解码边列表
  - results/figs/02_fci_truth_pag.png        真值 PAG（L 隐变量）
  - results/figs/02_fci_est_pag.png          FCI 估计 PAG
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

import scripts.patch_d_separation  # dag2pag 依赖 nx.is_d_separator（networkx 3.2.1 补丁）

from scripts.evaluate import _safe_ratio
from scripts.plotting import plot_graph
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.graph.SHD import SHD
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion

N = 3000
SEED = 42
OUT = os.path.join("results", "metrics", "02_fci.json")
FIG_DIR = os.path.join("results", "figs")

# 端点对 -> 边符号（str(get_endpoint(a,b)) 是 a 端标记, 另一是 b 端标记）
_EP_SYM = {
    ("TAIL", "ARROW"): "->", ("ARROW", "TAIL"): "<-",
    ("TAIL", "TAIL"): "--", ("ARROW", "ARROW"): "<->",
    ("CIRCLE", "ARROW"): "o->", ("ARROW", "CIRCLE"): "<-o",
    ("CIRCLE", "TAIL"): "o-", ("TAIL", "CIRCLE"): "-o",
    ("CIRCLE", "CIRCLE"): "o-o",
}


def decode_edges(g):
    """图 -> ['X1 o-o X2', 'X3 -> X5', ...]。

    ⚠️ get_endpoint(n1, n2) 返回 n2 端端点（实测），故 n1 端用 get_endpoint(n2, n1)。
    边对按节点名升序规范化（小名字在前），保证真值 PAG 与估计 PAG 输出同一字符串。
    """
    nodes = g.get_nodes()
    out = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if g.is_adjacent_to(a, b):
                n1, n2 = (a, b) if a.get_name() < b.get_name() else (b, a)
                ep1 = str(g.get_endpoint(n2, n1)).split(".")[-1]   # n1 端标记
                ep2 = str(g.get_endpoint(n1, n2)).split(".")[-1]   # n2 端标记
                out.append(f"{n1.get_name()} {_EP_SYM.get((ep1, ep2), '?')} {n2.get_name()}")
    return sorted(out)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- 合成含隐变量数据（L 混淆 X1、X2）----
    rng = np.random.RandomState(SEED)
    L = rng.randn(N)
    X1 = 0.8 * L + 0.6 * rng.randn(N)
    X2 = 0.8 * L + 0.6 * rng.randn(N)
    X3 = 0.7 * X1 + rng.randn(N)
    X4 = 0.7 * X2 + rng.randn(N)
    X5 = 0.6 * X3 + 0.6 * X4 + rng.randn(N)
    obs = np.column_stack([X1, X2, X3, X4, X5])
    print(f"合成数据: 5 观测（X1..X5）+ 1 隐变量 L 混淆 X1、X2, n={N}, seed={SEED}\n")

    # ---- 真值 DAG（含 L）-> 真值 PAG（dag2pag 对齐）----
    nodes = [GraphNode(f"X{i + 1}") for i in range(5)]
    Ln = GraphNode("L")
    dag = Dag(nodes + [Ln])
    for i, j in [(0, 2), (1, 3), (2, 4), (3, 4)]:
        dag.add_directed_edge(nodes[i], nodes[j])
    dag.add_directed_edge(Ln, nodes[0])
    dag.add_directed_edge(Ln, nodes[1])
    truth_pag = dag2pag(dag, [Ln])

    # ---- FCI ----
    t0 = time.time()
    est_pag, edges = fci(obs, fisherz, 0.05, verbose=False, show_progress=False)
    fci_time = round(time.time() - t0, 3)

    # ---- PAG vs PAG 评估（SHD 计端点差异；Adjacency P/R 为主指标）----
    shd = SHD(truth_pag, est_pag).get_shd()
    ac = AdjacencyConfusion(truth_pag, est_pag)
    ar = ArrowConfusion(truth_pag, est_pag)
    adj_tp, adj_fp, adj_fn = ac.get_adj_tp(), ac.get_adj_fp(), ac.get_adj_fn()
    arr_tp, arr_fp, arr_fn = ar.get_arrows_tp(), ar.get_arrows_fp(), ar.get_arrows_fn()
    m = {
        "SHD(PAG, 端点口径)": int(shd),
        "adj_precision": _safe_ratio(adj_tp, adj_tp + adj_fp),
        "adj_recall": _safe_ratio(adj_tp, adj_tp + adj_fn),
        "arrow_precision": _safe_ratio(arr_tp, arr_tp + arr_fp),
        "arrow_recall": _safe_ratio(arr_tp, arr_tp + arr_fn),
        "time_s": fci_time,
        "truth_pag_edges": decode_edges(truth_pag),
        "est_pag_edges": decode_edges(est_pag),
    }

    print("真值 PAG（dag2pag 对齐）:")
    for e in m["truth_pag_edges"]:
        print(f"    {e}")
    print("FCI 估计 PAG:")
    for e in m["est_pag_edges"]:
        print(f"    {e}")
    print("\nPAG 评估（SHD=0 表示端点全一致; 圆圈是'不确定'不是错误）:")
    print(f"    SHD(PAG)={m['SHD(PAG, 端点口径)']}  Adj P/R={m['adj_precision']}/{m['adj_recall']}  "
          f"Arr P/R={m['arrow_precision']}/{m['arrow_recall']}  {fci_time}s")
    print("\nFCI 边属性（nl=无隐变量混淆 / pl=可能被混淆 / pd=可能非直接）:")
    for e in edges:
        props = [str(p).split(".")[-1] for p in e.properties]
        print(f"    {e.get_node1().get_name()} {e.get_endpoint1()} {e.get_endpoint2()} "
              f"{e.get_node2().get_name()}  props={props}")

    plot_graph(truth_pag, os.path.join(FIG_DIR, "02_fci_truth_pag.png"), title="Truth PAG (latent L)")
    plot_graph(est_pag, os.path.join(FIG_DIR, "02_fci_est_pag.png"), title="FCI estimated PAG")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"data": "5 observed + 1 latent (L->X1, L->X2, X1->X3, X2->X4, X3->X5, X4->X5)",
                   "n_samples": N, "seed": SEED, **m}, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标落盘: {OUT}")


if __name__ == "__main__":
    main()
