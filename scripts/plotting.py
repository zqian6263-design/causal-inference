# -*- coding: utf-8 -*-
"""
causal-lab 轻量图渲染（scripts/plotting.py）
=============================================
本机无 graphviz（dot 二进制未装，`to_pydot().write_png()` 会 FileNotFoundError），
本模块用 matplotlib + networkx 直接渲染 causal-learn 图对象，CPDAG/PAG 通用：
  - ARROW 端点 -> 箭头（指向该节点）
  - TAIL  端点 -> 直线（无附加标记）
  - CIRCLE 端点 -> 小圆圈（PAG 的「不确定」标记，如 `X o-o Y`、`X o-> Y`）

用法:
    from scripts.plotting import plot_graph
    plot_graph(cg.G, "results/figs/01_pc.png", title="PC+fisherz")

位置布局用 networkx.spring_layout（seed 固定，可复现）；仅依赖 numpy/matplotlib/networkx，
全部在 requirements.txt 内。
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # 无显示环境，只落盘文件
import matplotlib.pyplot as plt

from causallearn.graph.Endpoint import Endpoint

# 端点绘图配置
_EP_COLORS = {"default": "#2b6cb0", "latent": "#c05621"}
_NODE_R = 0.028      # 节点圆半径（相对布局坐标）
_INSET = 0.085       # 边端点向内收缩比例（给箭头/圆圈留位）


def _edge_list(g):
    """提取图的所有边：[(name_a, name_b, ep_a, ep_b)]。

    ⚠️ causal-learn 的 get_endpoint(n1, n2) 返回的是「n2 端」的端点（实测 X3→X5 时
    get_endpoint(X3,X5)=ARROW），故取 a 端标记用 get_endpoint(b, a)、b 端用 get_endpoint(a, b)。
    ep_a = 节点 a 处的端点，ep_b = 节点 b 处的端点。
    """
    nodes = g.get_nodes()
    edges = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j:
                continue
            if g.is_adjacent_to(a, b):
                edges.append((a.get_name(), b.get_name(),
                              g.get_endpoint(b, a), g.get_endpoint(a, b)))
    return edges


def _is_latent(g, name):
    from causallearn.graph.NodeType import NodeType
    for n in g.get_nodes():
        if n.get_name() == name:
            return n.get_node_type() == NodeType.LATENT
    return False


def _draw_glyph(ax, p, ux, uy, ep, color, scale):
    """在 p 处画端点标记；ux/uy 是「从该节点指向另一端」的单位方向。"""
    if ep == Endpoint.ARROW:
        # 箭头尖指向节点（沿 -u 方向回缩 tip 距离）
        tip = (p[0] + ux * scale * 0.028, p[1] + uy * scale * 0.028)
        back = (p[0] - ux * scale * 0.030, p[1] - uy * scale * 0.030)
        perp = np.array([-uy, ux]) * scale * 0.016
        ax.fill([tip[0], back[0] + perp[0], back[0] - perp[0]],
                [tip[1], back[1] + perp[1], back[1] - perp[1]],
                color=color, zorder=3)
    elif ep == Endpoint.CIRCLE:
        c = (p[0] + ux * scale * 0.016, p[1] + uy * scale * 0.016)
        ax.add_patch(plt.Circle(c, scale * 0.014, fill=False, edgecolor=color,
                                lw=1.4, zorder=3))


def _plot_edges(ax, edges, pos, color_map):
    for a, b, ep_a, ep_b in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        p1 = (x1 + ux * _INSET * L, y1 + uy * _INSET * L)   # a 端内侧点
        p2 = (x2 - ux * _INSET * L, y2 - uy * _INSET * L)   # b 端内侧点
        color = color_map.get(a, _EP_COLORS["default"])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.8, zorder=1)
        _draw_glyph(ax, p1, -ux, -uy, ep_a, color, L)       # a 端：朝外方向 = -u
        _draw_glyph(ax, p2, ux, uy, ep_b, color, L)         # b 端：朝外方向 = u


def _plot_nodes(ax, pos, labels, color_map, highlight=None):
    highlight = set(highlight or [])
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    for nm in labels:
        if nm in color_map:
            continue
        if nm in highlight:
            color_map[nm] = "#ddd6fe"      # 强调色：马尔可夫毯成员（淡紫）
    edgec = ["#d946ef" if nm in highlight else "#334155" for nm in labels]
    lw    = [2.2 if nm in highlight else 1.4 for nm in labels]
    ax.scatter(xs, ys, s=2200, c=[color_map.get(nm, "#dbeafe") for nm in labels],
               edgecolors=edgec, linewidths=lw, zorder=2)
    for nm in labels:
        ax.text(pos[nm][0], pos[nm][1], nm, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=4)


def plot_graph(g, save_path, title="", figsize=(7, 5.5), highlight=None):
    """渲染 causal-learn 图（GeneralGraph/CausalGraph，CPDAG/PAG 通用）到 PNG。

    highlight: 可选的节点名集合（set），这些节点以强调色（橙色描边）绘制——
    如 markov_blanket.py 用其标出标签 Y 的马尔可夫毯成员。
    """
    labels = [n.get_name() for n in g.get_nodes()]
    edges = _edge_list(g)

    Gx = nx.Graph()
    for nm in labels:
        Gx.add_node(nm)
    for a, b, *_ in edges:
        Gx.add_edge(a, b)
    pos = nx.spring_layout(Gx, seed=42, k=0.9) if Gx.number_of_nodes() > 0 else {}

    color_map = {}
    highlight = set(highlight or [])
    for nm in labels:
        if _is_latent(g, nm):
            color_map[nm] = _EP_COLORS["latent"]

    fig, ax = plt.subplots(figsize=figsize)
    _plot_edges(ax, edges, pos, color_map)
    _plot_nodes(ax, pos, labels, color_map, highlight=highlight)
    if title:
        ax.set_title(title, fontsize=13)
    ax.axis("off")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # 自测：渲染一个 PC 的 CPDAG 与一个带隐变量的 PAG
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import scripts.patch_d_separation  # dag2pag 依赖 nx.is_d_separator（networkx 3.2.1 补丁）
    from scripts.data_gen import simulate_linear_gaussian
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    from causallearn.utils.DAG2PAG import dag2pag
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode

    data, truth = simulate_linear_gaussian(n=2000, seed=42)
    cg = pc(data, 0.05, fisherz, show_progress=False)
    plot_graph(cg.G, "_plot_selfcheck_cpdag.png", title="CPDAG self-check")
    print("CPDAG render OK")

    nodes = [GraphNode(f"X{i+1}") for i in range(5)]
    L = GraphNode("L")
    dag = Dag(nodes + [L])
    for i, j in [(0, 2), (1, 3), (2, 4), (3, 4)]:
        dag.add_directed_edge(nodes[i], nodes[j])
    dag.add_directed_edge(L, nodes[0])
    dag.add_directed_edge(L, nodes[1])
    pag = dag2pag(dag, [L])
    plot_graph(pag, "_plot_selfcheck_pag.png", title="PAG self-check (latent L)")
    print("PAG render OK")
    os.remove("_plot_selfcheck_cpdag.png")
    os.remove("_plot_selfcheck_pag.png")
