# -*- coding: utf-8 -*-
"""
experiments/11_doc_coverage/run.py — readthedocs 文档缺口补齐（I 轮）
====================================================================
对照官方 causal-learn readthedocs 45 页，把此前未覆盖的页面补齐为可运行实验：

  ① gsq.rst            G^2 检验（0.1.4.8 中 chisq/gsq 均为字符串标识符，pc() 内部按字符串分派）
                       —— PC+gsq vs PC+chisq 在真实离散 CPD 上对比（3 seed）
  ② gcv.rst + gml.rst  广义打分 local_score_CV_general / local_score_marginal_general（RKHS）
                       —— GES 在非线性 ANM 上运行并评估（0.1.4.8 可用但极慢, n=500 控制耗时）
  ③ PDAG2DAG.rst + TXT2GeneralGraph.rst  图操作工具页
                       —— pdag2dag（PC 的 CPDAG → 无环一致扩展）+ txt2generalgraph（载入 bnlearn 真值图）
  ④ Datasets.rst       load_dataset 真实数据集演示（sachs/boston_housing/airfoil，仅结构演示不做定量评估）

产出: results/metrics/doc_coverage.json + 各段打印。
注: ② 的 RKHS 打分运行 ~1-2 分钟；④ 需联网下载（无网则跳过并在 JSON 标注）。
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ScoreUtils RKHS 无痛警告
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.data_gen import (simulate_discrete_cpd, simulate_nonlinear_anm,
                              simulate_linear_gaussian, DEFAULT_DAG)
from scripts.evaluate import evaluate_graph

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, gsq, fisherz
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from causallearn.graph.Endpoint import Endpoint

SCORE_N = 500                 # RKHS 核打分 O(n^2)，n=500 平衡耗时与稳定性
SEEDS = [42, 1, 7]
OUT = os.path.join("results", "metrics", "doc_coverage.json")
# bnlearn 真值图路径（txt2generalgraph 载入演示）
BNLEARN_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "causal-learn", "tests", "TestData", "bnlearn_discrete_10000"))


def _mean(xs):
    return round(float(np.mean(xs)), 4)


def _std(xs):
    return round(float(np.std(xs)), 4) if len(xs) > 1 else 0.0


def graph_to_adj_arrow(G):
    """causallearn 图 -> 邻接矩阵（ARROW 出 = i→j）。"""
    nodes = G.get_nodes()
    idx = {n.get_name(): i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a in nodes:
        for b in nodes:
            if a.get_name() == b.get_name():
                continue
            if G.is_adjacent_to(a, b) and G.get_endpoint(a, b) == Endpoint.ARROW:
                adj[idx[a.get_name()], idx[b.get_name()]] = 1
    return adj


def skeleton_count(G):
    return sum(1 for i, a in enumerate(G.get_nodes()) for j, b in enumerate(G.get_nodes())
               if i < j and G.is_adjacent_to(a, b))


# ============ ① gsq（G^2 检验）============
def demo_gsq():
    print("\n① gsq.rst：G^2 检验（PC+gsq vs PC+chisq，真实离散 CPD，5 节点 7 边）")
    res = {"n_nodes": 5, "n_edges": len(DEFAULT_DAG), "seeds": SEEDS}
    for cit_name, cit in [("chisq", chisq), ("gsq", gsq)]:
        shds = []
        for seed in SEEDS:
            data, truth = simulate_discrete_cpd(n=2000, seed=seed)
            cg = pc(data, 0.05, cit, show_progress=False)
            shds.append(evaluate_graph(truth, cg)["SHD"])
        res[cit_name] = {"SHD_mean": _mean(shds), "SHD_std": _std(shds),
                         "SHD_per_seed": shds}
        print(f"    PC+{cit_name:<5} SHD={_mean(shds)}±{_std(shds)}  per_seed={shds}")
    return res


# ============ ② 广义打分（CV_general / marginal_general）============
def demo_general_scores():
    print(f"\n② gcv.rst+gml.rst：广义非线性打分（GES，非线性 ANM，n={SCORE_N}）")
    res = {"n_samples": SCORE_N}
    data, truth = simulate_nonlinear_anm(n=SCORE_N, seed=42)
    for sf, label in [("local_score_CV_general", "CV_general"),
                      ("local_score_marginal_general", "marginal_general"),
                      ("local_score_BIC", "BIC(线性基线)")]:
        t0 = time.time()
        try:
            G = ges(data, score_func=sf)["G"]
            m = evaluate_graph(truth, G)
            m["time_s"] = round(time.time() - t0, 3)
            res[label] = {"SHD": m["SHD"], "adjP": m["adj_precision"],
                          "adjR": m["adj_recall"], "time_s": m["time_s"]}
            print(f"    GES+{label:<14} SHD={m['SHD']:<3} "
                  f"adjP/R={m['adj_precision']}/{m['adj_recall']}  {m['time_s']}s")
        except Exception as e:
            res[label] = {"error": str(e)[:100]}
            print(f"    GES+{label} 失败: {str(e)[:80]}")
    return res


# ============ ③ 图操作工具 ============
def demo_graph_ops():
    print("\n③ PDAG2DAG.rst + TXT2GeneralGraph.rst：图操作工具")
    res = {}

    # PDAG2DAG: PC 在 5 节点线性高斯上得 CPDAG（含无向边）→ pdag2dag 转 DAG
    import networkx as nx
    data, truth = simulate_linear_gaussian(n=3000, seed=42)
    cg = pc(data, 0.05, fisherz, show_progress=False)
    dig = pdag2dag(cg.G)
    # 验证 (a) 骨架边数不变 (b) 输出为无环有向图
    Gx = nx.DiGraph()
    nodes = dig.get_nodes()
    for a in nodes:
        for b in nodes:
            if a.get_name() == b.get_name():
                continue
            if dig.is_adjacent_to(a, b) and dig.get_endpoint(a, b) == Endpoint.ARROW:
                Gx.add_edge(a.get_name(), b.get_name())
    sk_before, sk_after = skeleton_count(cg.G), skeleton_count(dig)
    acyclic = bool(nx.is_directed_acyclic_graph(Gx))
    res["pdag2dag"] = {"skeleton_before": sk_before, "skeleton_after": sk_after,
                       "acyclic": acyclic}
    print(f"    PDAG2DAG: 骨架边 {sk_before}→{sk_after}，DAG 无环={acyclic}")

    # TXT2GeneralGraph: 载入 bnlearn asia 真值图，与手写 parse 逐位对照
    def _parse_manual(p, n):
        a = np.zeros((n, n), dtype=int)
        for line in open(p, encoding="utf-8"):
            if "-->" in line:
                sp = line.split("-->")
                a[int(sp[0].strip().split()[-1][1:]) - 1,
                  int(sp[1].strip().split()[0][1:]) - 1] = 1
        return a

    if os.path.isdir(BNLEARN_DIR):
        truth_path = os.path.join(BNLEARN_DIR, "truth_dag_graph", "asia.graph.txt")
        try:
            g = txt2generalgraph(truth_path)
            g_adj = graph_to_adj_arrow(g)
            manual = _parse_manual(truth_path, 8)
            match = bool(np.array_equal(g_adj, manual))
            res["txt2generalgraph"] = {"nodes": len(g.get_nodes()), "match_manual_parse": match}
            print(f"    TXT2GeneralGraph: 载入 asia 真值图 {len(g.get_nodes())} 节点，"
                  f"与手写 parse 逐位一致={match}")
        except Exception as e:
            res["txt2generalgraph"] = {"error": str(e)[:100]}
            print(f"    TXT2GeneralGraph 失败: {str(e)[:80]}")
    else:
        res["txt2generalgraph"] = {"note": "bnlearn 数据目录缺失（外部数据），跳过"}
        print("    TXT2GeneralGraph: bnlearn 数据目录缺失（外部数据），跳过")
    return res


# ============ ④ Datasets.load_dataset ============
def demo_datasets():
    print("\n④ Datasets.rst：load_dataset 真实数据集演示（仅结构，不做定量评估）")
    # Windows Anaconda py3.9 的 urllib 默认 SSL 上下文加载系统证书失败（ASN1 NOT_ENOUGH_DATA），
    # 与 github_push.py 同因——用 certifi 的 CA 构建 HTTPS 上下文并 install_opener（实测可行）。
    import ssl, urllib.request, certifi
    _ctx = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx)))
    from causallearn.utils.Dataset import load_dataset
    res = {}
    for name in ["sachs", "boston_housing", "airfoil"]:
        try:
            t0 = time.time()
            data, labels = load_dataset(name)
            cg = pc(data, 0.05, fisherz, show_progress=False)
            res[name] = {"shape": list(data.shape), "labels": list(labels),
                         "pc_edges": cg.G.get_num_edges(),
                         "time_s": round(time.time() - t0, 3)}
            print(f"    {name:<14} {data.shape[0]}×{data.shape[1]} 标签={labels[:3]}… "
                  f"PC 边数={cg.G.get_num_edges()} ({time.time()-t0:.1f}s)")
        except Exception as e:
            res[name] = {"error": str(e)[:100]}
            print(f"    {name} 失败: {str(e)[:80]}（可能无网）")
    return res


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None  # 可选: --only gsq|scores|graphops|datasets
    # --only 模式下合并已有结果，避免覆盖其他段
    out = {"doc": "补齐 readthedocs 缺口: gsq / CV_general+marginal / PDAG2DAG+TXT2GeneralGraph / load_dataset"}
    if only and os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            out.update(json.load(f))
    if only in (None, "gsq"):
        out["gsq"] = demo_gsq()
    if only in (None, "scores"):
        out["general_scores"] = demo_general_scores()
    if only in (None, "graphops"):
        out["graph_ops"] = demo_graph_ops()
    if only in (None, "datasets"):
        out["datasets"] = demo_datasets()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False)")


if __name__ == "__main__":
    main()