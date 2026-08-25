# -*- coding: utf-8 -*-
"""
experiments/demo_remote_sensing/markov_blanket.py — 遥感因果特征选择管道（F4 轮，IMP #11 基础版）
==================================================================================================
把「特征 + 标签联合变量 → 因果图 → 标签马尔可夫毯 → 因果特征子集」完整管道跑一遍，
作为论文方法章「因果特征选择」的实验模板（TGRS 方向）。

因果结构（真值，6 变量：特征 X1-X5 + 标签 Y）：
    X3(土壤湿度) ──→ X1(NDVI), X4, Y          # 混淆根源：同时影响特征与标签
    X1(NDVI) ──────→ Y                        # 因果父节点
    X4 ─────────────→ X5                      # 派生链（与 Y 无直接因果，但经 X3 共因相关）
    X2(波段噪声) ──── 无关                      # 纯噪声特征（与 Y 无任何关联）
  → Y 的真值马尔可夫毯 MB(Y) = {X1, X3}；X2 无关特征；X4/X5 是「相关但非因果」特征。

三类特征集对比（逻辑回归 5 折 CV 准确率，≥3 seed 报 mean±std）：
    ① 全特征 X1-X5
    ② 相关筛选   |corr(X_i, Y)| ≥ 阈值        （会被 X4/X5 的共因相关骗进去）
    ③ 因果马尔可夫毯 从 PC 估计 CPDAG 提取 MB(Y) （最小充分因果特征集）

预期演示：③ 与 ① 准确率相当或更高，但特征更少；② 混入非因果特征 X4/X5。

用法: cd causal-lab && python experiments/demo_remote_sensing/markov_blanket.py
产出:
  - results/metrics/markov_blanket.json    逐 seed 全指标 + 汇总
  - results/figs/markov_blanket_cpdag.png  PC 估计 CPDAG（橙/紫强调 = Y 的马尔可夫毯成员）
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from scripts.evaluate import evaluate_graph
from scripts.plotting import plot_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint

N = 3000
SEEDS = [42, 1, 7]
CORR_THRESHOLD = 0.3
Y_INDEX = 5                      # 变量顺序 X1..X5, Y
FEATURE_NAMES = ["X1", "X2", "X3", "X4", "X5", "Y"]
OUT = os.path.join("results", "metrics", "markov_blanket.json")
FIG = os.path.join("results", "figs", "markov_blanket_cpdag.png")


def generate_data(n, seed):
    """遥感特征-标签 合成数据：X1-X5 连续特征 + 二值标签 Y。

    图学习用连续「产量得分」Y_cont（保证 fisherz 高斯假设成立）；
    分类用二值标签 Y_bin（真实任务形态）。两者的因果父节点同为 {X1, X3}。
    """
    rng = np.random.RandomState(seed)
    X3 = rng.randn(n)                                   # 土壤湿度（混淆根源）
    X2 = rng.randn(n)                                   # 波段噪声（无关）
    X1 = 0.8 * X3 + 0.5 * rng.randn(n)                  # NDVI（受湿度影响）
    X4 = 0.7 * X3 + 0.5 * rng.randn(n)                  # 派生特征（受湿度影响，与 Y 仅共因相关）
    X5 = 0.7 * X4 + 0.5 * rng.randn(n)                  # 更深的派生态（X4→X5）
    score = 1.2 * X1 + 0.9 * X3 + 0.4 * rng.randn(n)    # 连续「产量得分」
    Y_cont = score.reshape(-1, 1)
    Y_bin = (score > 0).astype(int)                     # 二值标签
    # 图学习矩阵: X1,X2,X3,X4,X5,Y_cont（连续，fisherz 前提成立）
    GX = np.column_stack([X1, X2, X3, X4, X5, Y_cont[:, 0]])
    truth = np.zeros((6, 6), dtype=int)                 # 行/列顺序同 GX
    truth[2, 0] = 1   # X3 → X1
    truth[2, 3] = 1   # X3 → X4
    truth[2, 5] = 1   # X3 → Y
    truth[0, 5] = 1   # X1 → Y
    truth[3, 4] = 1   # X4 → X5
    return GX, truth, X2, X3, X4, X5, Y_bin, score


def graph_to_adj(G):
    """causallearn GeneralGraph -> 邻接矩阵（同 plotting._edge_list 的端点约定）。"""
    nodes = G.get_nodes()
    n = len(nodes)
    idx = {nm.get_name(): i for i, nm in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=int)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j:
                continue
            if not G.is_adjacent_to(a, b):
                continue
            ep_a = G.get_endpoint(b, a)   # a 端端点
            ep_b = G.get_endpoint(a, b)   # b 端端点
            if ep_b == Endpoint.ARROW:
                adj[i, j] = 1            # a → b
            elif ep_a == Endpoint.ARROW:
                adj[j, i] = 1            # b → a
            else:
                adj[i, j] = adj[j, i] = 1  # 无向（TAIL-TAIL / 含 CIRCLE）
    return adj


def markov_blanket(adj, y_idx):
    """从估计 CPDAG 邻接中提取 Y 的马尔可夫毯（父 + 子 + 配偶）。

    配偶启发式（CPDAG 上）：Z 是配偶 ≤ Z 经某 W 与 Y 相连，且 Y→W 与 Z→W
    两箭头均指向 W（对撞点）。CPDAG 可能含未定向边，此判定为启发式——
    本 demo Y 是无子的叶子对撞点，MB = 邻接集即为精确.
    """
    n = len(adj)
    und = np.logical_or(adj, adj.T)
    mb = {i for i in range(n) if i != y_idx and und[y_idx, i]}
    for z in range(n):
        if z == y_idx or z in mb:
            continue
        for w in range(n):
            if w == y_idx or not und[y_idx, w] or not und[z, w]:
                continue
            if adj[y_idx, w] and not adj[w, y_idx] and adj[z, w] and not adj[w, z]:
                mb.add(z)
                break
    return mb


def corr_features(X_feat, Y_bin, threshold):
    """|corr(feature, Y)| ≥ threshold 的启发式相关筛选。返回特征索引集。"""
    assert X_feat.shape[0] == Y_bin.shape[0]
    Ys = Y_bin - Y_bin.mean()
    corrs = [abs(np.corrcoef(X_feat[:, i], Ys)[0, 1]) for i in range(X_feat.shape[1])]
    return {i for i, c in enumerate(corrs) if c >= threshold}


def cv_accuracy(X, y, cv=5, seed=42):
    """逻辑回归 5 折 CV 准确率（sklearn，max_iter 给足）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    lr = LogisticRegression(max_iter=2000)
    return float(np.mean(cross_val_score(lr, X, y, cv=cv, scoring="accuracy", n_jobs=1)))


def main():
    print(f"遥感因果特征选择：N={N}，特征 X1-X5 + 标签 Y，seeds={SEEDS}\n" + "=" * 70)
    per_seed = {}
    fig_done = False

    for seed in SEEDS:
        GX, truth, _, _, _, _, Y_bin, _ = generate_data(N, seed)
        X_feat = GX[:, :5]                          # X1..X5（分类用特征）
        t0 = time.time()
        cg = pc(GX, 0.05, fisherz, show_progress=False)
        adj = graph_to_adj(cg.G)
        m = evaluate_graph(truth, cg)
        mb_idx = markov_blanket(adj, Y_INDEX)
        corr_set = corr_features(X_feat, Y_bin, CORR_THRESHOLD)
        all_set = set(range(5))

        def acc_of(idx_set):
            cols = sorted(idx_set)
            return cv_accuracy(X_feat[:, cols], Y_bin, seed=seed)

        acc = {"all": acc_of(all_set),
               "corr": acc_of(corr_set),
               "mb": acc_of(mb_idx)}
        elapsed = round(time.time() - t0, 3)
        rec = {
            "graph": {"SHD": m["SHD"], "adj_precision": m["adj_precision"],
                      "adj_recall": m["adj_recall"], "time_s": elapsed},
            "truth_mb": sorted({i for i, nm in enumerate(FEATURE_NAMES)
                                if nm in ("X1", "X3")}),
            "est_mb": sorted(mb_idx),
            "est_mb_name": [FEATURE_NAMES[i] for i in sorted(mb_idx)],
            "corr_set": sorted(corr_set),
            "corr_set_name": [FEATURE_NAMES[i] for i in sorted(corr_set)],
            "accuracy": acc,
        }
        per_seed[str(seed)] = rec
        print(f"[seed={seed}] 图 SHD={m['SHD']} adjP/R={m['adj_precision']}/{m['adj_recall']} "
              f"({elapsed}s)\n"
              f"   MB(Y)={rec['est_mb_name']}  corr={rec['corr_set_name']}\n"
              f"   CV acc: all={acc['all']:.3f} corr={acc['corr']:.3f} mb={acc['mb']:.3f}")

        if not fig_done:   # 用 seed=42 的估计图落一张 PNG（强调 MB 成员）
            plot_graph(cg.G, FIG, title="Remote-sensing CPDAG (PC+fisherz) - purple = MB(Y) = {X1, X3}",
                       figsize=(8, 6), highlight={FEATURE_NAMES[i] for i in sorted(mb_idx)})
            fig_done = True

    # ---- 汇总 mean±std ----
    acc_all = [per_seed[str(k)]["accuracy"]["all"] for k in SEEDS]
    acc_corr = [per_seed[str(k)]["accuracy"]["corr"] for k in SEEDS]
    acc_mb = [per_seed[str(k)]["accuracy"]["mb"] for k in SEEDS]
    summary = {
        "cv_accuracy_all": {"mean": round(float(np.mean(acc_all)), 4),
                            "std": round(float(np.std(acc_all)), 4)},
        "cv_accuracy_corr": {"mean": round(float(np.mean(acc_corr)), 4),
                             "std": round(float(np.std(acc_corr)), 4)},
        "cv_accuracy_mb": {"mean": round(float(np.mean(acc_mb)), 4),
                           "std": round(float(np.std(acc_mb)), 4)},
        "mb_matches_truth_all_seeds": all(
            per_seed[str(k)]["est_mb"] == per_seed[str(k)]["truth_mb"] for k in SEEDS),
    }
    print("\n汇总（3 seed mean±std）:")
    for k in summary:
        if k.startswith("cv_"):
            print(f"  {k:<20} {summary[k]['mean']:.3f}±{summary[k]['std']:.3f}")
    print(f"  MB 恒等于真值{u'是' if summary['mb_matches_truth_all_seeds'] else '否（不一致需检查）'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(os.path.join("results", "metrics", "markov_blanket.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_samples": N, "seeds": SEEDS, "corr_threshold": CORR_THRESHOLD,
                   "truth_mb_name": ["X1", "X3"], "structure": "X3->{X1,X4,Y}, X1->Y, X4->X5, X2无关",
                   "per_seed": per_seed, "summary": summary},
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False)\n图已落盘: {FIG}")


if __name__ == "__main__":
    main()