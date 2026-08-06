# -*- coding: utf-8 -*-
"""
knowledge/05-隐变量与排列方法.md 复现脚本（示例与文档逐字一致）+ 批次 C 扩展
===========================================================================
用法: cd causal-lab && python experiments/05_gin_rlcd/run.py

批次 C 扩展:
  示例 3: GIN（LiNLAM 合成数据：隐变量 L1->X1,X2, L2->X3,X4, L1->L2；n=500）
          输出因果图（含隐变量节点）+ 因果序（期望簇 [[0,1],[2,3]]）

产出:
  - results/metrics/05_gin_rlcd.json   全示例指标
  - results/figs/05_gin.png            GIN 因果图（橙色 = 隐变量 L1/L2）
"""
import sys, os, json, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

OUT = os.path.join("results", "metrics", "05_gin_rlcd.json")
FIG_DIR = os.path.join("results", "figs")
results = {}

print("=" * 70)
print("示例 1: GRaSP + BOSS（线性非高斯）")
print("=" * 70)
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from scripts.data_gen import simulate_linear_nongaussian
from scripts.evaluate import evaluate_graph

data, truth = simulate_linear_nongaussian(n=3000, seed=42, dist="exponential")
random.seed(42)  # 批次 B/C 纪律: 固定 GRaSP/BOSS 内部随机, 保证可复现
G1 = grasp(data, verbose=False)
m1 = evaluate_graph(truth, G1)
print(f"GRaSP: SHD={m1['SHD']}  adjP/R={m1['adj_precision']}/{m1['adj_recall']}")
assert m1["SHD"] <= 5, f"GRaSP SHD 应 <=5, 实际 {m1['SHD']}"
results["GRaSP"] = {"SHD": m1["SHD"], "adj_precision": m1["adj_precision"],
                    "adj_recall": m1["adj_recall"]}

G2 = boss(data, verbose=False)
m2 = evaluate_graph(truth, G2)
print(f"BOSS:  SHD={m2['SHD']}  adjP/R={m2['adj_precision']}/{m2['adj_recall']}")
results["BOSS"] = {"SHD": m2["SHD"], "adj_precision": m2["adj_precision"],
                   "adj_recall": m2["adj_recall"]}

print()
print("=" * 70)
print("示例 2: RLCD 隐变量发现（1 隐变量生成 5 观测）")
print("=" * 70)
from causallearn.search.HiddenCausal.RLCD import Chi2RankTest, RLCD
from causallearn.graph.NodeType import NodeType

rng = np.random.default_rng(1)
ss = 3000
latent = rng.normal(size=ss)
data_r = np.column_stack([
    1.0 * latent + 0.05 * rng.normal(size=ss),
    1.2 * latent + 0.05 * rng.normal(size=ss),
    1.4 * latent + 0.05 * rng.normal(size=ss),
    1.6 * latent + 0.05 * rng.normal(size=ss),
    1.8 * latent + 0.05 * rng.normal(size=ss),
])
data_r = (data_r - data_r.mean(axis=0)) / data_r.std(axis=0)
cg = RLCD(data_r, ranktest_method=Chi2RankTest(data_r), maxk=2)
latents = [n.get_name() for n in cg.G.get_nodes() if n.get_node_type() == NodeType.LATENT]
print(f"RLCD 检测隐变量: {latents}")
print(f"all_vars: {cg.all_vars}")
assert "L1" in latents, "应检测到隐变量 L1"
results["RLCD"] = {"detected_latents": latents}

print()
print("=" * 70)
print("示例 3: GIN（LiNLAM 隐变量+非高斯, n=500, 期望簇 [[0,1],[2,3]]）")
print("=" * 70)
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from scripts.plotting import plot_graph

# TestGIN case1 结构: L1 -> X1,X2; L2 -> X3,X4; L1 -> L2; 均匀非高斯噪声
rng = np.random.RandomState(42)
s = 500
L1 = rng.uniform(-1, 1, s)
L2 = rng.uniform(1.2, 1.8) * L1 + rng.uniform(-1, 1, s)
X1 = rng.uniform(1.2, 1.8) * L1 + 0.2 * rng.uniform(-1, 1, s)
X2 = rng.uniform(1.2, 1.8) * L1 + 0.2 * rng.uniform(-1, 1, s)
X3 = rng.uniform(1.2, 1.8) * L2 + 0.2 * rng.uniform(-1, 1, s)
X4 = rng.uniform(1.2, 1.8) * L2 + 0.2 * rng.uniform(-1, 1, s)
d_gin = np.column_stack([X1, X2, X3, X4])
d_gin = (d_gin - d_gin.mean(axis=0)) / d_gin.std(axis=0)

t0 = time.time()
G, K = GIN(d_gin, indep_test_method="kci", alpha=0.05)
gin_time = round(time.time() - t0, 3)
order_sorted = [sorted(cluster_i) for cluster_i in K]
print(f"GIN 耗时: {gin_time}s")
print(f"GIN 因果序（簇）: {order_sorted}   (期望 [[0,1],[2,3]])")
print(f"GIN 隐变量节点: {[n.get_name() for n in G.get_nodes() if n.get_node_type() == NodeType.LATENT]}")
assert order_sorted == [[0, 1], [2, 3]], "GIN 应恢复两个隐变量簇"
results["GIN"] = {"time_s": gin_time, "causal_order_clusters": order_sorted,
                  "latents": [n.get_name() for n in G.get_nodes()
                              if n.get_node_type() == NodeType.LATENT]}
print("GIN 因果图（橙色节点 = 隐变量）已落盘")

os.makedirs(FIG_DIR, exist_ok=True)
plot_graph(G, os.path.join(FIG_DIR, "05_gin.png"),
           title="GIN: latent L1->{X1,X2}, L2->{X3,X4} (orange = latent)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
print(f"指标落盘: {OUT}")

print()
print("全部复现成功 [OK]")
