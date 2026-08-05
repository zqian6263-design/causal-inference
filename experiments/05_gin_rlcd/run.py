# -*- coding: utf-8 -*-
"""
knowledge/05-隐变量与排列方法.md 复现脚本（示例与文档逐字一致）
用法: cd D:/win/causal-lab && PYTHONPATH= D:/Anaconda/envs/pytorch/python.exe experiments/05_gin_rlcd/run.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

print("=" * 70)
print("示例 1: GRaSP + BOSS（线性非高斯）")
print("=" * 70)
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from scripts.data_gen import simulate_linear_nongaussian
from scripts.evaluate import evaluate_graph

data, truth = simulate_linear_nongaussian(n=3000, seed=42, dist="exponential")
G1 = grasp(data)
m1 = evaluate_graph(truth, G1)
print(f"GRaSP: {m1}")
# 注: GRaSP 内部有随机性(无 seed 参数), 多次运行 SHD 在 0~5 波动, 断言放宽
assert m1["SHD"] <= 5, f"GRaSP SHD 应 <=5, 实际 {m1['SHD']}"

G2 = boss(data)
m2 = evaluate_graph(truth, G2)
print(f"BOSS:  {m2}")

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

print()
print("全部复现成功 ✓")
