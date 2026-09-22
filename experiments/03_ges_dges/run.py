# -*- coding: utf-8 -*-
"""
knowledge/03-打分型方法.md 复现脚本（示例与文档逐字一致）
用法: cd causal-lab && python experiments/03_ges_dges/run.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

print("=" * 70)
print("示例 1: GES + BIC 线性高斯（seed=42 单次恰好完美；多 seed 平均 5.40±3.07，非通用表现）")
print("=" * 70)
from causallearn.search.ScoreBased.GES import ges
from scripts.data_gen import simulate_linear_gaussian
from scripts.evaluate import evaluate_graph

data, truth = simulate_linear_gaussian(n=30000, seed=42)
Record = ges(data)
m = evaluate_graph(truth, Record["G"], verbose=True)
print(f"Record keys: {list(Record.keys())}")
# 这里断言 SHD==0 只是校验「该 seed 的复现结果未漂移」，不代表 GES 的通常表现
assert m["SHD"] == 0, f"seed=42 的复现结果漂移了, 实际 SHD={m['SHD']}（多 seed 平均 5.40±3.07）"

print()
print("=" * 70)
print("示例 2: DGES 确定性关系（X2 = 2*X0 - 1.5*X1, 无噪声）")
print("=" * 70)
from causallearn.search.ScoreBased.DGES import dges

rng = np.random.RandomState(42)
n = 3000
x0 = rng.randn(n); x1 = rng.randn(n)
x2 = 2.0 * x0 - 1.5 * x1
Xd = np.column_stack([x0, x1, x2])
rec_d = dges(Xd)
print(f"DGES keys: {list(rec_d.keys())}")
print(f"mindc_sets: {rec_d['mindc_sets']}")
print(f"det_clusters: {[c.tolist() for c in rec_d['det_clusters']]}")
assert frozenset({0, 1, 2}) in rec_d["mindc_sets"], "MinDC 检测失败"

print()
print("=" * 70)
print("示例 3: ExactSearch (astar, 5 节点)")
print("=" * 70)
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

dag, stats = bic_exact_search(data, None, "astar", verbose=False)
print(f"dag shape: {dag.shape}, 边数: {dag.sum()}")
print(f"stats keys: {list(stats.keys())[:6]}")
print(f"搜索统计: {stats}")

print()
print("全部复现成功 [OK]")
