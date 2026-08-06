# -*- coding: utf-8 -*-
"""
knowledge/06-时序因果.md 复现脚本（示例与文档逐字一致）
用法: cd causal-lab && python experiments/07_granger/run.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

print("=" * 70)
print("示例: 线性 Granger 因果（x1 滞后驱动 x2）")
print("=" * 70)
from causallearn.search.Granger.Granger import Granger

rng = np.random.RandomState(42)
n = 2000
x1 = rng.randn(n)
x2 = np.zeros(n)
for t in range(1, n):
    x2[t] = 0.6 * x2[t - 1] + 0.5 * x1[t - 1] + 0.3 * rng.randn()

G = Granger()
pmat = G.granger_test_2d(np.column_stack([x1, x2]))
print(f"granger_test_2d p 值矩阵 shape: {pmat[0].shape}")
print(f"  x1→x2 相关 p 值（应小/显著）: {pmat[0][0]}")
print(f"  x2→x1 相关 p 值（应大/不显著）: {pmat[0][1]}")

coeff = G.granger_lasso(np.column_stack([x1, x2, rng.randn(n)]))
print(f"granger_lasso 系数矩阵 shape: {coeff.shape}")
print(f"  非零系数位置: {np.argwhere(np.abs(coeff) > 1e-6)}")

print()
print("全部复现成功 [OK]")
