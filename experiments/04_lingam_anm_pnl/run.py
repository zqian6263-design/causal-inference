# -*- coding: utf-8 -*-
"""
knowledge/04-函数因果模型.md 复现脚本（示例与文档逐字一致）
用法: cd causal-lab && python experiments/04_lingam_anm_pnl/run.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

print("=" * 70)
print("示例 1: ICA-LiNGAM + DirectLiNGAM（线性非高斯, 期望恢复因果序）")
print("=" * 70)
from causallearn.search.FCMBased import lingam
from scripts.data_gen import simulate_linear_nongaussian

data, truth = simulate_linear_nongaussian(n=3000, seed=42, dist="exponential")
m1 = lingam.ICALiNGAM(random_state=42)
m1.fit(data)
print(f"ICA-LiNGAM causal_order:    {m1.causal_order_}")
print(f"ICA-LiNGAM 系数非零项:      {np.count_nonzero(np.abs(m1.adjacency_matrix_) > 0.1)}")
assert list(m1.causal_order_) == [0, 1, 2, 3, 4], "因果序应等于真值拓扑序"
assert np.count_nonzero(np.abs(m1.adjacency_matrix_) > 0.1) == 7, "应有 7 条边"

m2 = lingam.DirectLiNGAM(random_state=42)
m2.fit(data)
print(f"DirectLiNGAM causal_order:  {m2.causal_order_}")
print(f"DirectLiNGAM 系数非零项:    {np.count_nonzero(np.abs(m2.adjacency_matrix_) > 0.1)}")

print()
print("=" * 70)
print("示例 2: ANM 成对方向判断（x→y, y=0.5x+sin(x)+e）")
print("=" * 70)
from causallearn.search.FCMBased.ANM.ANM import ANM

rng = np.random.RandomState(0)
x = rng.randn(500)
y = 0.5 * x + np.sin(x) + np.random.RandomState(1).randn(500) * 0.3
anm = ANM()
p_fwd, p_bwd = anm.cause_or_effect(x.reshape(-1, 1), y.reshape(-1, 1))
print(f"ANM p_forward (x→y): {p_fwd:.6f}")
print(f"ANM p_backward(y→x): {p_bwd:.6e}")
assert p_fwd > 0.05 and p_bwd < 0.05, "应判定 x→y"

print()
print("全部复现成功 ✓")
