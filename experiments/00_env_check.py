# -*- coding: utf-8 -*-
"""
Phase 1 知识库复现脚本 —— 与 knowledge/00-02 各文档中的代码示例逐字一致。
用法: cd causal-lab && python experiments/00_env_check.py
环境: causal-learn 0.1.4.8 / networkx 3.2.1（版本见 requirements.txt）
"""
import numpy as np

print("=" * 70)
print("文档 00：d-分离演示（networkx d_separated）")
print("=" * 70)
import networkx as nx
from networkx.algorithms.d_separation import d_separated

D = nx.DiGraph()
D.add_edges_from([("A", "C"), ("B", "C"), ("C", "D")])  # A->C<-B 且 C->D
print(d_separated(D, {"A"}, {"B"}, set()))   # 对撞: 无条件独立       -> True
print(d_separated(D, {"A"}, {"B"}, {"C"}))   # 条件在 C 上反而打通     -> False
print(d_separated(D, {"A"}, {"D"}, set()))   # 链 A->C->D: 相关        -> False
print(d_separated(D, {"A"}, {"D"}, {"C"}))   # 条件在 C 上阻断         -> True

print()
print("=" * 70)
print("文档 00：等价类演示（为什么 PC 只能学到 CPDAG）")
print("=" * 70)
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

np.random.seed(42); n = 3000

def gen_from(edges, coef=0.8):          # 按拓扑序采样一个 DAG
    B = np.zeros((3, 3))
    for i, j in edges: B[i, j] = coef
    G = nx.DiGraph(); G.add_nodes_from(range(3)); G.add_edges_from(edges)
    X = np.random.randn(n, 3)
    for j in nx.topological_sort(G):
        for i in range(3):
            if B[i, j] != 0: X[:, j] += B[i, j] * X[:, i]
    return X

def show(tag, Dmat):
    cg = pc(Dmat, 0.05, fisherz, show_progress=False)
    print(tag, '有向边:', sorted(cg.find_fully_directed()),
          '无向边:', sorted(cg.find_undirected()))

show('A->B->C ', gen_from([(0, 1), (1, 2)]))
show('A<-B<-C ', gen_from([(2, 1), (1, 0)]))
show('A<-B->C ', gen_from([(1, 0), (1, 2)]))
show('A->B<-C ', gen_from([(0, 1), (2, 1)]))   # 对撞, 可定向

print()
print("=" * 70)
print("文档 02：PC + fisherz（5 节点线性高斯，seed=42）")
print("=" * 70)
np.random.seed(42)
n = 2000
B = np.zeros((5, 5))
for i, j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i, j] = 0.7
X = np.random.randn(n, 5)
for j in range(5):
    for i in range(5):
        if B[i, j]: X[:, j] += B[i, j] * X[:, i]

cg = pc(X, 0.05, fisherz, show_progress=False)
print('耗时 %.3fs' % cg.PC_elapsed)
print('有向边:', sorted(cg.find_fully_directed()))
print('无向边:', sorted(cg.find_undirected()))
print(cg.G.graph)

print()
print("=" * 70)
print("文档 02：FCI + fisherz（含隐变量 L 混淆 X1、X2）")
print("=" * 70)
from causallearn.search.ConstraintBased.FCI import fci
np.random.seed(42)
n = 2000
L  = np.random.randn(n)                      # 未观测的共同原因
X1 = 0.8 * L + 0.6 * np.random.randn(n)
X2 = 0.8 * L + 0.6 * np.random.randn(n)      # X1、X2 被 L 混淆
X3 = 0.7 * X1 + np.random.randn(n)
X4 = 0.7 * X2 + np.random.randn(n)
X5 = 0.6 * X3 + 0.6 * X4 + np.random.randn(n)
obs = np.column_stack([X1, X2, X3, X4, X5])

G, edges = fci(obs, fisherz, 0.05, verbose=False, show_progress=False)
print(G.graph)
for e in edges:
    props = [str(p).split('.')[-1] for p in e.properties]
    print(f'{e.get_node1().get_name()} {e.get_endpoint1()} {e.get_endpoint2()} '
          f'{e.get_node2().get_name()}  props={props}')

print()
print("=" * 70)
print("文档 02：CD-NOD（机制分段切换，kci）")
print("=" * 70)
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
np.random.seed(42)
T = 600
x1 = np.random.randn(T)
x2 = np.concatenate([0.9*x1[:T//2] + 0.5*np.random.randn(T//2),   # 前段: 强因果
                     0.1*x1[T//2:] + 0.5*np.random.randn(T-T//2)]) # 后段: 弱因果
data = np.column_stack([x1, x2])
c_indx = np.arange(T).reshape(-1, 1)     # 时间索引

cg = cdnod(data, c_indx, 0.05, kci, True, 0, -1, show_progress=False)
print(cg.G.graph)   # 注意：最后一列 = c_indx 节点

print()
print("=" * 70)
print("文档 02：MVPC 缺失值（20% NaN）")
print("=" * 70)
from causallearn.utils.cit import mv_fisherz
X_missing = X.copy()
X_missing[np.random.rand(*X.shape) < 0.2] = np.nan
cg = pc(X_missing, 0.05, mv_fisherz, True, 0, 4, mvpc=True, show_progress=False)
print(cg.G.graph)

print()
print("全部复现成功 [OK]")
