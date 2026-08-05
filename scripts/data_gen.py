# -*- coding: utf-8 -*-
"""
causal-lab 统一数据生成器（scripts/data_gen.py）
=================================================
供所有实验脚本复用：4 种数据类型 x 任意 DAG 结构，seed 固定可复现。

用法示例:
    from scripts.data_gen import simulate_linear_gaussian, DEFAULT_DAG
    data, truth_adj = simulate_linear_gaussian(n=2000, dag_edges=DEFAULT_DAG, seed=42)

约定:
    - truth_adj: numpy 邻接矩阵, truth_adj[i,j]=1 表示 i -> j
    - 数据按拓扑序生成（父节点先于子节点）
"""
import numpy as np

# TestPC.py 官方基准图: 0->1->2->4, 0->3, 1->3, 2->3, 3->4（5 节点 7 边）
DEFAULT_DAG = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}

# 10 节点稀疏链式 DAG（扩展性测试用）
SPARSE10_DAG = {(i, i + 1) for i in range(9)} | {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}


def _dag_to_adj(dag_edges, n):
    adj = np.zeros((n, n), dtype=int)
    for i, j in dag_edges:
        adj[i, j] = 1
    return adj


def _topo_order(dag_edges, n):
    """返回拓扑序（父节点在前）。"""
    indeg = np.zeros(n, dtype=int)
    children = [[] for _ in range(n)]
    for i, j in dag_edges:
        indeg[j] += 1
        children[i].append(j)
    order, q = [], [i for i in range(n) if indeg[i] == 0]
    while q:
        u = q.pop(0)
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    assert len(order) == n, "DAG 存在环!"
    return order


def _parents(i, dag_edges):
    return [p for p, c in dag_edges if c == i]


def _num_nodes(dag_edges):
    return max([max(e) for e in dag_edges] + [-1]) + 1


def _generate(dag_edges, n_samples, noise_fn, link_fn, seed):
    rng = np.random.RandomState(seed)
    num_nodes = _num_nodes(dag_edges)
    adj = _dag_to_adj(dag_edges, num_nodes)
    X = np.zeros((n_samples, num_nodes))
    for node in _topo_order(dag_edges, num_nodes):
        parents = _parents(node, dag_edges)
        val = np.zeros(n_samples)
        for p in parents:
            val += link_fn(p, node) * X[:, p]
        X[:, node] = val + noise_fn(rng, n_samples)
    return X, adj


def simulate_linear_gaussian(n=2000, dag_edges=DEFAULT_DAG, seed=42, noise_scale=1.0):
    """线性高斯（对齐官方 utils_simulate_data.py）: X = (I-A)^-1 E, E~N(0,1).
    权重 uniform(0.5,0.9) 含 50% 负号 —— 与官方 TestPC 基准完全一致."""
    return _simulate_official(n, dag_edges, seed, noise_type="gaussian", noise_scale=noise_scale)


def simulate_linear_nongaussian(n=2000, dag_edges=DEFAULT_DAG, seed=42, dist="exponential"):
    """线性非高斯（对齐官方）: X = (I-A)^-1 E, E~Exp(1)-1 或 U(-1.5,1.5). LiNGAM 前提."""
    return _simulate_official(n, dag_edges, seed, noise_type=dist, noise_scale=1.0)


def _simulate_official(n, dag_edges, seed, noise_type, noise_scale):
    rng = np.random.RandomState(seed)
    num_nodes = _num_nodes(dag_edges)
    adj = _dag_to_adj(dag_edges, num_nodes).astype(float)
    # 权重 uniform(0.5,0.9), 50% 取负（官方 linear_weight_netative_prob=0.5）
    weights = rng.uniform(0.5, 0.9, (num_nodes, num_nodes))
    mask = rng.choice([1.0, -1.0], size=(num_nodes, num_nodes), p=[0.5, 0.5])
    # ⚠️ 关键: 官方用 A.T（X=(I-A^T)^-1 E），否则因果方向倒置
    A = adj.T * weights * mask
    mixing = np.linalg.inv(np.eye(num_nodes) - A)
    if noise_type == "gaussian":
        E = rng.randn(num_nodes, n) * noise_scale
    elif noise_type == "exponential":
        E = rng.exponential(1.0, (num_nodes, n)) - 1.0
    else:
        E = rng.uniform(-1.5, 1.5, (num_nodes, n))
    return (mixing @ E).T, adj.astype(int)


def simulate_nonlinear_anm(n=2000, dag_edges=DEFAULT_DAG, seed=42):
    """非线性 ANM: X_j = sum(f(X_pa)) + 高斯噪声（可逆加性噪声）. ANM/PNL 适用."""
    def noise(rng, n_samples):
        return rng.randn(n_samples) * 0.3
    def link(p, c):
        return None  # 不用系数
    rng = np.random.RandomState(seed)
    num_nodes = _num_nodes(dag_edges)
    adj = _dag_to_adj(dag_edges, num_nodes)
    X = np.zeros((n, num_nodes))
    for node in _topo_order(dag_edges, num_nodes):
        parents = _parents(node, dag_edges)
        val = np.zeros(n)
        for k, p in enumerate(parents):
            f = [np.sin, np.cos, lambda x: x ** 2][k % 3]
            val += 0.7 * f(X[:, p])
        X[:, node] = val + rng.randn(n) * 0.3
    return X, adj


def simulate_discrete(n=2000, dag_edges=DEFAULT_DAG, seed=42, states=3):
    """离散多值变量: 连续潜变量阈值分箱. chisq/gsq/BDeu 适用."""
    X, adj = simulate_linear_gaussian(n=n, dag_edges=dag_edges, seed=seed)
    Xd = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        qs = np.quantile(X[:, j], np.linspace(0, 1, states + 1)[1:-1])
        Xd[:, j] = np.digitize(X[:, j], qs)
    return Xd, adj


if __name__ == "__main__":
    # 自测: 每种生成器跑一遍, 校验形状与 seed 可复现
    for name, fn in [
        ("linear_gaussian", simulate_linear_gaussian),
        ("linear_nongaussian", simulate_linear_nongaussian),
        ("nonlinear_anm", simulate_nonlinear_anm),
        ("discrete", simulate_discrete),
    ]:
        d1, a1 = fn(n=1000, seed=42)
        d2, _ = fn(n=1000, seed=42)
        assert d1.shape == (1000, 5) and a1.shape == (5, 5)
        assert np.allclose(d1, d2), f"{name} 不可复现!"
        print(f"{name}: OK  shape={d1.shape}  edges={a1.sum()//2 + np.triu(a1).sum()}")
    print("data_gen 自测全部通过 ✓")
