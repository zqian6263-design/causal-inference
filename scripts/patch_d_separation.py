# -*- coding: utf-8 -*-
"""
networkx 3.2.1 兼容补丁（scripts/patch_d_separation.py）
=========================================================
causal-learn 的 d_separation 独立性检验调用 nx.is_d_separator（networkx ≥ 3.3 才有）。
本机 pytorch env 是 Python 3.9.25，networkx 3.3+ 要求 Python ≥ 3.10，无法升级。
但 networkx 3.2.1 自带的 d_separated(G, x, y, z) 与 is_d_separator 功能完全等价，
这里在运行时补上等价实现——官方 d_separation 检验即可正常使用。

用法（在任何使用 d_separation 的脚本最顶部）:
    import sys; sys.path.insert(0, <causal-lab 根目录>)
    import scripts.patch_d_separation   # 只需导入一次

验证:
    from causallearn.utils.cit import d_separation
    cg = pc(data, 0.05, d_separation, true_dag=true_dag_netx)   # 官方测试写法
"""
import networkx as nx
from networkx.algorithms.d_separation import d_separated

if not hasattr(nx, "is_d_separator"):
    def is_d_separator(G, x, y, z):
        """等价的 is_d_separator: 判断 x 与 y 在条件集 z 下是否 d-分离。"""
        return bool(d_separated(G, set(x), set(y), set(z)))
    nx.is_d_separator = is_d_separator
    print("[patch_d_separation] nx.is_d_separator 已由 d_separated 补齐 (networkx %s)" % nx.__version__)
