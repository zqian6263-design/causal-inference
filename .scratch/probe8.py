import numpy as np, time
np.set_printoptions(precision=3, suppress=True)

def gen_linear(edges, n, seed=42, lo=0.5, hi=0.9, flips=0.5, noise=1.0):
    """sign-flip linear Gaussian generator (官方 TestGES 风格). edges: list of (i,j) meaning i->j"""
    d = max(max(e) for e in edges)+1
    A = np.zeros((d,d))
    for i,j in edges: A[i,j]=1
    A = A.T
    wm = np.random.default_rng(seed).uniform(lo,hi,(d,d))
    idx = np.random.default_rng(seed).choice(np.arange(d*d), size=int(d*d*flips), replace=False)
    wm[np.unravel_index(idx, wm.shape)] *= -1.
    A = A*wm
    mix = np.linalg.inv(np.eye(d)-A)
    E = np.random.default_rng(seed).normal(0,noise,(d,n))
    return (mix@E).T

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD

def make_truth(edges):
    d = max(max(e) for e in edges)+1
    nodes=[GraphNode(f'X{i+1}') for i in range(d)]
    g=GeneralGraph(nodes)
    for i,j in edges: g.add_directed_edge(nodes[i],nodes[j])
    return dag2cpdag(g)

cases = {
 '4node X0->X1->X2<-X3': [(0,1),(1,2),(3,2)],
 '4node benchmark+': [(0,1),(0,3),(1,2),(2,3)],
 '5node TestPC': [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)],
}
for name, edges in cases.items():
    for n in [3000, 10000]:
        X = gen_linear(edges, n)
        R = ges(X, score_func='local_score_BIC')   # 默认参数
        shd = SHD(make_truth(edges), R['G']).get_shd()
        print(f'{name} n={n}: SHD={shd}')
    X = gen_linear(edges, 10000)
    R = ges(X, score_func='local_score_BIC')
    print(f'   G graph:\n{R["G"].graph}')
