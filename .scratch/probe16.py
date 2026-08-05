import numpy as np
np.set_printoptions(precision=3, suppress=True)
def gen_signflip(edges, n, seed=42, lo=0.5, hi=0.9):
    d = max(max(e) for e in edges)+1
    A = np.zeros((d,d))
    for i,j in edges: A[i,j]=1
    A = A.T
    wm = np.random.default_rng(seed).uniform(lo,hi,(d,d))
    idx = np.random.default_rng(seed).choice(np.arange(d*d), size=int(d*d*0.5), replace=False)
    wm[np.unravel_index(idx, wm.shape)] *= -1.
    A = A*wm
    mix = np.linalg.inv(np.eye(d)-A)
    E = np.random.default_rng(seed).normal(0,1,(d,n))
    return (mix@E).T
edges=[(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
nodes=[GraphNode(f'X{i+1}') for i in range(5)]
truth=GeneralGraph(nodes)
for i,j in edges: truth.add_directed_edge(nodes[i],nodes[j])
tc=dag2cpdag(truth)
for n in [3000, 10000]:
    X = gen_signflip(edges, n)
    G = grasp(X)
    print(f'GRaSP default n={n} SHD={SHD(tc,G).get_shd()}')
    print(G.graph.astype(int))
