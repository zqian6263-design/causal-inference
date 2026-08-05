import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD

def decode(G):
    d = G.graph; n = d.shape[0]
    directed, undirected = [], []
    for i in range(n):
        for j in range(i+1, n):
            if d[j,i]==1 and d[i,j]==-1: directed.append((i,j))
            elif d[i,j]==1 and d[j,i]==-1: directed.append((j,i))
            elif d[i,j]==-1 and d[j,i]==-1: undirected.append((i,j))
    return sorted(directed), sorted(undirected)

np.random.seed(42); n=2000
B = np.zeros((5,5))
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i,j]=0.7
X = np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j]+=B[i,j]*X[:,i]

R = ges(X, score_func='local_score_BIC')
print('GES dir:', decode(R['G'])[0])
print('GES undir:', decode(R['G'])[1])

nodes = [GraphNode(f'X{i}') for i in range(5)]
truth = GeneralGraph(nodes)
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]:
    truth.add_directed_edge(nodes[i], nodes[j])
tc = dag2cpdag(truth)
print('truth dir:', decode(tc)[0])
print('truth undir:', decode(tc)[1])
print('SHD(truth_cpdag, ges_G):', SHD(tc, R['G']).get_shd())
