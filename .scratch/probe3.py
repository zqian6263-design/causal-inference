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

# ---- Official TestGES simulation config ----
np.random.seed(42)
num_of_nodes = 5
truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
sample_size = 10000
adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
adjacency_matrix = adjacency_matrix.T
weight_mask = np.random.uniform(0.5, 0.9, (num_of_nodes, num_of_nodes))
weight_mask[np.unravel_index(np.random.choice(np.arange(weight_mask.size), replace=False,
                           size=int(weight_mask.size * 0.5)), weight_mask.shape)] *= -1.
adjacency_matrix = adjacency_matrix * weight_mask
mixing_matrix = np.linalg.inv(np.eye(num_of_nodes) - adjacency_matrix)
exogenous_noise = np.random.normal(0, 1, (num_of_nodes, sample_size))
data = (mixing_matrix @ exogenous_noise).T

R = ges(data, score_func='local_score_BIC', maxP=None, parameters=None)
print('GES dir:', decode(R['G'])[0])
print('GES undir:', decode(R['G'])[1])

nodes = [GraphNode(str(i)) for i in range(num_of_nodes)]
truth = GeneralGraph(nodes)
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]:
    truth.add_directed_edge(nodes[i], nodes[j])
tc = dag2cpdag(truth)
print('truth dir:', decode(tc)[0])
print('truth undir:', decode(tc)[1])
print('SHD:', SHD(tc, R['G']).get_shd())
