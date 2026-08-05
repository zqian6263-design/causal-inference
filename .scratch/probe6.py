import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()

# ===== GES with lambda=1.0 on n=2000 5-node =====
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
np.random.seed(42); n=2000
B=np.zeros((5,5))
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i,j]=0.7
X=np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j]+=B[i,j]*X[:,i]
nodes=[GraphNode(f'X{i+1}') for i in range(5)]
truth=GeneralGraph(nodes)
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: truth.add_directed_edge(nodes[i],nodes[j])
tc=dag2cpdag(truth)
R=ges(X, score_func='local_score_BIC', lambda_value=1.0)
print('GES lambda=1.0 SHD:', SHD(tc,R['G']).get_shd())
print(R['G'].graph)
print('time %.2fs'%(time.time()-t0))
