import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()

def make_ges_data(n, seed=42):
    rng = np.random.default_rng(seed)
    edges={(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)}
    A=np.zeros((5,5))
    A[tuple(zip(*edges))]=1
    A=A.T
    wm=rng.uniform(0.5,0.9,(5,5))
    wm[np.unravel_index(rng.choice(np.arange(25), size=int(25*0.5), replace=False), wm.shape)] *= -1.
    A=A*wm
    mix=np.linalg.inv(np.eye(5)-A)
    E=rng.normal(0,1,(5,n))
    return (mix@E).T

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
nodes=[GraphNode(f'X{i+1}') for i in range(5)]
truth=GeneralGraph(nodes)
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: truth.add_directed_edge(nodes[i],nodes[j])
tc=dag2cpdag(truth)

for n in [3000, 10000]:
    X=make_ges_data(n)
    for lv in [0.5, 1.0]:
        R=ges(X, score_func='local_score_BIC', lambda_value=lv)
        print(f'n={n} lambda={lv} SHD={SHD(tc,R["G"]).get_shd()}')
X=make_ges_data(3000)
R=ges(X, score_func='local_score_BIC', lambda_value=1.0)
print('GES graph (n=3000, lambda=1.0):')
print(R['G'].graph)
print('--- %.2fs ---'%(time.time()-t0))
