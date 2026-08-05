import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()

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

edges = [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]

# ---- GES ----
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
nodes=[GraphNode(f'X{i+1}') for i in range(5)]
truth=GeneralGraph(nodes)
for i,j in edges: truth.add_directed_edge(nodes[i],nodes[j])
tc=dag2cpdag(truth)

Xg = gen_signflip(edges, 10000)
t=time.time(); R = ges(Xg, score_func='local_score_BIC')
print('GES time=%.2fs SHD=%d'%(time.time()-t, SHD(tc,R['G']).get_shd()))
print('GES graph:'); print(R['G'].graph)
print('GES score=%.2f'%R['score'])

# ---- ExactSearch astar ----
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
t=time.time(); dag_est, stats = bic_exact_search(Xg, search_method='astar')
print('Exact astar time=%.2fs stats=%s'%(time.time()-t, stats))
print('dag_est:'); print(dag_est.astype(int))
# convert to cpdag
from causallearn.graph.Dag import Dag
nd=[GraphNode(str(i)) for i in range(5)]
D=Dag(nd)
for i,j in zip(*np.where(dag_est==1)): D.add_directed_edge(nd[i],nd[j])
ec=dag2cpdag(D)
print('Exact CPDAG vs truth SHD:', SHD(tc,ec).get_shd())

# ---- DGES deterministic ----
from causallearn.search.ScoreBased.DGES import dges
np.random.seed(42); n=2000
x1 = np.random.uniform(size=n); x2 = np.random.uniform(size=n)
x3 = 2.0*x1 + 0.5*x2
x4 = 0.6*x3 + np.random.uniform(size=n)
Xd = np.column_stack([x1,x2,x3,x4])
t=time.time(); Rd = dges(Xd)
print('DGES time=%.2fs'%(time.time()-t))
print('DGES mindcs:', Rd['mindcs'])
print('DGES det_clusters:', [c.tolist() for c in Rd['det_clusters']])
print('DGES G.graph:'); print(Rd['G'].graph.astype(int))
