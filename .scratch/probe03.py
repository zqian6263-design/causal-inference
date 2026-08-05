import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.DGES import dges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

np.set_printoptions(precision=3, suppress=True)

# ---- 5-node benchmark linear gaussian, seed=42 ----
np.random.seed(42)
n = 2000
edges = [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]
B = np.zeros((5,5))
for i,j in edges: B[i,j] = 0.7
X = np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j] += B[i,j]*X[:,i]

print('=== GES ===')
Rec = ges(X, score_func='local_score_BIC', maxP=None, parameters=None)
G = Rec['G']
print('keys:', list(Rec.keys()))
print('score:', Rec['score'])
print('graph:\n', G.graph)
print('directed:', sorted(G.find_fully_directed()))
print('undirected:', sorted(G.find_undirected()))

print('=== ExactSearch dp/astar ===')
Xc = X - X.mean(0)
dag_dp, stats_dp = bic_exact_search(Xc, search_method='dp', use_path_extension=True)
print('dp stats keys:', list(stats_dp.keys()))
print('dp stats:', {k: v for k,v in stats_dp.items() if not isinstance(v, np.ndarray)})
print('dp dag:\n', dag_dp.astype(int))
dag_astar, stats_astar = bic_exact_search(Xc, search_method='astar', use_path_extension=True)
print('astar dag:\n', dag_astar.astype(int))

print('=== DGES deterministic ===')
np.random.seed(42)
n2 = 2000
X0 = np.random.rand(n2)
X1 = 2.0*X0 + np.random.rand(n2)
X2 = 3.0*X1                      # deterministic: X2 = f(X1)
X3 = 0.8*X2 + np.random.rand(n2)
D = np.column_stack([X0, X1, X2, X3])
try:
    R2 = ges(D)   # standard GES on deterministic data
    print('GES-on-det: score=', R2['score'], 'graph:\n', R2['G'].graph)
except Exception as e:
    print('GES-on-det FAILED:', repr(e)[:200])
Rd = dges(D)
print('DGES keys:', list(Rd.keys()))
print('DGES score:', Rd['score'])
print('DGES graph:\n', Rd['G'].graph)
print('mindcs:', Rd['mindcs'])
print('det_clusters:', Rd['det_clusters'])
