import numpy as np, time
np.set_printoptions(precision=3, suppress=True)

def tic(): return time.time()

# ---------- GES ----------
from causallearn.search.ScoreBased.GES import ges
np.random.seed(42); n=2000
B = np.zeros((5,5))
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i,j]=0.7
X = np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j]+=B[i,j]*X[:,i]
t=tic(); R = ges(X, score_func='local_score_BIC'); print('GES keys:', list(R.keys()))
print('GES G.graph:\n', R['G'].graph)
print('GES score:', R['score'], 'elapsed %.3fs'%(tic()-t))

# ---------- DGES on deterministic ----------
from causallearn.search.ScoreBased.DGES import dges
np.random.seed(42); n=2000
x1 = np.random.uniform(size=n); x2 = np.random.uniform(size=n)
x3 = 2.0*x1 + 0.5*x2                      # 确定性关系, 无噪声
x4 = 0.6*x3 + np.random.uniform(size=n)   # 正常加噪
Xd = np.column_stack([x1,x2,x3,x4])
t=tic(); Rd = dges(Xd); print('DGES keys:', list(Rd.keys()))
print('DGES mindcs:', Rd['mindcs']); print('DGES det_clusters:', Rd['det_clusters'])
print('DGES G.graph:\n', Rd['G'].graph, 'elapsed %.3fs'%(tic()-t))

# standard GES on same deterministic data
t=tic(); Rg = ges(Xd, score_func='local_score_BIC'); print('GES(on det) G.graph:\n', Rg['G'].graph, 'elapsed %.3fs'%(tic()-t))

# ---------- ExactSearch ----------
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
t=tic(); dag_est, stats = bic_exact_search(X, search_method='astar'); print('Exact astar keys:', list(stats.keys()))
print('Exact astar dag_est:\n', dag_est.astype(int)); print('elapsed %.3fs'%(tic()-t))
t=tic(); dag_dp, _ = bic_exact_search(X, search_method='dp'); print('Exact dp:\n', dag_dp.astype(int), 'elapsed %.3fs'%(tic()-t))
t=tic(); dag_sp, _ = bic_exact_search(X, search_method='astar', max_parents=2); print('Exact maxP=2:\n', dag_sp.astype(int), 'elapsed %.3fs'%(tic()-t))
