import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()
from causallearn.search.FCMBased import lingam

# ===== ICA-LiNGAM / DirectLiNGAM: 5 var linear non-Gaussian =====
np.random.seed(42); n=4000
x3 = np.random.uniform(size=n)
x0 = 3.0*x3 + np.random.uniform(size=n)
x2 = 6.0*x3 + np.random.uniform(size=n)
x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=n)
x4 = 4.0*x0 - 1.0*x2 + np.random.uniform(size=n)
X = np.column_stack([x0,x1,x2,x3,x4])

t=time.time()
m = lingam.ICALiNGAM(random_state=42); m.fit(X)
print('ICA causal_order_:', m.causal_order_)
print('ICA adjacency_matrix_:')
print(np.round(m.adjacency_matrix_,3))
print('ICA time %.2fs'%(time.time()-t))

t=time.time()
md = lingam.DirectLiNGAM(random_state=42); md.fit(X)
print('Direct causal_order_:', md.causal_order_)
print('Direct time %.2fs'%(time.time()-t))

# with prior knowledge: force 3->0 (x3 causes x0), forbid 0->3
import numpy as np
pk = np.full((5,5), -1); pk[3,0]=1; pk[0,3]=0
md2 = lingam.DirectLiNGAM(random_state=42, prior_knowledge=pk); md2.fit(X)
print('Direct(prior) causal_order_:', md2.causal_order_)

# ===== VAR-LiNGAM =====
np.random.seed(42); T=2000
A = np.array([[0, 0.5, 0], [0.3, 0, 0.4], [0, 0.2, 0]])
Y = np.zeros((T,3)); err = np.random.uniform(-1,1,(T,3))
for t in range(1,T):
    Y[t] = A @ Y[t-1] + err[t]
mv = lingam.VARLiNGAM(lags=1, criterion='bic', prune=False); mv.fit(Y)
print('VAR causal_order_:', mv.causal_order_)
print('VAR lag1 matrix:'); print(np.round(mv.adjacency_matrices_[0],3))
print('--- %.2fs ---'%(time.time()-t0))
