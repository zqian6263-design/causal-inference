import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.random.seed(42); T=2000
A = np.array([[0, 0.5, 0], [0.3, 0, 0.4], [0, 0.2, 0]])
Y = np.zeros((T,3)); err = np.random.uniform(-1,1,(T,3))
for t in range(1,T):
    Y[t] = A @ Y[t-1] + err[t]

from causallearn.search.FCMBased import lingam
mv = lingam.VARLiNGAM(lags=1, criterion=None, prune=False)
mv.fit(Y)
print('lags:', mv._lags)
print('ar_coefs (M_taus):'); print(np.round(mv._ar_coefs,3))
print('residuals shape:', mv.residuals_.shape, 'std:', np.round(mv.residuals_.std(axis=0),3))
print('causal_order_:', mv.causal_order_)
print('adjacency_matrices_:')
for k in range(mv._lags):
    print(' lag', k+1, ':'); print(np.round(mv.adjacency_matrices_[k],3))
# DirectLiNGAM prior semantics test
x3 = np.random.uniform(size=4000); x0 = 3.0*x3+np.random.uniform(size=4000); x2=6.0*x3+np.random.uniform(size=4000)
x1=3.0*x0+2.0*x2+np.random.uniform(size=4000); x4=4.0*x0-1.0*x2+np.random.uniform(size=4000)
X=np.column_stack([x0,x1,x2,x3,x4])
pk=np.full((5,5),-1); pk[3,0]=1; pk[0,3]=0
md2=lingam.DirectLiNGAM(random_state=42, prior_knowledge=pk); md2.fit(X)
print('prior order:', md2.causal_order_)
print('prior adjacency:'); print(np.round(md2.adjacency_matrix_,3))
