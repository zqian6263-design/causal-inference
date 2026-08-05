import numpy as np
np.set_printoptions(precision=3, suppress=True)
# 真结构: x1 -> x0  (x0 = 0.8*x1 + e), 非高斯噪声
np.random.seed(42); n=4000
x1 = np.random.uniform(size=n)
x0 = 0.8*x1 + np.random.uniform(size=n)
X = np.column_stack([x0, x1])   # col0=x0, col1=x1; 真边: 1->0

from causallearn.search.FCMBased import lingam

for tag, pk in [
    ('无先验', np.full((2,2), -1)),
    ('pk[0,1]=1 (想强制 1->0)', (lambda p: (p.__setitem__((0,1),1), p)[1])(np.full((2,2), -1))),
    ('pk[1,0]=1 (想强制 0->1)', (lambda p: (p.__setitem__((1,0),1), p)[1])(np.full((2,2), -1))),
    ('pk[0,1]=0', (lambda p: (p.__setitem__((0,1),0), p)[1])(np.full((2,2), -1))),
]:
    m = lingam.DirectLiNGAM(random_state=42, prior_knowledge=pk); m.fit(X)
    print(f'{tag}: causal_order={m.causal_order_}  adj=\n{m.adjacency_matrix_}')
