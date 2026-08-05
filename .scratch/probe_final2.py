import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
# ANM n=2000
from causallearn.search.FCMBased.ANM.ANM import ANM
np.random.seed(42); n=2000
x = np.random.randn(n,1); x=x/np.std(x)
e = np.random.randn(n,1); e=e/np.std(e)
y = x + 2*x**3 + e
anm=ANM()
t=time.time(); pf,pb = anm.cause_or_effect(x,y)
print(f'ANM n=2000 p_fwd={float(pf):.4f} p_bwd={float(pb):.4f} time={time.time()-t:.1f}s')

# BDeu with v-structure: X0->X2<-X1
from causallearn.search.ScoreBased.GES import ges
np.random.seed(42); n=5000
X0 = np.random.choice([0,1], size=n, p=[0.5,0.5])
X1 = np.random.choice([0,1], size=n, p=[0.5,0.5])
X2 = np.array([int((a or b) and (np.random.rand()<0.9 or a==1 or b==1)) for a,b in zip(X0,X1)])
Xd = np.column_stack([X0,X1,X2])
t=time.time(); Rd = ges(Xd, score_func='local_score_BDeu')
print('BDeu time=%.2fs graph:'%(time.time()-t)); print(Rd['G'].graph)
