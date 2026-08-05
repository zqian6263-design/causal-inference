import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()

# ===== ANM n=3000 (TestANM 风格) =====
from causallearn.search.FCMBased.ANM.ANM import ANM
np.random.seed(42)
n=3000
x = np.random.randn(n,1); x = x/np.std(x)
e = np.random.randn(n,1); e = e/np.std(e)
y = x + 2*x**3 + e
anm = ANM()
t=time.time(); pf, pb = anm.cause_or_effect(x, y)
print(f'ANM n=3000: p_fwd={float(pf):.4f} p_bwd={float(pb):.4f} time={time.time()-t:.1f}s')

# ===== PNL n=200 =====
from causallearn.search.FCMBased.PNL.PNL import PNL
np.random.seed(42)
xp = np.random.randn(200,1); xp = xp/np.std(xp)
ep = np.random.randn(200,1); ep = ep/np.std(ep)
yp = (xp + xp**3 + ep)**2
pnl = PNL()
t=time.time(); pf2, pb2 = pnl.cause_or_effect(xp, yp)
print(f'PNL: p_fwd={float(pf2[0]):.3f} p_bwd={float(pb2[0]):.3f} time={time.time()-t:.1f}s')

# ===== GIN (TestGIN case1) =====
from causallearn.search.HiddenCausal.GIN.GIN import GIN
sample_size = 500
np.random.seed(42)
L1 = np.random.uniform(-1, 1, size=sample_size)
L2 = np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
X1 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X2 = np.random.uniform(1.2, 1.8) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X3 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
X4 = np.random.uniform(1.2, 1.8) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
data = np.array([X1, X2, X3, X4]).T
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
t=time.time(); G, K = GIN(data, indep_test_method='kci', alpha=0.05)
print(f'GIN: K={K} time={time.time()-t:.1f}s')
print('GIN G.graph:'); print(G.graph)
