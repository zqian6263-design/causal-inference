import numpy as np, time
np.set_printoptions(precision=3, suppress=True)
t0=time.time()

# ===== RLCD (官方示例: 1 个共享隐变量 → 5 个观测) =====
from causallearn.graph.NodeType import NodeType
from causallearn.search.HiddenCausal.RLCD import Chi2RankTest, RLCD
rng = np.random.default_rng(1)
sample_size = 3000
latent = rng.normal(size=sample_size)
data = np.column_stack([
    1.0*latent + 0.05*rng.normal(size=sample_size),
    1.2*latent + 0.05*rng.normal(size=sample_size),
    1.4*latent + 0.05*rng.normal(size=sample_size),
    1.6*latent + 0.05*rng.normal(size=sample_size),
    1.8*latent + 0.05*rng.normal(size=sample_size),
])
data = (data - data.mean(axis=0)) / data.std(axis=0)
t=time.time()
cg = RLCD(data, ranktest_method=Chi2RankTest(data), maxk=2)
print(f'RLCD time={time.time()-t:.1f}s')
print('all_vars:', cg.all_vars)
latent_nodes = [node for node in cg.G.get_nodes() if node.get_node_type() == NodeType.LATENT]
print('latent nodes:', [node.get_name() for node in latent_nodes])
print('cg.G.graph:'); print(cg.G.graph.astype(int))
print('cg.adjacency (含隐变量):'); print(np.asarray(cg.adjacency).astype(int))

# ===== GRaSP / BOSS on 5-node linear Gaussian =====
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
np.random.seed(42); n=2000
B=np.zeros((5,5))
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i,j]=0.7
X=np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j]+=B[i,j]*X[:,i]
t=time.time(); G1 = grasp(X, depth=1, parameters={'lambda_value': 4})
print(f'GRaSP time={time.time()-t:.1f}s graph:')
print(G1.graph.astype(int))
t=time.time(); G2 = boss(X, parameters={'lambda_value': 4})
print(f'BOSS time={time.time()-t:.1f}s graph:')
print(G2.graph.astype(int))
