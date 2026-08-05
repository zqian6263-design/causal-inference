import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.graph.SHD import SHD

# Load official test data & truth CPDAG
data = np.loadtxt('D:/win/causal-learn/tests/TestData/test_ges_simulated_linear_gaussian_data.txt')
truth_cpdag = np.loadtxt('D:/win/causal-learn/tests/TestData/test_ges_simulated_linear_gaussian_CPDAG.txt')
print('data shape:', data.shape, 'truth cpdag:')
print(truth_cpdag.astype(int))

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
num_of_nodes = 5
nodes = [GraphNode(f'X{i+1}') for i in range(num_of_nodes)]
truth_G = GeneralGraph(nodes)
truth_G.graph = truth_cpdag.astype(int)

R = ges(data, score_func='local_score_BIC', maxP=None, parameters=None)
G = R['G']
print('GES graph:')
print(G.graph)
shd = SHD(truth_G, G)
print('SHD:', shd.get_shd())

# Also test on my n=2000 data but with the same node naming
np.random.seed(42); n=2000
B = np.zeros((5,5))
for i,j in [(0,1),(0,3),(1,2),(1,3),(2,3),(2,4),(3,4)]: B[i,j]=0.7
X = np.random.randn(n,5)
for j in range(5):
    for i in range(5):
        if B[i,j]: X[:,j]+=B[i,j]*X[:,i]
R2 = ges(X, score_func='local_score_BIC')
nodes2 = [GraphNode(f'X{i+1}') for i in range(5)]
truth2 = GeneralGraph(nodes2); truth2.graph = truth_cpdag.astype(int)
print('my n=2000 GES SHD vs official cpdag:', SHD(truth2, R2['G']).get_shd())
print('my GES graph:'); print(R2['G'].graph)
