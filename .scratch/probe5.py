import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD

data = np.loadtxt('D:/win/causal-learn/tests/TestData/test_ges_simulated_linear_gaussian_data.txt')
truth_cpdag = np.loadtxt('D:/win/causal-learn/tests/TestData/test_ges_simulated_linear_gaussian_CPDAG.txt')
nodes = [GraphNode(f'X{i+1}') for i in range(5)]
truth_G = GeneralGraph(nodes); truth_G.graph = truth_cpdag.astype(int)

for lv in [None, 0.1, 0.5, 1.0, 2.0]:
    R = ges(data, score_func='local_score_BIC', lambda_value=lv)
    shd = SHD(truth_G, R['G']).get_shd()
    dirs = [(e.get_node1().get_name(), e.get_node2().get_name()) for e in R['G'].get_graph_edges() if R['G'].is_directed_from_to(e.get_node1(), e.get_node2())]
    print(f'lambda={lv}: SHD={shd} score={R["score"]:.1f} n_dir={len(dirs)}')

# also check with cov input to avoid recomputation
import causallearn.utils.GESUtils as GU
print('GESUtils funcs:', [x for x in dir(GU) if 'score' in x.lower()])
