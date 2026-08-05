import numpy as np
np.set_printoptions(precision=3, suppress=True)
from causallearn.search.ScoreBased.DGES import dges
from causallearn.search.ScoreBased.GES import ges
np.random.seed(42); n=2000
x1 = np.random.uniform(size=n); x2 = np.random.uniform(size=n)
x3 = 2.0*x1 + 0.5*x2                      # 确定性: x3 = f(x1,x2), 无噪声
x4 = 0.6*x3 + np.random.uniform(size=n)   # 正常加噪
Xd = np.column_stack([x1,x2,x3,x4])
Rd = dges(Xd)
print('DGES keys:', list(Rd.keys()))
print('DGES mindcs:', Rd['mindcs'])
print('DGES mindc_sets:', [sorted(s) for s in Rd['mindc_sets']])
print('DGES det_clusters:', [c.tolist() for c in Rd['det_clusters']])
print('DGES G.graph:'); print(Rd['G'].graph.astype(int))
print('DGES score:', round(Rd['score'],2))
print()
Rg = ges(Xd, score_func='local_score_BIC')
print('GES(on det data) G.graph:'); print(Rg['G'].graph.astype(int))
print('GES score:', round(Rg['score'],2))
