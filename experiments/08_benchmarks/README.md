# experiments/08_benchmarks — bnlearn 离散基准（冒烟）

> 用真实离散贝叶斯网络基准（bnlearn）跑约束型方法，验证代码在「非合成数据」上可用。
> 批次 C 定位：**冒烟**（smoke）——每个基准跑通即可，不追求全量基准表。

## 数据源（asia 等 13 个 bnlearn 离散基准）

| 数据 | 节点 | 边 | 状态数 |
|---|---|---|---|
| asia | 8 | 8 | 2 |
| cancer / earthquake / survey | 5 / 5 / 6 | 4 / 4 / 6 | 2–3 |
| sachs / child / insurance / alarm / barley / water / hailfinder / hepar2 / win95pts / andes | 11–223 | 17–662 | 2–61 |

获取方式（任选其一）：

1. **causal-learn 官方仓库自带的 TestData**（本机首选）：
   `D:\win\causal-learn\tests\TestData\bnlearn_discrete_10000\`
   - `data/*.txt`：10000 样本离散数据（首行是节点名，如 `X1 X2 … X8`）
   - `truth_dag_graph/*.graph.txt`：真值 DAG（`Graph Edges:` 段，如 `X1 --> X2`）
   - causal-learn 仓库地址：https://github.com/py-why/causal-learn （tests/TestData/）

2. **bnlearn 官方数据仓库**（与 causal-learn 同源）：https://www.bnlearn.com/bnrepository/
   - 下载 `asia.bif` 等；用 `bnlearn`（R 包）`bnlearn::rbn()` 采样生成数据。
   - 注：causal-learn 的 `X1…Xn` 命名与 bnlearn 的 `asia` 语义名不对应，直接用
     `data/*.txt` 更省事。

## 冒烟脚本

```bash
cd causal-lab
python experiments/08_benchmarks/smoke_asia.py                # 默认找仓库同级 causal-learn/
python experiments/08_benchmarks/smoke_asia.py <bnlearn数据目录>   # 或显式指定
```

- 数据：`asia.txt`（8 节点离散，10000 样本）
- 方法：`PC + chisq`（离散数据约束型默认）
- 评估：真值 DAG → CPDAG 对齐 → SHD / Adjacency P/R / Arrow P/R（`scripts/evaluate.py`）
- 产出：`results/metrics/08_benchmarks_asia.json` + `results/figs/08_benchmarks_asia.png`

## TODO（非批次 C 范围）

- 13 个基准全量表（alarm/child/insurance 等）批量跑 + 与官方 `TestPC.py` 基准 MD5 对照
- 与合成数据结论（knowledge/08 ②b）交叉验证：真实离散图上 PC+chisq 的表现
