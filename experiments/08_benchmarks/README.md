# experiments/08_benchmarks — bnlearn 离散基准（全量）

> 用真实离散贝叶斯网络基准（bnlearn，10000 样本）跑约束型与打分型方法，验证代码在
> 「非合成数据」上的可用性，并为「离散大图」场景提供实证（IMP #8 全量，2026-08 批次 E）。

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

## 基准脚本

```bash
cd causal-lab
python experiments/08_benchmarks/smoke_asia.py                # 冒烟：单数据集 asia
python experiments/08_benchmarks/run_all_bnlearn.py           # 全量：13 数据集 × 2 方法
python experiments/08_benchmarks/run_all_bnlearn.py <数据目录> # 或显式指定 bnlearn 数据目录
```

- 数据：`data/*.txt`（10000 样本离散，首行节点名 `X1 … Xn`）
- 方法：`PC + chisq`（约束型，全部 13 集）；`BOSS + BDeu`（离散打分，中小规模；大图见下表备注）
- 评估：真值 DAG → CPDAG 对齐 → SHD / Adjacency P/R / Arrow P/R（`scripts/evaluate.py`）
- `run_all_bnlearn.py` 逐 (数据集, 方法) 用**子进程 + wall-clock 超时**跑，挂掉的格如实标注不阻塞其余
- 产出：`results/metrics/bnlearn_all.json` + `results/metrics/08_benchmarks_asia.json` + `results/figs/bnlearn_{alarm,child,win95pts}_pc_chisq.png`

## 全量基准结果（13 集，真值 DAG→CPDAG 对齐，10000 样本）

| 数据集 | 节点 | 边 | PC+chisq SHD | BOSS+BDeu SHD | PC 时间 | BOSS 时间 |
|---|---|---|---|---|---|---|
| asia | 8 | 8 | 5 | **2** | 0.07s | 2.3s |
| cancer | 5 | 4 | 2 | **0** | 0.02s | 0.3s |
| earthquake | 5 | 4 | **0** | **0** | 0.02s | 0.4s |
| sachs | 11 | 17 | 14 | **0** 🔥 | 0.4s | 28.6s |
| survey | 6 | 6 | **2** | 6 | 0.02s | 0.4s |
| child | 20 | 25 | 10 | **4** | 1.9s | 110.4s |
| insurance | 27 | 52 | 25 | **15** | 4.7s | 309.3s |
| water | 32 | 66 | 45 | **32** | 1.1s | 102.4s |
| alarm | 37 | 46 | 6 | **超时**(>600s) | 2.9s | — |
| barley | 48 | 84 | 49 | 跳过† | 14.8s | — |
| hailfinder | 56 | 66 | 96* | 跳过† | 5.9s | — |
| hepar2 | 70 | 123 | 92 | 跳过† | 24.5s | — |
| win95pts | 76 | 112 | 57 | 跳过† | 13.2s | — |

† `BOSS+BDeu` 对 >40 节点真实图直接跳过并标注理由：alarm(37) 已 >600s 不收敛，更大图必然更慢
（离散打分的父配置分组随图规模指数膨胀——这是 **BDeu 打分开销**，非 BOSS 算法本身；可换 `BIC_from_cov` 连续评分）。

\* hailfinder 是 PC+chisq 退化最严重的一集：SHD=96、adjP/R≈0.18/0.11（alpha=0.05 固定 + 10000 样本在 56 变量上累积误报）。

**结论**：中小规模真实离散网络 **BOSS+BDeu 全面领先**（可比 8 集 6 赢 1 平 1 输，sachs/child/insurance/water 优势明显）；
但 **BOSS+BDeu 无法扩展到大图**（≥37 节点超时），**PC+chisq 是大图唯一可行选项**（质量随规模退化）。
与合成数据口径（分箱 vs 真实 logit CPD）的关系见 `knowledge/08` ②b 离散行——三口径合论：**离散最优方法取决于生成结构与规模**。

> **α 自适应治理复测（F2，`results/metrics/alpha_adaptive.json`）**：对三大大图对比 fixed=0.05 / 0.01 / Bonferroni，
> 结论为**限定性**——α 收紧只在「精度主导」的图上有效：hailfinder SHD 96→89（但召回卡死在 0.106，
> 图的问题在数据结构不在 alpha）；win95pts 在 α=0.01 有甜蜜点 57→53；hepar2/win95pts 上 Bonferroni
> 过度保守反而更差（SHD 70/97）。**推荐小步收紧 α≈0.01 作起点，而非全量 Bonferroni**。

## TODO（非批次 E 范围）

- 与官方 `TestPC.py` 基准 MD5 对照（causal-learn 自带 `benchmark_returned_results/` 可作参照）
- 大图 PC+chisq 退化治理：`alpha` 自适应（如 BIC 选择阈值）或先骨架筛选
