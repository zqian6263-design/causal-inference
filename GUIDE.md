# causal-lab 使用指南（GUIDE.md）

> 系统梳理 causal-learn（CMU CLeaR 官方因果发现库）全部文档与代码的产物——目标是让你拿到任意因果推断数据分析任务时，**直接选方法、设计方案**。

## 快速导航

| 想干什么 | 看哪里 |
|---|---|
| 学习方法论（9 篇知识库） | `knowledge/00` → `knowledge/08` |
| 查「我的数据用什么方法」 | `knowledge/08-方法选型指南.md`（决策树+实证矩阵） |
| 跑已有实验 | `experiments/00-09` 全部 run.py（13 方法全可复现，含 bnlearn 基准） |
| ⭐ 做自己的因果分析 | 复制 `experiments/10_templates/` |
| 查实证指标 | `results/comparison_report.md`、`results/metrics/*.json` |
| 总任务书/设计 | `PLAN.md` |

## 环境（必须先知道）

- 依赖：`requirements.txt` 锁定全部版本（causal-learn 0.1.4.8）
- 安装：`pip install -r requirements.txt`
- 官方源码（可选，仅供阅读 API）：仓库同级目录 `causal-learn/`（官方仓库克隆）
- ⚠️ 本机开发环境（Anaconda pytorch env + `PYTHONPATH=` 前缀）说明见 `CLAUDE.md`，仅本机需要

## 三种使用场景

### 1️⃣ 学习
按 `00 → 01 → … → 08` 顺序读知识库，每篇末尾示例可运行：

```bash
cd causal-lab
python experiments/03_ges_dges/run.py
```

### 2️⃣ 复现/跑实验
```bash
python experiments/04_lingam_anm_pnl/run.py   # 单方法
python experiments/09_comparison/run_all.py   # 全对比矩阵
```

### 3️⃣ ⭐ 新分析任务（推荐流程）
```bash
# ① 复制模板
cp -r experiments/10_templates/ 我的新任务/
# ② 改 template_data_gen.py 的 load_your_data() → 你的数据
# ③ 体检 + 方法建议（自动判断连续/离散/缺失/时序）
python 我的新任务/template_data_gen.py
# ④ 一键管道（数据→方法→评估→Markdown 报告）
python 我的新任务/template_pipeline.py
```

完整演示见 `experiments/demo_remote_sensing/`（遥感特征-标签案例，含混淆结构）。

## 方法选型速记（实测结论，详见 08）

| 数据 | 选 | 实测 SHD |
|---|---|---|
| 连续线性高斯 | PC+fisherz / GES+BIC / BOSS | 0 |
| 连续线性非高斯 | ICA-LiNGAM | 完整 DAG |
| 离散 | BOSS | 1 |
| 非线性 | PC+KCI（大样本）/ ANM·PNL 成对 | 线性方法全受限 |
| 时序 | Granger / VAR-LiNGAM | — |
| 疑似隐变量 | FCI / RLCD | — |
| 缺失值 | PC + mv_fisherz | — |

## 常见坑（踩过实录）

1. 本机终端 `PYTHONPATH=` 前缀不可省（防 Hermes venv 劫持 import，见 `CLAUDE.md`）；其他机器直接 `python` 即可
2. 真值图评估先 `dag2cpdag` 对齐（约束/打分型输出 CPDAG）
3. GRaSP 无 seed 参数、结果随机——多跑取稳
4. KCI 慢且小样本检验力不足（1200 样本实测漏边）
5. 用 `d_separation` 检验前先 `import scripts.patch_d_separation`
6. GRaSP/BOSS 图矩阵约定：`G.graph[j,i]=1 且 G.graph[i,j]=-1` → i→j
