# causal-lab：因果推断系统学习与方法选型项目

> 系统梳理 causal-learn（CMU CLeaR / py-why）全部文档与代码，建立**可复用的因果推断知识库 + 方法选型能力**，为后续一切因果推断数据分析任务提供「直接选方法、设计方案」的决策支撑。

## 背景与定位

- **研究对象**：causal-learn 0.1.4.8（文档站 + GitHub `py-why/causal-learn` main 分支，本地克隆于 `D:\win\causal-learn`）
- **学习范围**：文档站全部 45 个页面 → 6 大类搜索方法、5 种独立性检验、4 类打分函数、评估/图操作/可视化工具
- **与 research-repro 的关系**：本项目的知识库与实验模板是 Phase F（因果图自动发现论文）的 **baseline 工具箱与方法储备库**——PC/FCI/GES/LiNGAM 等即是论文要对比/引用的标准方法
- **执行模式**：Hermes 顶层设计（本文件）→ Claude Code 按 PLAN.md 分阶段执行 → Hermes 验收

## 能力目标（完成后应具备）

| 层级 | 能力 | 产出物 |
|---|---|---|
| 概念 | 因果发现三范式（约束/打分/函数模型）的理论前提、输出语义差异（CPDAG vs DAG vs PAG） | knowledge/00、01 |
| 方法 | 13 个方法的假设、适用数据、参数、局限、典型用例 | knowledge/02–06 |
| 操作 | 每个方法可运行的实验脚本、评估指标、可视化 | experiments/01–08 |
| 选型 | 给定数据特征（线性/非线性、高斯/非高斯、离散/连续、有无隐变量、时序/截面）→ 方法决策 | knowledge/08（★核心交付物） |
| 复用 | 通用实验模板：数据生成→方法选择→运行→评估→报告 | experiments/10_templates/ |

## 目录结构

```
D:\win\causal-lab\
├── README.md            项目定位（本文件）
├── PLAN.md              顶层设计与分阶段执行计划（总任务书）
├── CLAUDE.md            Claude Code 项目上下文（环境硬约束）
├── .claude/settings.local.json   Claude Code 权限白名单
├── knowledge/           知识库（Claude Code 产出，中文）
│   ├── 00-因果推断基础.md        DAG/CPDAG/PAG、d-分离、三范式、可识别性
│   ├── 01-方法全景与分类.md      13 方法总览表 + 适用性初判
│   ├── 02-约束型方法.md          PC / FCI / CD-NOD（含 MVPC）
│   ├── 03-打分型方法.md          GES / DGES / ExactSearch + 打分函数
│   ├── 04-函数因果模型.md        LiNGAM 家族 / ANM / PNL
│   ├── 05-隐变量与排列方法.md    GIN / RLCD / GRaSP / BOSS
│   ├── 06-时序因果.md            Granger（回归检验 + lasso）
│   ├── 07-评估与工程实践.md      SHD/混淆矩阵、可视化、cache、数据集
│   └── 08-方法选型指南.md        ★决策树 + 速查表（Phase 3 定稿）
├── experiments/         实验代码（每个方法一个目录）
│   ├── 00_env_check.py           环境验证脚本
│   ├── 01_pc/ … 07_granger/      逐方法实验
│   ├── 08_benchmarks/            bnlearn 13 数据集基准
│   ├── 09_comparison/            统一协议横向对比
│   └── 10_templates/             ★通用实验模板（数据生成/评估/报告）
├── scripts/             公共工具（数据生成器、评估器、画图）
├── results/             全部输出（图/表/JSON 指标/报告）
└── logs/                Claude Code 运行日志
```

## 环境（硬约束）

- **Python**：`D:\Anaconda\envs\pytorch\python.exe`（Anaconda pytorch env，已验证依赖齐全：numpy 1.26.4 / scipy 1.13.1 / sklearn 1.5.1 / statsmodels 0.14.6 / networkx 3.2.1 / pandas 2.3.3 / pydot 4.0.1 / torch 2.4.0+cu124）
- **causal-learn**：pip 版 **0.1.4.8** 已装入 pytorch env（实验一律用 pip 版；`D:\win\causal-learn` 仓库源码仅作 API 深度阅读，**禁止**在仓库目录内运行实验脚本——cwd 会污染 import，导致误加载源码版）
- **PYTHONPATH 污染**：终端环境变量 `PYTHONPATH=C:\Users\win\AppData\Local\hermes\hermes-agent;...` 会劫持 import（已实测 networkx 加载到 Hermes venv）。**所有 Python 命令必须前缀 `PYTHONPATH=`**
- 不装任何 GPU/CUDA 组件；本任务纯 CPU，无需 GPU

---

# 分阶段执行计划

## Phase 0 — 环境准备（Hermes 已完成 ✅）

- [x] 克隆仓库 → `D:\win\causal-learn`
- [x] pip 安装 causal-learn 0.1.4.8 到 pytorch env
- [x] 冒烟测试：PC（fisherz）/ GES / ICA-LiNGAM 全部跑通
- [x] 验证 PYTHONPATH 剥离方法

## Phase 1 — 知识库构建（Claude Code，第一批任务）

**任务**：通读 `D:\win\causal-learn\docs\source\`（45 个 rst）与对应源码，撰写 `knowledge/00–07` 共 8 篇中文文档（08 先写框架，Phase 3 定稿）。

**每篇文档的统一结构**（方法类）：
1. 一句话定位
2. 核心思想（含最小数学形式）
3. 假设条件（哪些假设不满足会失效）
4. 输入/输出语义（CPDAG/DAG/PAG，图矩阵编码）
5. API 与参数要点（含 v0.1.2.8 后的类式 CIT 接口）
6. 优缺点与适用场景
7. 与同类方法的对比（放对比小节）
8. 最小可运行示例（引用 experiments/ 中的脚本）

**验收**：8 篇文档齐备、每篇含真实可运行的示例代码、无占位符；01 总览表与 08 选型框架一致。

## Phase 2 — 逐方法实验（Claude Code，第二批任务）

**任务**：按 `experiments/01_pc/ … 07_granger/` 逐个方法写实验脚本并运行，统一使用 `scripts/` 提供的数据生成器与评估器。

**数据生成器**（scripts/data_gen.py，4 种类型 × 2 规模）：
| 类型 | 分布/结构 | 适配方法 |
|---|---|---|
| 线性高斯 | 正态噪声，X→Y 链/叉/对撞混合 DAG | PC+fisherz、GES+BIC、ExactSearch |
| 线性非高斯 | 均匀/指数噪声（LiNGAM 前提） | LiNGAM 家族、PC+kci、GRaSP/BOSS |
| 非线性 | ANM 类（X=f(Y)+E 单向可逆） | ANM、PNL、PC+kci |
| 离散 | 多值分类变量 | PC+chisq/gsq、GES+BDeu |

默认：5 节点 7 边 DAG（沿用 TestPC.py 基准图），n=2000；另加 10 节点稀疏图验证扩展性。

**统一评估**（scripts/evaluate.py）：真值 DAG → CPDAG（dag2cpdag）后计算 SHD、Adjacency precision/recall、Arrow precision/recall（causallearn.graph 的 SHD/AdjacencyConfusion/ArrowConfusion）；记录运行时间与 PC_elapsed 等内部计时。

**输出规范**：每个方法目录产出 ① `run.py`（可复现）② `results/` 下 PNG 图 + `metrics.json` ③ 实验小结追加到 `knowledge/` 对应章节的「实验验证」小节。

**seed 纪律**：所有随机过程固定 seed=42（np.random + 方法自带 random_state）；禁止调 seed 刷指标。

## Phase 3 — 横向对比与选型指南（Claude Code，第三批任务）

**任务**：`experiments/09_comparison/` 用统一协议在 4 种数据上跑全部方法，产出对比矩阵（方法 × 数据类型 → SHD/PR/时间），据此**定稿 `knowledge/08-方法选型指南.md`**。

**选型指南必须包含**：
1. 决策树（数据特征 → 方法）：数据是否有隐变量→约束型选 FCI；线性非高斯→LiNGAM；非线性→ANM/PNL/PC+kci；离散→chisq/BDeu；时序→Granger/VAR-LiNGAM；样本小且图小→ExactSearch……
2. 方法速查表（每方法一行：假设/输出/时间成本/代表用例）
3. 「常见场景配方」：如「遥感多模态分类的因果特征提取」→ 推荐管道（示例：PC 找骨架 + 领域知识定向 + 因果特征选择）
4. 局限与坑（KCI 慢、PC 只能到 CPDAG、fisherz 线性假设等）

**验收**：决策树能对任意新数据集给出 1–2 个推荐方法 + 理由；配方覆盖用户研究方向（图像处理+因果推断）。

## Phase 4 — 通用实验模板（Claude Code，收尾）

**任务**：将 scripts/ 数据生成器 + evaluate.py + 各方法 run.py 的模式提炼为 `experiments/10_templates/`：
- `template_data_gen.py`（用户数据 → 适配 causal-learn 输入，含缺失值处理说明）
- `template_pipeline.py`（数据→方法选择（读 knowledge/08）→运行→评估→图表→Markdown 报告）
- `TEMPLATE_README.md`（使用说明：用户拿到任意因果数据分析任务时如何套用）

**验收**：模板可直接复制到新任务目录运行；README 说明覆盖「我有数据，怎么选方法」的完整路径。

## 全局验收标准

1. 13 个搜索方法全部跑通并各有 1 个可复现实验
2. knowledge/ 9 篇全部完成且互相引用一致
3. 选型指南可回答「任意新数据集用什么方法」
4. 模板可复用（未来任务零门槛启动）
5. 每阶段 git commit，REPORT.md 记录进度

## 实验设计规范（通用）

- 真值图必须已知（合成数据）才能评估；真实数据集（sachs/boston_housing/airfoil，causallearn.utils.Dataset.load_dataset）只做方法演示不做定量评估
- 约束型方法结果 → CPDAG/PAG；打分/函数模型 → DAG/顺序；评估前必须对齐语义（用 dag2cpdag 转换真值）
- 所有指标 JSON 落盘 `results/metrics/`，图表 `results/figs/`
- 运行耗时 > 5 分钟的实验：后台跑 + 记录时间戳
