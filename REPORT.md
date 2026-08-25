# 项目进度报告

## Phase 0 — 环境准备（2026-08-05，Hermes）
- [x] 克隆 py-why/causal-learn main → D:\win\causal-learn
- [x] causal-learn 0.1.4.8 装入 Anaconda pytorch env
- [x] 冒烟测试通过：PC(fisherz) / GES / ICA-LiNGAM
- [x] 顶层设计完成：PLAN.md / CLAUDE.md / README.md


## Phase 1 批1 — 知识库 00-02（2026-08-05，Claude Code 产出 + Hermes 核验）
- [x] knowledge/00-因果推断基础.md（DAG/d-分离/等价类/CPDAG-PDAG-PAG/三大范式/SCM，12.5KB）
- [x] knowledge/01-方法全景与分类.md（13 方法总览表 + 5 检验 + 4 打分 + 快速初判，9.2KB）
- [x] knowledge/02-约束型方法.md（PC/FCI/CD-NOD/MVPC 深入，14.8KB）
- [x] experiments/00_env_check.py 复现脚本，Hermes 亲自运行核验：5 组示例全部通过 ✓
- [x] 环境发现（实测）：networkx 3.2.1 缺 `is_d_separator` → `d_separation` 独立性检验不可用；ICA-LiNGAM 在纯高斯数据上必然不收敛（非高斯前提）
- 坑：Claude Code `--max-turns 40` 写 3 篇深度文档不够收尾 → 后续批次拆小/加 turns

## Phase 1 批2-批3 — 知识库 03-08 + 实验脚本（2026-08-06，Hermes 直接产出）
- [x] knowledge/03-打分型方法.md（GES/DGES/ExactSearch + 打分函数）
- [x] knowledge/04-函数因果模型.md（LiNGAM 家族/ANM/PNL）
- [x] knowledge/05-隐变量与排列方法.md（GIN/RLCD/GRaSP/BOSS）
- [x] knowledge/06-时序因果.md（Granger）
- [x] knowledge/07-评估与工程实践.md（SHD/可视化/cache/数据集）
- [x] knowledge/08-方法选型指南.md（★决策树+速查表+场景配方+SOP）
- [x] 复现脚本 experiments/03_ges_dges/run.py、04_lingam_anm_pnl/run.py、05_gin_rlcd/run.py、07_granger/run.py 全部实跑通过
- 执行策略调整：Claude Code 两轮 max-turns 耗尽零产出（过度探针）→ 知识库改由 Hermes 直接产出（文档已全部消化），Claude Code turns 留给 Phase 2 实验
- 实测新发现：DGES MinDC={0,1,2} 正确检测；GRaSP 完美恢复(SHD=0)；BOSS 此数据 SHD=5；RLCD 检出 L1；Granger lasso 捕捉 x1→x2

## Phase 2/3 补充 — KCI 非线性实验（2026-08-06）
- [x] PC+KCI 实测（results/metrics/kci_supplement.json）: 1200 样本下 nonlinear SHD=7 / linear SHD=6, 75-105s/数据集
- 发现: KCI 非参数但**样本需求大**（1200 样本检验力不足, 漏 2 边定向失败）→ 08 选型指南已回填三条非线性建议

## 优化批次 A — 可复现性与仓库卫生（2026-08-06，Claude Code 执行）

**做了什么**
- [x] A1 根目录新建 `requirements.txt`：10 个依赖锁定版本（causal-learn 0.1.4.8 / numpy 1.26.4 / scipy 1.13.1 / scikit-learn 1.5.1 / statsmodels 0.14.6 / networkx 3.2.1 / pandas 2.3.3 / pydot 4.0.1 / joblib 1.5.2 / matplotlib 3.9.2）——已用本机 pytorch env `pip show` 实测核对，全部一致
- [x] A2 去机器专属路径：README/GUIDE/PLAN/CLAUDE/knowledge 00-08 参考·验证输出小节/experiments 各 run.py 顶部注释/模板管道与 TEMPLATE_README，全部改为相对描述；运行命令统一 `python`；README 加「安装」一节（`pip install -r requirements.txt`）；causal-learn 源码改称「仓库同级目录 `causal-learn/`（官方克隆，可选，仅供阅读 API），pip 版即可运行一切实验」；本机开发环境说明**仅保留在 CLAUDE.md** 一处
- [x] A3 仓库卫生：`.gitignore` 追加 `.claude/settings.local.json`、`results/metrics/cache_*.json`；`git rm --cached .claude/settings.local.json`（本地文件保留）；根目录新增 `LICENSE`（文档 CC-BY-4.0 + 代码 MIT 两段式）；空目录 01_pc / 02_fci / 06_grasp_boss / 08_benchmarks 各放 `.gitkeep`
- [x] A4 文档结构声明对齐：README/GUIDE/PLAN 中引用 01_pc / 02_fci / 06_* / 08_benchmarks 处标注「建设中（批次 C）」

**注意 / 待 Hermes 决策**
- ⚠️ `results/metrics/` 在 `.gitignore` 中为整目录忽略，`comparison.json`、`kci_supplement.json` 目前**未入库**（与 A3「是证据，保留」的意图冲突）。本次按任务书仅「追加」两行，未改既有规则；若需证据入库，应把 `results/metrics/` 收窄为 `results/metrics/cache_*.json` 后 `git add -f` 两文件——留给 Hermes 定夺。

**下一步**：批次 B（修实证证据矛盾：多 seed 重跑 + BOSS/BDeu 归因修正 + ArrowConfusion NaN + LiNGAM 统一 SHD 矩阵）。

## 优化批次 B — 修实证证据矛盾（2026-08-06，Claude Code 执行）

**做了什么**
- [x] B1 改造 `experiments/09_comparison/run_all.py`：
  - ① 每个方法跑 **5 个 seed（42/1/7/2024/999）**，数据逐 seed 重生成，SHD 报 **mean±std**；GRaSP/BOSS 内部随机按 `random.seed(seed)` 固定 → 整次运行可复现（此前每次运行结果都变，证据不稳定）
  - ② GRaSP/BOSS 在离散数据上**显式对比** `local_score_BIC_from_cov`（默认）vs `local_score_BDeu`——修正「BOSS 碾压归因 BDeu」的错误结论
  - ③ 修 Arrow/Adjacency Confusion 无边 0/0：**根因在共享 `scripts/evaluate.py`**（`get_arrows_precision()` 返回 nan、`get_adj_precision()` 抛 ZeroDivisionError），改为计数 + `_safe_ratio` 统一置 0 → 全部实验受益，JSON 不再出现 NaN 字面量
  - ④ ICA-LiNGAM 纳入统一矩阵：`adjacency_matrix_` 阈值 0.1 → 邻接掩码 → **dag2cpdag**（与其余方法同 CPDAG 口径）→ evaluate_graph。**发现并修复关键 bug**：causal-learn 的 `adjacency_matrix_[i,j]` 是 X_j 在 X_i 方程中的系数（边 j→i），相对本仓库 adj[i,j]=边 i→j 约定**需转置**（实测 mask.T==真值）；未转置时 LiNGAM 的 CPDAG 全无向、SHD 虚高为 5
  - ⑤ 输出合法 JSON（`allow_nan=False`）+ 更新 `results/comparison_report.md`
- [x] B2 用新数据重写 `knowledge/08` ②b 实证对比矩阵：每格 mean±std；修正 GRaSP 离散 SHD 矛盾（旧文档 8 vs JSON 1 → 多 seed 真实值 6.0±3.0）；修正 BOSS 归因；补 LiNGAM 的 SHD 行；结论改「多 seed 平均」口径；同步修正 ③ 场景 A（旧「GRaSP 实测 SHD=0」）、④ 坑速查（旧「BOSS 偶发不稳、GRaSP 优先」）、⑤ SOP（seed=42 → 多 seed）

**关键数值**（3000 样本 × 5 seed，5 节点 7 边，SHD mean±std）

| 数据 \ 方法 | PC | GES | GRaSP+BIC | GRaSP+BDeu | BOSS+BIC | BOSS+BDeu | ICA-LiNGAM |
|---|---|---|---|---|---|---|---|
| 线性高斯 | 4.8±2.5 | 5.4±3.1 | 6.6±1.4 | — | **0.0±0.0** | — | — |
| 线性非高斯 | 5.6±1.6 | 6.6±1.4 | 5.2±2.9 | — | 2.4±3.0 | — | **0.0±0.0** |
| 非线性 ANM | **4.4±0.5** | 4.8±0.4 | 5.8±1.0 | — | 5.0±0.0 | — | — |
| 离散 | 6.8±1.2 | 6.4±1.0 | 6.0±3.0 | 6.0±0.9 | **2.4±2.8** | 6.0±0.9 | — |

**修正的结论**
1. 旧「PC/GES/BOSS 线性高斯完美（SHD=0）」是 **seed=42 的运气**：多 seed 下 PC/GES/GRaSP 平均 4.8–6.6，**仅 BOSS 稳定完美**
2. 旧「BOSS 碾压归因 BDeu」**双重错误**：默认就是 BIC_from_cov；显式用 BDeu 反而退化（6.0±0.9 > 2.4±2.8）
3. 旧「ICA-LiNGAM 完美（因果序+边数）」现用统一 CPDAG 尺子严谨验证：**SHD=0.0±0.0（含完整箭头）**，结论成立
4. GRaSP 离散「8 vs 1」矛盾消解：多 seed 真实值 **6.0±3.0**（单 seed 从 1 到 9 高方差，排列型必须多 seed）

**注意 / 决策记录**
- **LiNGAM CPDAG 口径**：为落实「与其他方法同一把尺子」，LiNGAM 的完整 DAG 先经 dag2cpdag 对齐到 CPDAG 再评估（直接对比真值 CPDAG 会因「比等价类更具体」而虚高 SHD）。这是对任务书「同一把尺子」的工程解释，已写入 run_all.py docstring 与 08 文档
- 仓库卫生顺带：`git rm --cached` 两个历史误提交的 `scripts/__pycache__/*.pyc`（与 `.gitignore` 的 `__pycache__/` 矛盾）
- 实验实跑验证（~16s/次）：SHD/P/R **完全可复现**（数据 RandomState + 方法 random.seed 双重固定；`time_s` 为墙钟计时，随负载漂移属正常）；`comparison.json` 已通过 `json.load` + 无 NaN/Infinity 字面量检查

**下一步**：批次 C（补实验缺口：01_pc / 02_fci / 06_grasp_boss / GIN / VAR-LiNGAM / PNL / results/figs 图表落盘）。

## 优化批次 C — 补实验缺口（2026-08-06，Claude Code 执行）

**做了什么**
- [x] C1a `experiments/01_pc/run.py`：PC 专属实验——PC+fisherz（线性高斯 n=3000，**SHD=0** 完美）；PC+kci（同真值图 n=600，SHD=6、20.6s，验证 KCI 样本需求大）
- [x] C1b `experiments/02_fci/run.py`：FCI + 隐变量——真值 L 混淆 X1/X2（观测只含 X1..X5）；评估用 **dag2pag 对齐**（非 dag2cpdag，CPDAG 表达不了圆圈端点）→ **SHD(PAG)=0、Adj P/R=1.0/1.0**，FCI 精确恢复真值 PAG
- [x] C1c `experiments/02_fci/cdnod_example.py`：CD-NOD 时变数据（c_indx=时间索引, kci）——机制分段切换 → 恢复 **X1→X2** 且 **C→X2**（指出 X2 机制在变），1.3s
- [x] C1d `experiments/06_grasp_boss/run.py`：GRaSP/BOSS 5-seed 稳定性（random.seed 固定）——线性高斯 GRaSP 6.6±1.36 vs **BOSS 0.0±0.0**；线性非高斯 5.2±2.93 vs 2.4±3.01（BOSS 明显更稳；与批次 B comparison.json 数值完全一致）
- [x] C1e `experiments/04_lingam_anm_pnl/run.py` 扩展：VAR-LiNGAM（2 变量 lags=2：lag-1 X1→X2=0.52、lag-2=-0.34，真值 0.5/-0.3）；PNL（后非线性 n=400：p_fwd=0.684/p_bwd=0.0 → x→y，import+run 67s）；RCD 最小调用（`lingam.RCD`，np.random.seed 固定可复现）
- [x] C1f `experiments/05_gin_rlcd/run.py` 扩展：GIN（LiNLAM 隐变量+非高斯）——恢复隐变量簇 **[[0,1],[2,3]]**（L1→{X1,X2}, L2→{X3,X4}, L1→L2），0.19s
- [x] C1g RCD/CAM-UV：RCD 加最小调用；**CAM-UV 标注「官方实现，未单独实验」**（依赖 pygam 本机未装，禁止装组件）
- [x] C2 图表落盘：10 张 PNG → results/figs/（新增 `scripts/plotting.py`：本机无 graphviz `dot` 二进制，改 matplotlib+networkx 直渲 CPDAG/PAG）
- [x] C3 `experiments/08_benchmarks/`：README（bnlearn 数据源说明）+ smoke_asia.py（asia 10000 样本 PC+chisq → SHD=5、adjP/R=0.83/0.63，真值 DAG 对齐评估）

**关键实测数值**

| 方法 | 数据 | 结果 |
|---|---|---|
| PC+fisherz | 线性高斯 n=3000 | SHD=0（完美） |
| PC+kci | 同真值图 n=600 | SHD=6, 20.6s（KCI 样本需求大） |
| FCI (fisherz) | 5 观测 + 1 隐变量 L | **SHD(PAG)=0**, Adj P/R=1.0（dag2pag 对齐） |
| CD-NOD (kci) | 时变 600 点 | X1→X2 + C→X2（机制变化检出） |
| GRaSP / BOSS | 线性高斯 ×5 seed | 6.6±1.36 / **0.0±0.0** |
| GRaSP / BOSS | 线性非高斯 ×5 seed | 5.2±2.93 / 2.4±3.01 |
| VAR-LiNGAM | 2 变量 lags=2 | lag-1 0.52 / lag-2 -0.34（真值 0.5/-0.3） |
| PNL | 后非线性 n=400 | x→y（p=0.684/0.0），import+run 67s |
| GIN | LiNLAM n=500 | 隐变量簇 [[0,1],[2,3]] 精确恢复 |
| RLCD | 1 隐变量→5 观测 | 检出 L1 |
| PC+chisq | bnlearn asia 10000 | SHD=5, adjP/R=0.83/0.63 |

**注意 / 决策记录**
- **图渲染根因**：本机无 graphviz `dot` 二进制，`to_pydot().write_png()` 与 `GraphUtils.plot_graph` 均失败（knowledge/07 旧「pydot 可用」说法已修正）→ 新增 `scripts/plotting.py`（matplotlib+networkx，CPDAG/PAG 端点全支持）
- **FCI 评估语义**：PAG 必须 dag2pag 对齐（不能 dag2cpdag）；SHD(PAG) 计端点差异（圆圈 vs 箭头），比 CPDAG SHD 更严格——本实验 FCI 恰好精确恢复（SHD=0）
- **causal-learn `get_endpoint(a,b)` 坑**：返回的是 **b 端**端点（非 a 端），初版方向画反，已修正并加断言验证（plotting.py / 02_fci decode 均改对）
- 7 个 metrics JSON 全部合法（无 NaN 字面量）+ 10 张图落盘；实验全部实跑验证
- GRaSP/BOSS 数值与批次 B comparison.json 完全一致（跨脚本交叉验证通过）

**下一步**：批次 D（模板自包含、demo README 重写、data_gen 死代码清理、CI 冒烟可选）。

## 优化批次 D — 收尾工程（2026-08-06，Claude Code 执行）

**做了什么**
- [x] D1 **10_templates 完全自包含**：`template_pipeline.py` 移除对仓库根的 `sys.path.insert(..., "..", "..")`，把 evaluate 逻辑（`evaluate_graph` / `graph_from_adj` / `_safe_ratio`）**内嵌进模板**；验证：复制到临时目录（`../..` 指向非 causal-lab）独立跑通 ✓
- [x] D2 **demo_remote_sensing/TEMPLATE_README.md 重写**：不再与 10_templates 逐字相同；讲清遥感因果结构（**X3 土壤湿度是混淆变量**同时影响 X1 与 Y、X2 波段噪声无关）、跑法、预期输出（**SHD=0、X2 正确剔除**，实测 PC/GES 均 adjP/R=1.0）
- [x] D3 **data_gen 死代码清理**：删除无人调用的 `_generate`（仅 task 文档引用），保留 `_parents/_topo_order/_dag_to_adj/_num_nodes`；自测通过
- [x] D4 Pearl 笔记修正：**远程文件，Claude Code 无法操作**（本地无该文件）——已在 OPTIMIZATION-PLAN 标注，由 **Hermes 通过 API 拉取修改后推送**（James Robins = Jamie Robins，合并贡献者表两行）
- [x] D5 **CI 冒烟 `.github/workflows/smoke.yml`**：ubuntu + python 3.9 → `pip install -r requirements.txt` → 跑 00_env_check + 01~07 各 run.py + 09 run_all + **模板自包含验收**（复制 10_templates 到 /tmp 独立跑通）；08_benchmarks 需外部 bnlearn 数据不进 CI
- [x] 顺带工程修复：00_env_check / 03 / 07 的 `✓` 在 GBK 控制台 UnicodeEncodeError（退出码 1）→ 改 ASCII `[OK]`；04 的 PNL 段加 **torch 缺失自动跳过**（causal-learn 不强依赖 torch，requirements 未含 → CI 无 torch 也能跑 04，PNL 降级为跳过）

**关键实测数值**
- 模板独立运行：复制到 `.scratch/tpl_test`（`../..`=非 causal-lab）`template_pipeline.py` 跑通（PC/GES 各 2 边）
- demo 遥感案例（4 变量 n=3000）：PC+fisherz / GES+BIC 均 **SHD=0、adjP/R=1.0/1.0**（X2 无任何边，伪相关剔除）
- CI 冒烟全套本机模拟：00_env / 01 / 02(+cdnod) / 03 / 04 / 05 / 06 / 07 / run_all 全部 exit=0

**注意 / 决策记录**
- **PNL 与 torch**：causal-learn 0.1.4.8 的 PyPI 依赖不含 torch，但 PNL 模块顶层 `import torch`。本地 pytorch env 有 torch（2.4.0），CI 装 requirements 后无 torch → 04/run.py 捕获 ImportError 降级「跳过 PNL」，不阻断 CI；requirements.txt 不补 torch（保持批次 A 的 10 依赖清单，CI 冒烟不需要 PNL）
- 08_benchmarks/smoke_asia.py 依赖外部 bnlearn 数据（D:\win\causal-learn\tests\TestData\），CI 无此数据 → 工作流注释说明、不进 smoke
- 本批次多次本地重跑实验会令 results/metrics|figs 的 evidence 文件出现「仅 timing/metadata 漂移」（SHD/P/R 不变），提交前已 `git checkout` 回滚噪声，保持批次 D commit 只含工程改动

**下一步**：批次 A–D 全部完成（HERMES 推送 + 验收）；剩余可选：CI 实际在 GitHub Actions 上跑通确认、08_benchmarks 全量基准、远程 Pearl 笔记修正。

## 优化批次 E1 — 证据加固（2026-08-25，Claude Code 执行，IMP #1 #4 #8）

**做了什么**
- [x] E1-1 **离散真实 CPD 复核**（#1）：`scripts/data_gen.py` 新增 `simulate_discrete_cpd()`（多项 logit CPD，非高斯分箱，5 节点 7 边 3 状态）；新脚本 `experiments/09_comparison/discrete_cpd.py` 在两种生成口径下对比 BOSS+BIC / GRaSP+BIC / PC+chisq / GES+BDeu 各 5 seed（42/1/7/2024/999）→ `results/metrics/discrete_cpd.json`
- [x] E1-2 **样本量敏感性扫描**（#4）：新脚本 `experiments/09_comparison/sensitivity.py`，样本量梯度 [1000/3000/10000/30000] × 方法 × 3 seed（42/1/7）→ `results/metrics/sensitivity.json` + `results/sensitivity_report.md`
- [x] E1-3 **bnlearn 全量基准**（#8）：新脚本 `experiments/08_benchmarks/run_all_bnlearn.py` 跑 13 数据集 ×（PC+chisq / BOSS+BDeu），子进程 + wall-clock 超时（大图止损），真值 DAG→CPDAG 对齐 → `results/metrics/bnlearn_all.json` + 3 张大图 PNG → `results/figs/`
- [x] 知识库/README/template 同步真实化：knowledge/08 离散行改「三口径合论」+ 新增「样本量分档建议」②b-1 节 + 决策树/坑速查更新；08_benchmarks/README 全量结果表；README 速查表离散行；10_templates/template_data_gen 离散建议

**关键实测数值**

| 实验 | 结论 | 数值 |
|---|---|---|
| 离散真 CPD 复核（logit） | **PC+chisq 反超最优** | PC 3.2±1.0 vs BOSS 5.8±0.7 vs GRaSP 6.0±0.6 vs GES 5.0±1.3 |
| 对照：高斯分箱 | BOSS 最优（旧结论确认） | BOSS 2.4±2.8 vs PC 6.8±1.2 |
| 样本量敏感性 | 大样本≠更稳 | 30000 下全部方法 `[0,7,0]`（2/3 seed 完美、1 seed 全错）；BOSS 最优带 3000–10000（0.0±0.0） |
| 样本量敏感性 | ICA-LiNGAM 全样本量稳定 | 非高斯 1000→30000 一律 0.0±0.0 |
| bnlearn sachs（11 节点） | **BOSS+BDeu 完美恢复** | bOSS 0 vs PC 14🔥 |
| bnlearn 小/中（可比 8 集） | **BOSS+BDeu 全面领先** | 6 赢 1 平 1 输（cancer 0/child 4/water 32 全线压过 PC） |
| bnlearn 大图 | **BOSS+BDeu 不可行 / PC 退化** | alarm(37) BOSS>600s 超时；>40 节点跳过；PC hailfinder SHD=96（adjP/R≈0.18）、water SHD=45 |

**注意 / 决策记录**
- **离散结论随生成口径翻转是本次最大教训**：高斯分箱（近有序）→ BOSS 最优；真实 bnlearn 网络小/中 → BOSS+BDeu 领先；纯随机 logit 小图 → PC+chisq 领先；大图 → 只能 PC（退化）。knowledge/08 改为「必须标注生成结构与规模」。
- **BOSS+BDeu 大图不实用的根因**是 BDeu 打分父配置分组指数膨胀（非 BOSS 算法本身），alarm(37) 实测 >600s 不收敛 → 以 600s 有界尝试为锚点，>40 节点干脆跳过并显式标注（避免每集烧 15 分钟）。
- 大图 PC+chisq 退化：固定 alpha=0.05 + 10000 样本在 50+ 变量上累积误报（hailfinder adjP/R 仅 ~0.18）→ 已列入 08_benchmarks TODO（alpha 自适应治理）。
- 数值交叉验证：asia PC+chisq SHD=5 与既有 08_benchmarks_asia.json（批次 C）一致 ✓；evaluate.py 统一尺子。
- 遗留`results/template_out`漂移已按纪律回滚；E2 将补 CI/测试/文档。

**下一步**：阶段 E2（CI 兼容检查 + 模板回归测试 + 文档补充），完成后 commit 并收尾。

## 优化批次 E2 — CI 兼容 + 工程收尾（2026-08-25，Claude Code 执行，IMP #6 #7 #9 #10）

**做了什么**
- [x] E2-1 **CI 兼容性检查**（#6）：`python -m py_compile` 全部 .py 通过；`PYTHONPATH=`（Linux 无污染等价环境）模拟跑 00_env_check + 01_pc + 07_granger 全部 exit=0。已核查：全部实验脚本 `sys.path` 基于 `__file__` 相对仓库根（无 D:\ 依赖）、输出相对 `results/`、matplotlib 全部 `use("Agg")`（04/06/plotting.py 均在 pyplot 之前设置，无 `show()`）；E1 新增脚本（discrete_cpd/sensitivity/run_all_bnlearn）同样无污染可跑。仅 08_benchmarks 需外部 bnlearn 数据（同 smoke_asia，CI 已排除）
- [x] E2-2 **模板回归测试固化**（#7）：新建 `tests/test_template_portable.py`——复制 10_templates 到独立临时目录 → 子进程跑 template_data_gen.py + template_pipeline.py + 给真值图的定量评估驱动，断言 exit=0、输出含关键标记、metrics.json 合法（无 NaN/Infinity）。本地跑通（5/5 步通过）
- [x] E2-3 **文档补充**（#9 + #10）：README 安装节补「可选装 graphviz（系统级）启用 to_pydot 原生渲染，无则用内置 plotting.py」；knowledge/07 图渲染节同步一句（graphviz 为可选依赖）；确认 04 run.py 的 PNL import 已在 try 块内（惰性）并补注释说明其动机（批量不背 45s 启动、torch 缺失自动降级）

**关键实测数值**
- py_compile：全仓库 .py 语法 OK（0 失败）
- CI 模拟（无 PYTHONPATH）：00_env_check exit=0 / 01_pc exit=0 / 07_granger exit=0
- `tests/test_template_portable.py`：复制即用回归 5/5 步通过，模板默认路径 + 定量评估路径均验证

**注意 / 决策记录**
- E2-1 实测会重写 `01_pc.json` 等 evidence 文件（仅 timing 漂移，SHD/P/R 不变）——按纪律已回滚，E2 commit 只含工程改动
- CI 工作流（smoke.yml）**未改动**：新增 E1 实验脚本与本机 git 非必须入 CI（discrete_cpd/sensitivity 确定性可复现但 wall-time ~1-2min，bnlearn 需外部数据）；test_template_portable.py 作为本地回归防护。GitHub Actions 真跑由 Hermes 推送后跟进
- 可选后续：把 CI 内联「template standalone」步骤换成 `python tests/test_template_portable.py`（覆盖更强）；大图 PC+chisq alpha 自适应（已列 08_benchmarks TODO）

**下一步**：E1+E2 全部完成；工作树已净。Hermes 推送 + 验收。

## 优化轮次 F — CI 升级 + 大图 alpha 自适应 / KCI / 马尔可夫毯 / CAM-UV（2026-08-25，Claude Code 执行）

### F 阶段第一部分（F1+F2 已提交，IMP 后续项 #2 #3 #11 见下半节）

**做了什么**
- [x] F1 **CI 工作流升级**（E2 遗留）：`.github/workflows/smoke.yml` ① 内联「模板独立目录」验收整段替换为 `python tests/test_template_portable.py`（E2 回归测试，覆盖更强）；② 在 09_comparison 后新增 `discrete_cpd.py` + `sensitivity.py` 两步（E1 确定性脚本入 CI）；timeout-minutes 20→35。本地模拟：回归测试 + discrete_cpd 冒烟均 exit=0（PYTHONPATH= 等价 CI）
- [x] F2 **大图 PC alpha 自适应**（E1 遗留 hailfinder SHD=96）：新 `experiments/08_benchmarks/alpha_adaptive.py`——对 PC+chisq 做 **Bonferroni**（alpha/C(n,2)，说明选 BH-FDR 不可行的理由：causal-learn PC 不暴露内部 p 值集合）＋ α=0.01 探针；在 hailfinder/hepar2/win95pts 三图对比 fixed vs mid vs bonf → `results/metrics/alpha_adaptive.json`

**关键实测数值（alpha_adaptive.json）**

| 图 | 节点/边 | fixed 0.05 | mid 0.01 | Bonferroni |
|---|---|---|---|---|
| hailfinder | 56/66 | SHD=96, adjP/R 0.18/0.11 | 93 | **89**, adjP 0.21（召回仍 0.106）|
| hepar2 | 70/123 | SHD=92, adjP/R 0.93/0.57 | 96 | 97, adjP 1.0/0.46 |
| win95pts | 76/112 | SHD=57, adjP/R 0.96/0.68 | **53** | 70 |

**注意 / 决策记录**
- **α 自适应是限定性结论**：只在「精度主导」的图上有效（hailfinder 单调改善但召回卡 0.106——图的问题在数据结构非 alpha）；hepar2/win95pts 上 Bonferroni 过度保守反而更糟。**推荐小步收紧 α≈0.01 作起点，不做全量 Bonferroni**——已写入 knowledge/08 ④ 坑表 + 08_benchmarks/README 注记。
- F1 本地模拟重跑 discrete_cpd 产生的 JSON timing 漂移已按纪律回滚，commit 只含工程改动。

**下一步**：F3 KCI 大样本脚本 + F4 马尔可夫毯管道 + F5 CAM-UV 检查，完成后第二个 commit 并收尾。

### F 阶段第二部分（F3+F4+F5 已提交）

**做了什么**
- [x] F3 **KCI 大样本脚本**（IMP #2）：新 `experiments/09_comparison/kci_large.py`——`--samples` 参数 + 按样本量**累积合并**入库（Hermes 后续扩跑 5000 可增量），cache 文件名内嵌样本量；本地跑完 1500 + 3000 全量（共 4 格）→ `results/metrics/kci_large.json`
- [x] F4 **遥感马尔可夫毯因果特征选择管道**（IMP #11 基础版）：新 `experiments/demo_remote_sensing/markov_blanket.py`——X1-X5+Y 合成数据（X3 混淆、X2 纯噪声、X4→X5 派生链）→ PC+fisherz 找 CPDAG → 提取 Y 马尔可夫毯 → **全特征 / 相关筛选 / 因果毯** 三组逻辑回归 5 折 CV 对比（3 seed）；`scripts/plotting.py` 增 `highlight=` 可选参数（兼容）标出 MB 节点；结论写入 TEMPLATE_README.md
- [x] F5 **CAM-UV 检查**（IMP #3）：`import pygam` 探针确认未装 → knowledge/04 两处标注「需 pygam（等待批准安装，F 轮未装）」，不安装

**关键实测数值**

| 实验 | 结论 | 数值 |
|---|---|---|
| KCI 大样本（F3） | **大样本非解药，SHD 非单调** | 非线性 1200/1500/3000 → 7/5/6；线性 6/5/5；3000 时 ~18min/数据集 |
| KCI 召回 | 召回 0.71→0.86 有提升但卡「漏 1 边」 | adjR 3000 两数据集均 0.857；非线性还出假阳性（adjP 0.857） |
| 马尔可夫毯（F4） | PC 3 seed 全 SHD=0，MB 恒={X1,X3} | 相关筛选被骗入 {X1,X3,X4,X5}（X4/X5 仅共因相关） |
| 三类特征分类 | **2 特征毯 = 5 全特征准确率** | 均 0.935±0.003（MB 只用 2/5 特征） |
| CAM-UV | pygam 未装 → 标注待批准 | `import pygam` ModuleNotFoundError |

**注意 / 决策记录**
- **knowledge/08 非线性结论修正**（重要）：旧建议「大样本（≥3000）用 KCI」被 F3 推翻——3000 样本 KCI 在 5 节点图上 SHD 仍 ≥5，且非线性出现假阳性、耗时爆炸。改为「PC 骨架 + ANM/PNL 定向为首选；KCI 仅作骨架交叉验证」。决策树非线性分支、④ 坑表 KCI 行同步更新。
- **F4 的设计选择**：图学习用连续 Y 得分（保证 fisherz 高斯前提），分类用二值标签（真实任务形态）；马尔可夫毯提取实现「父+子+配偶」并从真值对照验证（本 demo Y 是无子叶子对撞点，MB=邻接精确）。X4/X5 的关键是「**经 X3 共因相关、非因果相关**」——相关筛选被骗、毯正确剔除，这正是因果特征选择的论文价值点。
- **范围确认**：IMPROVEMENTS #5（推送脚本 gitignore 语义）为 Hermes 侧工具，F 轮不做，标留给 Hermes。
- KCI 3000 产生的 cache_kci_large_*.json 已被 gitignore（cache_* 规则）覆盖，不入库。

**下一步**：F 轮全部完成；两个 commit（F1-2 / F3-4-5）。Hermes 推送 + 验收；后续可选：pygam 批准后补 CAM-UV、KCI 5000（Hermes 后台 `kci_large.py --samples 5000` 增量入库）、真实遥感数据应用。
