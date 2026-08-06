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
