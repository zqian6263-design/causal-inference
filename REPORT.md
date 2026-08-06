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
