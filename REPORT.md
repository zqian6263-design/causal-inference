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
