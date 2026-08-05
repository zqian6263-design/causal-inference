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
