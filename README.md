# causal-lab：因果推断系统学习与方法选型项目

> 系统梳理 causal-learn（CMU CLeaR / py-why）全部文档与代码，建立**可复用的因果推断知识库 + 方法选型能力**，为后续一切因果推断数据分析任务提供「直接选方法、设计方案」的决策支撑。

**总任务书见 `PLAN.md`**（顶层设计 + Phase 0–4 分阶段计划 + 验收标准）。本文件为项目入口与定位。

## 快速导航

| 想干什么 | 看哪里 |
|---|---|
| 总任务/分阶段计划/验收 | `PLAN.md` |
| 给 Claude Code 的执行上下文 | `CLAUDE.md` |
| 因果推断知识库（中文） | `knowledge/00–08` |
| 逐方法实验代码 | `experiments/01_pc/ … 07_granger/` |
| 横向对比实验 | `experiments/09_comparison/` |
| 通用实验模板（未来任务直接套用） | `experiments/10_templates/` |
| 方法选型决策树（★核心交付物） | `knowledge/08-方法选型指南.md` |
| 全部输出（图/表/指标） | `results/` |
| causal-learn 官方源码（阅读参考） | `D:\win\causal-learn`（禁止在其目录内跑实验脚本） |

## 环境一句话

Python = `D:\Anaconda\envs\pytorch\python.exe`（causal-learn 0.1.4.8 已装）；**所有命令前缀 `PYTHONPATH=`**（防 Hermes venv 劫持 import）。详见 `CLAUDE.md`。
