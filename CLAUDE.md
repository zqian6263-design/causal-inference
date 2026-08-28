# 项目：causal-lab — causal-learn 系统学习与方法选型

本目录是因果推断方法学习项目。**你的总任务是 `PLAN.md`**（Phase 0–4：环境→知识库→逐方法实验→横向对比→模板）。先读 PLAN.md，再动手。

## 环境硬约束（违反会浪费时间）

> 本节是**本机开发环境**说明，仅对当前这台机器生效；其他机器 `pip install -r requirements.txt` 后直接 `python` 运行即可（依赖版本见 `requirements.txt`）。

- **本机 Python**：`D:/Anaconda/envs/pytorch/python.exe`（Anaconda pytorch env，依赖齐全）
- **所有 Python 命令前缀 `PYTHONPATH=`**（本机终端 PYTHONPATH 指向 Hermes venv，会劫持 import，已实测 networkx 加载错误）。本机正确写法：`PYTHONPATH= D:/Anaconda/envs/pytorch/python.exe xxx.py`
- causal-learn **0.1.4.8（pip 版）** 已装入 pytorch env，实验一律用它
- 官方源码克隆（仓库同级目录 `causal-learn/`，可选）：**仅供阅读 API 参考，禁止在它目录内运行任何实验脚本**（cwd 会污染 import 到源码版）；一切实验在本仓库根目录下进行
- 本任务纯 CPU，不涉及 GPU；**禁止安装任何新组件**（依赖已全部就绪：numpy/scipy/sklearn/statsmodels/networkx/pandas/pydot/joblib/matplotlib/torch）
- 本机路径一律在本仓库根目录下（D 盘根目录无写权限）

## 目录结构

- `knowledge/` — 中文知识库 9 篇（00 基础 / 01 全景 / 02 约束型 / 03 打分型 / 04 函数模型 / 05 隐变量与排列 / 06 时序 / 07 评估工程 / 08 ★选型指南）
- `experiments/` — 逐方法实验全部完成：00 环境验证 / 01 PC / 02 FCI+CD-NOD / 03 GES/DGES/ExactSearch / 04 LiNGAM 家族+ANM+PNL+RCD+CAM-UV / 05 GIN+RLCD+GRaSP+BOSS / 06 GRaSP/BOSS / 07 Granger / 08 bnlearn 13 数据集基准+官方 TestPC 对照 / 09 横向对比与专项（run_all/discrete_cpd/sensitivity/KCI）/ 10_templates 通用模板 / 11_doc_coverage 文档覆盖（gsq/gcv-gml/图工具/load_dataset）
- `scripts/` — 公共工具（data_gen.py 数据生成器、evaluate.py 评估器）
- `results/` — 输出（metrics/ 指标 JSON、figs/ 图表、报告 md）

## 文档与代码来源

- 官方文档源文件：`causal-learn/docs/source/**/*.rst`（45 个，readthedocs 的原始素材；`causal-learn/` 为仓库同级目录的官方源码克隆，可选）
- 官方测试示例：`causal-learn/tests/Test*.py`（TestPC/TestGES/TestFCI/TestGIN/TestRLCD/TestGRaSP/TestBOSS/TestGranger… 是最佳学习用例）
- 方法实现：`causal-learn/causallearn/search/**`

## 工作纪律

- 每个 Phase 完成：git commit + `REPORT.md` 追加一节（做了什么 / 关键数值 / 下一步）
- 知识库文档一律简体中文，用户偏好：简单易懂、表格/要点、真实代码示例，不堆术语
- 实验：seed=42 固定，禁止调 seed 刷指标；数据生成用 `scripts/data_gen.py`；评估用 `scripts/evaluate.py`（SHD + Adjacency/Arrow precision-recall）
- 指标 JSON 落盘 `results/metrics/`，图表 `results/figs/`
- 运行 >5 分钟的实验：后台跑 + 记录时间戳
- 遇到不确定的设计决策：停下来在报告里说明，不要自作主张改 PLAN 范围

## 报告格式

每个 Phase 结束给一屏摘要（中文）：做了什么 / 关键数值表 / 卡在哪 / 下一步。
