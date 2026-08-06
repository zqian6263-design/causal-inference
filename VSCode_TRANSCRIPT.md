# VSCode Claude Code 转述文本（VSCode_TRANSCRIPT.md）

> 用法：在 VSCode 打开 `D:\win\causal-lab`，把下方代码块内容整体粘贴给 Claude Code（或作为首个提示）。执行批次 B，完成后告诉 Hermes 验收。

---

```
工作目录：D:\win\causal-lab

背景：causal-learn 因果推断学习项目。先读三个文件：
1. README.md —— 项目定位
2. OPTIMIZATION-PLAN.md —— 优化任务书（共 A/B/C/D 四批；批次 A 已完成并推送，你只做批次 B）
3. CLAUDE.md —— 环境约束（必读）

【环境要点（违反会浪费时间）】
- Python 必须用绝对路径：D:/Anaconda/envs/pytorch/python.exe（causal-learn 0.1.4.8 已装）
- 所有 python 命令前缀 PYTHONPATH=（本机终端 PYTHONPATH 指向 Hermes venv，会劫持 import）
- 不要进 D:\win\causal-learn（官方源码克隆）目录跑实验；一切在 D:\win\causal-lab 下进行
- 本任务纯 CPU，禁止安装任何新组件（依赖已齐）

【你的任务】OPTIMIZATION-PLAN.md 的【批次 B】——修实证证据矛盾：
- B1 改造 experiments/09_comparison/run_all.py：
  ① 每个方法跑 5 个 seed（42/1/7/2024/999），SHD 报 mean±std
  ② GRaSP/BOSS 显式对比 score_func='local_score_BIC_from_cov' vs 'local_score_BDeu'
    （离散数据两组都跑，修正"BOSS 碾压归因 BDeu"的错误结论——默认是 BIC_from_cov）
  ③ ArrowConfusion 无边时 NaN → precision 置 0（根因：run_all 未处理）
  ④ ICA-LiNGAM 纳入统一矩阵：adjacency_matrix_ 阈值 0.1 → 邻接掩码 → evaluate_graph 算 SHD
    （与其他方法同一把尺子，不再单独走"因果序+边数"）
  ⑤ 输出合法 JSON（无 NaN 字面量）+ 更新 results/comparison_report.md
- B2 用新数据重写 knowledge/08-方法选型指南.md 的「②b 实证对比矩阵」：
  每格 mean±std；修正 GRaSP 离散 SHD 矛盾（原文档 8，实际 JSON 曾为 1，多 seed 后取真实值）；
  修正 BOSS 归因；补 LiNGAM 的 SHD 行；结论改"多 seed 平均"口径
- 复用 scripts/data_gen.py（数据生成）和 scripts/evaluate.py（评估），不要另写

【纪律】
- 实验必须实际运行验证，指标落盘 results/metrics/comparison.json（合法 JSON）
- 多 seed 用固定集合，禁止调 seed 刷指标
- 完成后 git add -A + commit（信息：'opt(B): 修实证证据矛盾 - 多seed重跑/NaN/LiNGAM统一评估/08对齐'，git 身份 Hermes）
- REPORT.md 追加「优化批次 B」一节
- 时间预算：完成即收尾，不要扩展范围
```

---

## 后续批次（B 完成后执行批次 C，任务书同源）

批次 C = 补实验缺口：01_pc/run.py、02_fci/run.py（FCI+CD-NOD）、06_grasp_boss/run.py、GIN、VAR-LiNGAM、PNL、RCD/CAM-UV 标注、results/figs/*.png 图表落盘。
批次 D = 模板自包含（scripts 内嵌）、demo README 重写、data_gen 死代码清理、CI 冒烟（可选）。

批次 C/D 的详细要求见 OPTIMIZATION-PLAN.md 对应章节，直接让 Claude Code 读任务书执行即可。
