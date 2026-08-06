# causal-lab 优化任务书（OPTIMIZATION-PLAN.md）

> 依据 Claude Code 审查反馈（2026-08-06），全部指控已经 Hermes 独立核验确认。按批次执行，每批完成后 git commit，Hermes 负责验收 + GitHub 推送。

## 批次 A — 可复现性与仓库卫生（最高优先）

### A1. requirements.txt
项目根新建 `requirements.txt`，锁定可复现版本（对照本机 pytorch env 实测版本）：
```
causal-learn==0.1.4.8
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.1
statsmodels==0.14.6
networkx==3.2.1
pandas==2.3.3
pydot==4.0.1
joblib==1.5.2
matplotlib==3.9.2
```

### A2. 去机器专属路径
文档里所有 `D:/Anaconda/envs/pytorch/python.exe`、`D:\win\causal-learn`、`D:\win\causal-lab` 绝对路径改为：
- 运行命令统一写 `python` + `pip install -r requirements.txt`（README 加「安装」一节说明）
- causal-learn 源码路径改相对描述：「仓库同级目录 causal-learn/ 或 pip 安装」
- 保留一处「本机开发环境」说明（写清本机用的什么，但标注其他机器用 requirements.txt 即可）

涉及文件：README.md、GUIDE.md、CLAUDE.md、knowledge/00-08 的「参考/验证输出」小节、experiments/*/run.py 顶部注释、scripts/*.py docstring。

### A3. 仓库卫生
- `.gitignore` 追加：`.claude/settings.local.json`、`results/metrics/cache_*.json`（CI 缓存不进 VCS；comparison.json 和 kci_supplement.json 是证据，保留）
- `git rm --cached .claude/settings.local.json`（保留本地文件）
- 根目录加 `LICENSE`：README 采用 CC-BY-4.0 声明，代码采用 MIT（两个段落写在一个 LICENSE 文件或分开，参考 GitHub 惯例）
- 空目录 `experiments/01_pc/ 02_fci/ 06_grasp_boss/ 08_benchmarks/` 各放一个 `.gitkeep`（目录结构成立，内容由批次 C 补齐）

### A4. 文档结构声明对齐
README/PLAN/GUIDE 中引用但尚未实现的目录（01_pc、02_fci、06_*、08_benchmarks）标注「建设中（批次 C）」或调整表述，避免文档与仓库不符。

## 批次 B — 修实证证据矛盾（最伤信任，必须立刻修）

### B1. 多 seed 重跑对比实验
改造 `experiments/09_comparison/run_all.py`：
- 每个方法跑 5 个 seed（42/1/7/2024/999），报 SHD 的 mean±std
- **GRaSP/BOSS 显式传参对比**：`score_func='local_score_BIC_from_cov'`（默认）vs `'local_score_BDeu'`（离散数据），修正「BOSS 碾压归因 BDeu」的错误结论
- 修 ArrowConfusion 的 NaN（无边时 precision 置 0 而非 NaN）
- ICA-LiNGAM 纳入统一矩阵：`adjacency_matrix_` 阈值 0.1 → 邻接掩码 → evaluate_graph 算 SHD（与其他方法同一把尺子）
- 输出 `results/metrics/comparison.json`（**合法 JSON，无 NaN 字面量**）+ 更新 `results/comparison_report.md`

### B2. 对齐 knowledge/08
用 B1 的新数据重写 08 选型指南的「②b 实证对比矩阵」：每格 mean±std、修正 GRaSP 离散 SHD 矛盾（原来文档 8 vs JSON 1）、修正 BOSS 归因、补 LiNGAM 的 SHD 行。结论表述改为「多 seed 平均」口径。

## 批次 C — 补实验缺口（对齐 PLAN 验收标准）

### C1. 补齐方法实验（每方法一个 run.py + 指标 + 图）
- `experiments/01_pc/run.py`：PC+fisherz/kci 专属实验（5 节点基准 + 指标）
- `experiments/02_fci/run.py`：FCI（含隐变量合成数据，输出 PAG + SHD 对齐说明 PAG 语义不直接算 SHD——用 dag2pag 真值对齐）
- `experiments/02_fci/` 或单独：CD-NOD（c_indx 时变数据）
- `experiments/06_grasp_boss/run.py`：GRaSP/BOSS 专属（从 05 拆出，含多 seed 稳定性展示）
- `experiments/04_lingam_anm_pnl/run.py` 扩展：VAR-LiNGAM（时序数据）
- `experiments/05_gin_rlcd/run.py` 扩展：GIN（LiNLAM 合成数据）
- PNL 在 04 补一个成对方向示例（PNL import 慢，标注运行时间）
- RCD/CAM-UV：文档标注「官方实现，未单独实验」（或补最小调用）

### C2. 图表落盘
每个 run.py 输出一张图到 `results/figs/*.png`（GraphUtils.to_pydot 或 plot_graph），落实 PLAN 的「图表落盘 results/figs/」验收项。

### C3. 08_benchmarks
`experiments/08_benchmarks/` 放 bnlearn 冒烟（可选：若时间紧标注 TODO + README 说明）。

## 批次 D — 模板自包含与杂项

### D1. 模板复制即用
`experiments/10_templates/` 改为**完全自包含**：
- 把 `scripts/evaluate.py` 的 evaluate_graph 逻辑内嵌进模板（或模板目录内放一份 scripts 副本 + 相对导入）
- 去掉 `sys.path.insert(..., "..", "..")` 依赖，改为基于 `__file__` 的纯相对结构
- 验证：复制模板到临时目录能独立跑通（这是验收标准）

### D2. demo_remote_sensing 真实演示说明
重写 demo 的 TEMPLATE_README.md：讲清遥感案例的因果结构（X3 混淆）、跑法、预期输出（SHD=0、X2 被剔除）；不要与 10_templates 的 README 相同。

### D3. data_gen 死代码清理
`scripts/data_gen.py` 删除 `_generate`（无人调用），保留 `_parents/_topo_order/_dag_to_adj/_num_nodes`（simulate_nonlinear_anm 在用），整理 docstring。

### D4. Pearl 笔记修正
远程文件 `因果推断入门-JudeaPearl演讲精华.md`：James Robins / Jamie Robins 是同一人（James M. Robins，Jamie 为昵称），合并贡献者表两行。此文件在远程仓库（本地无），由 Hermes 通过 API 拉取修改后推送，Claude Code 只需在计划中标注。

### D5（可选）. CI 冒烟
`.github/workflows/smoke.yml`：ubuntu + python 3.9 → pip install -r requirements.txt → 跑 00_env_check.py + 各 run.py + run_all.py。若时间紧标注 TODO。

## 执行纪律
- 每批一个 git commit（信息含批次号），不要混批次
- 实验类改动必须实际运行验证（本机 pytorch env，前缀 PYTHONPATH=）
- 文档改动与实验数据保持一致（B 批完成后 C 批引用新数据）
- 遇到不确定的（如 LICENSE 具体格式）：按反馈建议执行，不做超范围设计
