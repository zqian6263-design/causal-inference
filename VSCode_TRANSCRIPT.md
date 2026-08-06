# VSCode Claude Code 转述文本（VSCode_TRANSCRIPT.md）

> 用法：在 VSCode 打开 `D:\win\causal-lab`，把对应批次的代码块整体粘贴给 Claude Code。当前待执行：**批次 C**。每批完成后 git commit，告诉 Hermes 验收。

---

## 📌 当前批次 C — 补实验缺口

```
工作目录：D:\win\causal-lab

背景：causal-learn 因果推断学习项目。先读三个文件：
1. README.md —— 项目定位
2. OPTIMIZATION-PLAN.md —— 优化任务书（批次 A 已完成推送、批次 B 已完成并验收，你只做批次 C）
3. CLAUDE.md —— 环境约束（必读）
4. knowledge/08-方法选型指南.md 的「②b 实证矩阵」——参考已定稿的多 seed 口径

【环境要点（违反会浪费时间）】
- Python 必须用绝对路径：D:/Anaconda/envs/pytorch/python.exe（causal-learn 0.1.4.8 已装）
- 所有 python 命令前缀 PYTHONPATH=（本机终端 PYTHONPATH 指向 Hermes venv，会劫持 import）
- 不要进 D:\win\causal-learn（官方源码克隆）目录跑实验；一切在 D:\win\causal-lab 下进行
- 本任务纯 CPU，禁止安装任何新组件（依赖已齐）
- 复用 scripts/data_gen.py（数据生成）、scripts/evaluate.py（评估，已有 _safe_ratio NaN 修复）——不要另写

【你的任务】OPTIMIZATION-PLAN.md 的【批次 C】——补实验缺口，让 PLAN 验收标准「13 个方法各有可复现实验」成立：

C1 补齐方法实验（每个方法一个 run.py + 指标 + 图）：
- experiments/01_pc/run.py：PC 专属实验（5 节点基准图，PC+fisherz；可加 PC+kci 小样本对比；输出指标 + 图）
- experiments/02_fci/run.py：FCI 实验——合成含隐变量的数据（1 个隐变量混淆两个观测变量），输出 PAG；评估注意：FCI 输出 PAG，真值要用 dag2pag 对齐（causallearn.utils.DAG2PAG.dag2pag），不要用 dag2cpdag（语义错位会虚高）；无直接 SHD 时说明 PAG 语义并给 Adjacency P/R
- experiments/02_fci/ 内追加 cdnod 示例（CD-NOD 需要 c_indx：构造机制分段切换的时变数据，参考知识库 02 的示例）
- experiments/06_grasp_boss/run.py：GRaSP/BOSS 专属实验（线性高斯 + 线性非高斯各一组，沿用 5 seed mean±std 纪律，random.seed(seed) 固定；展示两方法在多 seed 下的稳定性对比）
- experiments/04_lingam_anm_pnl/run.py 扩展：加 VAR-LiNGAM（构造 2 变量滞后因果时序数据，lags=2）；PNL 成对方向判断（PNL import 约 45s，样本 n=400 控制时间，标注运行耗时）
- experiments/05_gin_rlcd/run.py 扩展：加 GIN（LiNLAM 合成数据：隐变量 L → 多个观测 + 非高斯噪声，GIN(data) 返回 (G, K)；输出图 + 因果序）
- RCD / CAM-UV：在 04 文档对应小节标注「官方实现，未单独实验」（如时间充裕可加最小调用，否则如实标注）

C2 图表落盘：上述每个 run.py 输出一张图到 results/figs/*.png（GraphUtils.to_pydot(...).write_png(...) 或 plot_graph），落实 PLAN「图表落盘 results/figs/」验收项

C3 experiments/08_benchmarks/：放 README.md 说明 bnlearn 基准数据集的获取方式（D:\win\causal-learn\tests\TestData\bnlearn_discrete_10000\ 或官方链接）+ 一个冒烟脚本（跑 asia 数据 PC+chisq 即可）；如时间紧标注 TODO

【纪律】
- 每个方法实验必须实际运行验证（输出指标 JSON 落盘 results/metrics/ 或 run.py 同目录，图落 results/figs/）
- 沿用批次 B 确立的多 seed 纪律（GRaSP/BOSS 必须 random.seed(seed) 固定 + 5 seed mean±std；PC/GES/LiNGAM 无内部随机可单 seed）
- 完成后 git add -A + commit（信息：'opt(C): 补实验缺口 - 01_pc/02_fci/06_grasp_boss/GIN/VAR-LiNGAM/PNL/figs'，git 身份 Hermes）
- REPORT.md 追加「优化批次 C」一节（每个新方法的实测数字）
- 时间预算：C1 优先（核心），C2 每脚本一张图，C3 可最小化；完成即收尾，不要扩展范围
```

---

## 批次 D（C 完成后执行，预告）

批次 D = 收尾工程：
1. **模板自包含**：`experiments/10_templates/` 去掉对仓库根的 `sys.path.insert(..., "..", "..")` 依赖（把 evaluate 逻辑内嵌或模板目录内置 scripts 副本），验证复制到临时目录能独立跑通
2. **demo_remote_sensing 真实演示说明**：重写其 TEMPLATE_README.md（讲清遥感案例因果结构：X3 土壤湿度为混淆变量、X2 无关特征被剔除、跑法、预期 SHD=0）——当前与 10_templates 的 README 逐字相同，是误导
3. **data_gen 死代码清理**：删除无人调用的 `_generate` 函数（保留 `_parents/_topo_order/_dag_to_adj/_num_nodes`，simulate_nonlinear_anm 在用）
4. **CI 冒烟（可选）**：`.github/workflows/smoke.yml`（ubuntu + python 3.9 → pip install -r requirements.txt → 跑 00_env_check.py + 各 run.py + run_all.py）

批次 D 详细要求见 `OPTIMIZATION-PLAN.md` 对应章节，让 Claude Code 读任务书执行。
