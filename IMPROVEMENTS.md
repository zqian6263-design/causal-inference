# 后续改进方向（IMPROVEMENTS.md）

> 来源：causal-lab 优化 A/B/C 批次执行与 Hermes 验收过程中暴露的问题梳理（2026-08-06）。分「本轮已闭环」与「后续待办」两部分。后续待办按优先级排列，可在批次 D 完成后陆续执行。

## 一、本轮已闭环的问题（A/B/C 批次修复）

| # | 问题 | 修复方式 | 批次 |
|---|---|---|---|
| 1 | 无依赖清单、机器路径写满文档 | requirements.txt + 路径清理 + README 安装节 | A |
| 2 | .claude/settings.local.json 误提交 | gitignore + untrack | A |
| 3 | 无 LICENSE | CC-BY-4.0（文档）+ MIT（代码） | A |
| 4 | metrics 缓存/证据文件 gitignore 混乱 | 收窄为只忽略 cache_*，证据入库 | A |
| 5 | GRaSP/BOSS 无 seed 参数、单 seed 结论不可信 | random.seed 全局固定 + **5 seed mean±std 纪律** | B |
| 6 | 「PC/GES 线性高斯完美」是 seed=42 运气 | 多 seed 重跑，修正为仅 BOSS 稳定完美 | B |
| 7 | 「BOSS 碾压归因 BDeu」归因错误 | 显式传参对比，实锤默认 BIC_from_cov 且 BDeu 退化 | B |
| 8 | GRaSP 离散「8 vs 1」文档矛盾 | 多 seed 真实值 6.0±3.0 对齐 | B |
| 9 | comparison.json 含 NaN 字面量（非法 JSON） | evaluate.py `_safe_ratio` 统一置 0 + allow_nan=False 兜底 | B |
| 10 | LiNGAM 未进统一评估矩阵（尺子不一致） | 系数阈值→邻接掩码→CPDAG 对齐→evaluate_graph | B |
| 11 | LiNGAM adjacency_matrix_ 转置约定坑 | `.T` 处理 + 实测校验 mask.T==真值 | B |
| 12 | FCI 评估用 dag2cpdag 语义错位 | 改用 **dag2pag**（PAG 端点口径） | C |
| 13 | graphviz dot 二进制缺失导致图渲染全挂 | 新增 scripts/plotting.py（matplotlib+networkx 直渲），修正 knowledge/07 旧说法 | C |
| 14 | get_endpoint(a,b) 返回 b 端端点（绘图方向反） | 修正 + 断言验证 | C |
| 15 | GIN/PNL/VAR-LiNGAM/FCI/CD-NOD/PC 无专属实验 | 批次 C 补齐 11 方法实验 + 10 图落盘 results/figs/ | C |
| 16 | 空目录与文档引用不符 | .gitkeep + 批次 C 内容补齐 | A/C |

## 二、后续改进方向（按优先级）

### 🔴 高优先级（方法学可信度）

✅ **已完成**（E 轮 `09_comparison/discrete_cpd.py`：高斯分箱 vs 真实 logit CPD vs bnlearn 三口径合论，结论随生成口径翻转并已写入 08）。1. **离散数据验证根基偏脆**：`simulate_discrete` 是连续高斯分位数分箱，不是真实离散 CPD。基于它得出的「BOSS 离散最佳、chisq 弱」结论需要**用真实离散生成模型复核**（如 bnlearn 数据集的 asia/cancer/sachs，或泊松/多项 logit 结构）。已有 asia 冒烟（SHD=5），可扩展为离散专项对比。
✅ **已完成**（F/G 轮 `kci_large.py`：1500/3000/5000 累积，08 结论④——≤3000 勿用、5000 线性可完美/非线性漏 1 边）。2. **KCI 大样本补充实验**：当前结论「KCI 小样本检验力不足」基于 1200 样本（75-105s/数据集）。需在 **≥5000 样本**下复测非线性数据的 KCI（预计 10-30 分钟/数据集）——可挂 A10 或后台跑。若 KCI 大样本恢复良好，08 选型指南的非线性建议可升级。
✅ **已完成**（G 轮：pygam 已批准安装，04 run.py 示例 6——父节点 4/5 + UCP/UBP 报告）。3. **CAM-UV 未运行**：依赖 pygam 未装（遵守禁止装组件规则）。后续若用户批准 `pip install pygam`（纯 python 小包），补 CAM-UV 最小实验。
✅ **已完成**（E 轮 `sensitivity.py`：4 梯度×5 方法×3 seed，08 ②b-1 分档建议）。4. **样本量敏感性分析**：批次 B 暴露 3000 样本下 PC/GES 线性高斯不稳（4.8±2.5）。建议实验协议加入样本量梯度（1000/3000/10000/30000）的稳定性扫描，选型结论按样本量分档。

### 🟡 中优先级（工程与可复现性）

5. **推送脚本内置 gitignore 语义**：~~Git Data API 推送脚本是手动 walk 文件系统，曾把 cache_*.json 误推远程~~ —— **已完成（scripts/github_push.py）**：`git check-ignore --stdin` 过滤 + 硬跳过兜底（.git/.scratch/__pycache__/.claude/cache_*）+ 推送后自验证；已实测两次推送（含 dogfood 自推）。
✅ **已完成**（D/F 轮 `smoke.yml` 升级，CI run #10 全绿；含 E 轮新脚本接入与模板回归测试步骤）。6. **CI 冒烟（批次 D 可选，强烈建议做）**：`.github/workflows/smoke.yml`（ubuntu + py3.9 + pip install -r requirements.txt → import 检查 → 快方法 run.py 冒烟 → run_all.py 定时任务）。这是公开仓库可复现性的最终保险——批次 D 若未做，后续补。
✅ **已完成**（E 轮 `tests/test_template_portable.py`，5/5 步通过）。7. **模板复制即用的回归测试**：批次 D 改完模板后，把「复制到临时目录独立跑通」固化为一个测试脚本（`tests/test_template_portable.py`），防止未来改动再破坏。
✅ **已完成**（E 轮 `run_all_bnlearn.py` 13 数据集全量 + I 轮 `compare_official.py` 官方 TestPC 对照 13/13 逐位一致）。8. **bnlearn 全量基准（08_benchmarks 扩展）**：当前仅 asia 冒烟。13 个 bnlearn 数据集（alarm→win95pts）全量跑 PC+chisq 可给「离散大图」场景提供实证（时间约 10-30 分钟，可后台）。
   > 已知约束（批次 D 记录）：`smoke_asia.py` 依赖**仓库外**的 bnlearn 数据文件（`D:\win\causal-learn\tests\TestData\bnlearn_discrete_10000\` 或官方仓库下载），CI 无法访问 → 已从 CI smoke 排除并在其 README 注释说明获取方式。后续做全量基准时需先下载数据到本地。

### 🟢 低优先级（体验与生态）

✅ **已完成**（E 轮：README/knowledge07 补 graphviz 可选说明）。9. **图渲染依赖说明**：knowledge/07 已修正 graphviz 说法，但 README 安装节可补充「可选装 graphviz 以启用 to_pydot 原生渲染；无则用 plotting.py」。
✅ **已完成**（E 轮：04 run.py PNL 惰性导入 + 注释）。10. **PNL import 慢（45s）的工程化**：批量任务中 PNL 只 import 一次复用；文档已标注，可进一步提供 `--lazy-import` 模式。
✅ **已完成基础版**（F 轮 `markov_blanket.py`：MB 恒={X1,X3}、2 特征=5 特征准确率；真实遥感落地留论文阶段 research-repro）。11. **研究方向的因果特征选择案例**（面向 TGRS 论文）：demo_remote_sensing 已演示混淆剔除，后续可扩展为「特征+标签联合变量 → 马尔可夫毯 → 因果特征子集」完整管道，作为论文方法章节的实验模板。

## 三、方法论纪律沉淀（供后续所有实验遵守）

1. **多 seed 是硬纪律**：凡内部有随机性的方法（GRaSP/BOSS 等），必须 `random.seed(seed)` 固定 + 5 seed mean±std；单 seed 对比结论一律不可信。
2. **评估语义对齐**：CPDAG 用 dag2cpdag、PAG 用 dag2pag、DAG 用邻接掩码——尺子不统一等于没比较。
3. **数值交叉验证**：新实验的关键数字必须能与既有 results/ 交叉对上（本流程 06 vs comparison.json 一致才算过）。
4. **生成器先对齐官方**：数据生成器写完后，先用官方基准（TestPC 数据/benchmark）验证一致性，再用于实验。
5. **JSON 落盘规范**：metrics 一律合法 JSON（无 NaN/Infinity），`allow_nan=False` 兜底；缓存与证据分目录管理。
6. **evidence 文件与 commit 分离**（批次 D 经验）：本地重跑实验产生的指标 JSON 若**仅 timing/metadata 漂移**（SHD/P/R 数值不变），commit 前回滚，保持每次 commit 只含该批次的真实改动；只有 SHD/P/R 等数值发生变化才值得作为 evidence 提交。避免"重跑一次就污染一次 commit"。
