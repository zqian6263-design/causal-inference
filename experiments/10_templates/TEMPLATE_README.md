# 通用实验模板使用说明（TEMPLATE_README.md）

把 `experiments/10_templates/` 当作任何新因果分析任务的**起点**。

## 三步使用法

1. **改数据**：在 `template_data_gen.py` 的 `load_your_data()` 里换成你的数据源
   （CSV / Excel / npy / 数据库），返回 `(data, meta)`，meta 里给 `feature_names`；
   合成数据（有真值图）时加 `meta["truth_adj"]` 即可自动定量评估。
2. **看建议**：跑 `template_data_gen.py` 看「数据体检 + 方法建议」
   （对应 `knowledge/08-方法选型指南.md` 的决策树，自动判断连续/离散/缺失/时序）。
3. **跑管道**：`template_pipeline.py` 一键完成
   数据 → 方法选择 → 运行 → 评估（SHD/P-R）→ 报告落盘 `results/template_out/`。

## 文件清单

| 文件 | 作用 |
|---|---|
| `template_data_gen.py` | 数据加载 + 体检 + 快速方法建议（决策树实现） |
| `template_pipeline.py` | 一键管道：方法运行 + 评估 + Markdown 报告 |
| 本 README | 使用说明 |

## 常见任务模板组合

| 任务类型 | 改哪里 | 用什么方法 |
|---|---|---|
| 通用观测数据 | `load_your_data()` | PC+fisherz / GES+BIC / BOSS（连续线性） |
| 离散/分类特征 | `load_your_data()` + 确认 `discrete_cols` | BOSS / GES+BDeu / PC+chisq |
| 疑似非高斯连续 | `load_your_data()` | ICA-LiNGAM（完整 DAG） |
| 时序 | 传入 (T,D) 矩阵 + `is_time_series=True` | Granger / VAR-LiNGAM |
| 有缺失 | 保留 np.nan | PC + mv_fisherz |
| 疑似隐变量 | 手动 | FCI / RLCD |

## 注意事项

- 运行前缀必须是 `PYTHONPATH=`（防 Hermes venv 劫持 import）
- 真值图评估：管道自动做 CPDAG 对齐（`dag2cpdag`），不要手动对比 DAG vs CPDAG
- seed 固定 42；真实数据无真值 → 加 bootstrap 稳定性检验（边出现频率），别只跑一次
- 图表：`GraphUtils.to_pydot(cg.G).write_png(...)` 落盘 `results/figs/`
