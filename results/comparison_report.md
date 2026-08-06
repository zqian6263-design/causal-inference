# 方法 × 数据 对比矩阵（多 seed 口径，Phase 3 输入）

- 数据: 5 节点 7 边 DAG（TestPC 基准图），3000 样本
- 每个方法跑 5 个 seed（[42, 1, 7, 2024, 999]），数据逐 seed 重生成；SHD 报 mean±std（越小越好），P/R 与时间为 5 seed 平均
- GRaSP/BOSS 在离散数据显式对比 score_func：BIC_from_cov（默认）vs BDeu（此前误把默认当 BDeu）
- ICA-LiNGAM 纳入统一矩阵：系数阈值 0.1 → 邻接掩码 → dag2cpdag（与其余方法同 CPDAG 口径）→ evaluate_graph
- Arrow/Adjacency Confusion 无边时 precision 置 0（evaluate.py 根因修复）

| 数据 | 方法 | SHD(mean±std) | adjP | adjR | arrP | arrR | 时间(s) |
|---|---|---|---|---|---|---|---|
| linear_gaussian | PC+fisherz | 4.80±2.48 | 1.00 | 0.80 | 0.27 | 0.28 | 0.024 |
| linear_gaussian | GES+BIC | 5.40±3.07 | 0.84 | 0.89 | 0.35 | 0.32 | 0.035 |
| linear_gaussian | GRaSP+BIC | 6.60±1.36 | 0.85 | 0.80 | 0.13 | 0.16 | 0.014 |
| linear_gaussian | BOSS+BIC | 0.00±0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.015 |
| linear_nongaussian | PC+fisherz | 5.60±1.62 | 0.94 | 0.80 | 0.20 | 0.24 | 0.029 |
| linear_nongaussian | GES+BIC | 6.60±1.36 | 0.83 | 0.83 | 0.13 | 0.16 | 0.037 |
| linear_nongaussian | GRaSP+BIC | 5.20±2.93 | 0.88 | 0.83 | 0.33 | 0.36 | 0.015 |
| linear_nongaussian | BOSS+BIC | 2.40±3.01 | 0.94 | 0.94 | 0.67 | 0.68 | 0.015 |
| linear_nongaussian | ICA-LiNGAM | 0.00±0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.024 |
| nonlinear_anm | PC+fisherz | 4.40±0.49 | 0.90 | 0.71 | 0.40 | 0.24 | 0.020 |
| nonlinear_anm | GES+BIC | 4.80±0.40 | 0.97 | 0.71 | 0.13 | 0.08 | 0.022 |
| nonlinear_anm | GRaSP+BIC | 5.80±0.98 | 1.00 | 0.66 | 0.00 | 0.00 | 0.011 |
| nonlinear_anm | BOSS+BIC | 5.00±0.00 | 1.00 | 0.71 | 0.00 | 0.00 | 0.011 |
| discrete | PC+chisq | 6.80±1.17 | 0.80 | 0.91 | 0.23 | 0.28 | 0.027 |
| discrete | GES+BDeu | 6.40±1.02 | 0.97 | 0.80 | 0.07 | 0.08 | 0.393 |
| discrete | GRaSP+BIC | 6.00±3.03 | 0.77 | 0.86 | 0.32 | 0.32 | 0.016 |
| discrete | BOSS+BIC | 2.40±2.80 | 0.84 | 0.94 | 0.67 | 0.80 | 0.017 |
| discrete | GRaSP+BDeu | 6.00±0.89 | 0.97 | 0.80 | 0.13 | 0.16 | 0.611 |
| discrete | BOSS+BDeu | 6.00±0.89 | 0.97 | 0.80 | 0.13 | 0.16 | 0.862 |
