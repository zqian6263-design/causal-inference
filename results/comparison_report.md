# 方法 × 数据 对比矩阵（Phase 3 输入）

- 数据: 5 节点 7 边 DAG（TestPC 基准图）, 3000 样本, seed=42
- 指标: SHD（越小越好）/ Adjacency P/R / Arrow P/R / 时间

| 数据 | 方法 | SHD | adjP | adjR | arrP | arrR | 时间(s) |
|---|---|---|---|---|---|---|---|
| linear_gaussian | PC+fisherz | 0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.026 |
| linear_gaussian | GES+BIC | 0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.044 |
| linear_gaussian | GRaSP | 0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.016 |
| linear_gaussian | BOSS | 0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.017 |
| linear_nongaussian | PC+fisherz | 8 | 0.8571 | 0.8571 | 0.0 | 0.0 | 0.024 |
| linear_nongaussian | GES+BIC | 5 | 0.8571 | 0.8571 | 0.3333 | 0.4 | 0.046 |
| linear_nongaussian | GRaSP | 5 | 0.8571 | 0.8571 | 0.3333 | 0.4 | 0.017 |
| linear_nongaussian | BOSS | 5 | 0.8571 | 0.8571 | 0.3333 | 0.4 | 0.017 |
| linear_nongaussian | ICA-LiNGAM | - | - | - | - | - | 0.032 |
| nonlinear_anm | PC+fisherz | 4 | 0.8333 | 0.7143 | 0.6667 | 0.4 | 0.019 |
| nonlinear_anm | GES+BIC | 4 | 0.8333 | 0.7143 | 0.6667 | 0.4 | 0.027 |
| nonlinear_anm | GRaSP | 5 | 1.0 | 0.7143 | nan | 0.0 | 0.01 |
| nonlinear_anm | BOSS | 5 | 1.0 | 0.7143 | nan | 0.0 | 0.011 |
| discrete | PC+chisq | 7 | 0.7778 | 1.0 | 0.3333 | 0.4 | 0.035 |
| discrete | GES+BDeu | 5 | 1.0 | 0.8571 | 0.3333 | 0.4 | 0.506 |
| discrete | GRaSP | 1 | 0.875 | 1.0 | 0.8333 | 1.0 | 0.016 |
| discrete | BOSS | 1 | 0.875 | 1.0 | 0.8333 | 1.0 | 0.016 |
