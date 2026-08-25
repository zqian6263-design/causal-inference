<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/causal--learn-0.1.4.8-4B8BBE?style=flat-square" alt="causal-learn"/>
  <img src="https://img.shields.io/badge/License-MIT%20%2B%20CC--BY--4.0-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/CI-smoke%20passing-0AAB0A?style=flat-square" alt="CI"/>
  <img src="https://img.shields.io/badge/实验-13%20方法全复现-orange?style=flat-square" alt="experiments"/>
</p>

# 🧬 causal-lab：因果推断学习与方法选型工具箱

> **一句话**：把 causal-learn（CMU CLeaR 官方因果发现库）从文档到实验全量梳理——**9 篇中文知识库、13 个方法可复现实验、实证对比矩阵、复制即用的分析模板**。拿到任何因果数据分析任务，照着选方法、抄模板就能跑。

---

## ✨ 为什么值得用

| | 内容 |
|---|---|
| 📚 **9 篇中文知识库** | 从 DAG/等价类/三大范式 → 13 个方法逐个深入，每篇附**可运行示例 + 实测输出 + 踩坑记录** |
| 🧪 **13 方法实验全可复现** | PC / FCI / CD-NOD / GES / DGES / ExactSearch / LiNGAM 家族 / ANM / PNL / GIN / RLCD / GRaSP / BOSS / Granger——每个都有脚本 + 指标 + 图 |
| 📊 **实证对比矩阵** | 4 类数据 × 7 方法 × **5 个 seed（mean±std）**，结论全部有数据支撑，不靠运气 |
| 🚀 **复制即用模板** | 新任务 3 步上手：复制模板 → 改数据 → 出报告 |
| ✅ **CI 冒烟** | `clone → pip install → 全跑通`，可复现性有机器兜底 |

---

## 🚀 30 秒上手

```bash
git clone https://github.com/zqian6263-design/causal-inference.git
cd causal-inference
pip install -r requirements.txt

# 跑第一个因果发现实验（PC 算法 + 评估）
python experiments/01_pc/run.py
```

输出：`SHD=0  adjP/R=1.0/1.0` —— 从 3000 个样本里把 5 个变量的因果图完整找回来了。

---

## 🆕 新分析任务 3 步走

```bash
# ① 复制模板到你的任务目录
cp -r experiments/10_templates/ /path/to/你的任务/

# ② 打开 template_data_gen.py，改 load_your_data() 换成你的数据（CSV/npy 都行）
# ③ 跑管道：自动体检 → 推荐方法 → 运行 → 评估 → 生成 Markdown 报告
python template_pipeline.py
```

> 模板自带「数据体检 + 方法建议」，会自动判断你的数据是连续/离散/时序/有缺失，并给出推荐方法。

---

## 🎯 方法选型速查（实证结论，详见 knowledge/08）

| 你的数据 | 直接选 | 为什么 |
|---|---|---|
| 连续 + 线性 + 高斯 | **BOSS** / PC+fisherz / GES+BIC | BOSS 5 seed 全对（SHD=0.0±0.0） |
| 连续 + 线性 + 非高斯 | **ICA-LiNGAM** | 唯一稳定完美恢复完整 DAG（0.0±0.0） |
| 离散 / 分类（小/中规模） | **BOSS+BDeu** | 真实 bnlearn 实证 6/8 领先（sachs SHD=0） |
| 离散 / 分类（大图 ≥37 节点） | PC+chisq | BOSS+BDeu 打分超时不可行，PC 唯一可用（质量退化） |
| 非线性 | PC+KCI（大样本）或 ANM/PNL | 线性假设方法全受限 |
| 时序 | Granger / VAR-LiNGAM | 滞后因果 |
| 怀疑有隐变量 | FCI / RLCD | PAG 或显式隐变量节点 |
| 有缺失值 | PC + mv_fisherz | 无需删行 |

---

## 📁 仓库结构

```
causal-inference/
├── knowledge/          9 篇中文知识库（00 基础 → 08 ★选型指南）
├── experiments/        13 方法实验 + 统一对比 + 模板 + 遥感演示
├── scripts/            数据生成器 / 评估器 / 绘图工具
├── results/            实证对比矩阵 + 全部指标 JSON + 图
├── requirements.txt    锁定依赖（clone 即装）
└── .github/workflows/  CI 冒烟（smoke）
```

---

## 📖 怎么学

按 `knowledge/00 → 01 → … → 08` 顺序读，每篇末尾有可运行示例：

| 篇目 | 内容 |
|---|---|
| 00 因果推断基础 | DAG / d-分离 / 等价类 / 三大范式 |
| 01 方法全景 | 13 方法总览 + 快速初判 |
| 02–06 方法深入 | 约束型 / 打分型 / 函数模型 / 隐变量排列 / 时序 |
| 07 评估与工程 | SHD / 可视化 / 缓存 / 数据集 |
| 08 ★选型指南 | 决策树 + 实证矩阵 + 场景配方 |

---

## 🧪 实验证据

- 📊 对比矩阵报告：`results/comparison_report.md`（4 数据 × 7 方法 × 5 seed，mean±std）
- 📈 全部图：`results/figs/`（10 张，含 FCI 的 PAG、GIN 隐变量簇等）
- ✅ 指标 JSON：`results/metrics/`（合法 JSON，可复现）

---

## ⚠️ 已知限制（诚实声明）

- 实验基于合成数据（真值已知才能评估）；真实数据集见 `knowledge/07` 的说明
- KCI 检验慢且小样本检验力不足（详见 knowledge/08）
- GRaSP/BOSS 有内部随机性 → 仓库统一用 5 seed mean±std 纪律

---

## 📄 许可

- 文档与知识库：**CC-BY-4.0**
- 代码：**MIT**

---

*Built with [causal-learn](https://github.com/py-why/causal-learn) · 数据生成对齐官方基准 · 结论经多 seed 验证*
