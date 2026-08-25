# -*- coding: utf-8 -*-
"""
通用数据适配器（template_data_gen.py）
======================================
把任意数据集变成 causal-learn 可用的输入。

用法: 把本文件复制到你的任务目录, 实现 load_your_data() 返回 (data, meta)。
meta 至少含: {'n_samples': N, 'n_features': D, 'feature_names': [...]}

数据形态要求:
  - data: numpy (n_samples, n_features), 连续或离散均可
  - 缺失值: 用 np.nan, 选择 MVPC (mv_fisherz) 方法处理
  - 时序: 传入 (T, D) 时间有序矩阵, 用 Granger / VAR-LiNGAM
"""
import numpy as np


def describe_data(data, feature_names=None, is_time_series=False):
    """输出数据体检报告——用于方法选型决策树输入。"""
    n, d = data.shape
    n_nan = int(np.isnan(data).sum())
    info = {
        "n_samples": n,
        "n_features": d,
        "missing_ratio": round(n_nan / (n * d), 4),
        "has_missing": n_nan > 0,
        "is_time_series": is_time_series,
        "feature_names": feature_names or [f"X{i + 1}" for i in range(d)],
        "dtypes": [str(data[:, i].dtype) for i in range(d)],
        # 粗略判断连续/离散: 唯一值数 < 10 视为离散
        "discrete_cols": [i for i in range(d) if np.unique(data[:, i][~np.isnan(data[:, i])]).size < 10],
    }
    info["has_discrete"] = len(info["discrete_cols"]) > 0
    print("[数据体检]")
    for k, v in info.items():
        print(f"   {k}: {v}")
    return info


def quick_select_method(info):
    """基于体检结果给出方法建议（对应 knowledge/08 决策树）。"""
    rec = []
    if info["is_time_series"]:
        rec.append(("Granger / VAR-LiNGAM", "时序数据首选"))
    if info["has_missing"]:
        rec.append(("PC + mv_fisherz (MVPC)", "缺失值用 testwise-deletion 检验"))
    if info["has_discrete"]:
        rec.append(("BOSS+BDeu / PC+chisq", "离散数据（小/中规模 BOSS+BDeu 实证最优；大图 PC+chisq）"))
    else:
        rec.append(("PC+fisherz / GES+BIC / BOSS", "连续线性默认（实测 SHD=0）"))
        rec.append(("ICA-LiNGAM", "若怀疑非高斯噪声（完整 DAG）"))
    if info["n_features"] <= 15:
        rec.append(("ExactSearch", "小图追求全局最优"))
    # 隐变量无法自动判断——提醒用户
    print("[建议方法]", rec)
    print("   [!] 若怀疑存在未观测混杂 → 换 FCI (PAG) 或 RLCD (显式隐变量)")
    return rec


# ====== 示例: 遥感特征-标签数据（含混淆结构，演示用）======
# 因果结构（真值）:
#   X3(土壤湿度) → X1(植被指数) → Y(产量)
#   X3 → Y                     （X3 是混淆变量: 同时影响 X1 和 Y）
#   X2(波段噪声) → 无关         （测试方法能否剔除伪相关）
# 因果上 Y 的父节点 = {X1, X3}; X2 与 Y 无关
def load_your_data():
    rng = np.random.RandomState(42)
    n = 3000
    X3 = rng.randn(n)                                   # 土壤湿度
    X2 = rng.randn(n)                                   # 无关波段噪声
    X1 = 0.8 * X3 + 0.5 * rng.randn(n)                  # 植被指数(受湿度影响)
    Y = 0.6 * X1 + 0.4 * X3 + 0.5 * rng.randn(n)        # 产量(受植被指数+湿度)
    data = np.column_stack([X1, X2, X3, Y])
    meta = {
        "n_samples": n, "n_features": 4,
        "feature_names": ["NDVI(植被指数)", "BandNoise(波段噪声)", "SoilMoist(土壤湿度)", "Yield(产量)"],
        "truth_adj": np.array([[0, 0, 0, 1],      # X1 → Y
                               [0, 0, 0, 0],      # X2 无关
                               [1, 0, 0, 1],      # X3 → X1, X3 → Y
                               [0, 0, 0, 0]]),    # Y 无出边
    }
    return data, meta


if __name__ == "__main__":
    data, meta = load_your_data()
    info = describe_data(data, meta.get("feature_names"))
    quick_select_method(info)
