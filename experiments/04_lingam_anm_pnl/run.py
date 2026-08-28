# -*- coding: utf-8 -*-
"""
knowledge/04-函数因果模型.md 复现脚本（示例与文档逐字一致）+ 批次 C 扩展
======================================================================
用法: cd causal-lab && python experiments/04_lingam_anm_pnl/run.py

批次 C 扩展:
  示例 3: VAR-LiNGAM（2 变量滞后因果时序数据, lags=2, n=500）——输出滞后系数矩阵
  示例 4: PNL 成对方向判断（后非线性 y=(x+x^3+e)^2, n=400）——PNL import 约 45s
  示例 5: RCD 最小 API 调用（`lingam.RCD`；数据无隐变量, 仅演示 API 可用性）
  示例 6: CAM-UV 最小调用（`lingam.CAMUV`；依赖 pygam, 2026-08 已批准安装——
          CI/requirements 无 pygam 时 try/except 降级跳过，不阻断整脚本）

产出:
  - results/metrics/04_lingam_anm_pnl.json   全示例指标
  - results/figs/04_var_lingam_lags.png       VAR-LiNGAM 滞后系数热力图
  - results/figs/04_pnl_scatter.png           PNL 数据散点
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

OUT = os.path.join("results", "metrics", "04_lingam_anm_pnl.json")
FIG_DIR = os.path.join("results", "figs")
results = {}

print("=" * 70)
print("示例 1: ICA-LiNGAM + DirectLiNGAM（线性非高斯, 期望恢复因果序）")
print("=" * 70)
from causallearn.search.FCMBased import lingam
from scripts.data_gen import simulate_linear_nongaussian

data, truth = simulate_linear_nongaussian(n=3000, seed=42, dist="exponential")
m1 = lingam.ICALiNGAM(random_state=42)
m1.fit(data)
print(f"ICA-LiNGAM causal_order:    {m1.causal_order_}")
print(f"ICA-LiNGAM 系数非零项:      {np.count_nonzero(np.abs(m1.adjacency_matrix_) > 0.1)}")
assert list(m1.causal_order_) == [0, 1, 2, 3, 4], "因果序应等于真值拓扑序"
assert np.count_nonzero(np.abs(m1.adjacency_matrix_) > 0.1) == 7, "应有 7 条边"
results["ICA-LiNGAM"] = {"causal_order": [int(v) for v in m1.causal_order_],
                         "n_edges": int(np.count_nonzero(np.abs(m1.adjacency_matrix_) > 0.1))}

m2 = lingam.DirectLiNGAM(random_state=42)
m2.fit(data)
print(f"DirectLiNGAM causal_order:  {m2.causal_order_}")
print(f"DirectLiNGAM 系数非零项:    {np.count_nonzero(np.abs(m2.adjacency_matrix_) > 0.1)}")
results["DirectLiNGAM"] = {"causal_order": [int(v) for v in m2.causal_order_],
                           "n_edges": int(np.count_nonzero(np.abs(m2.adjacency_matrix_) > 0.1))}

print()
print("=" * 70)
print("示例 2: ANM 成对方向判断（x→y, y=0.5x+sin(x)+e）")
print("=" * 70)
from causallearn.search.FCMBased.ANM.ANM import ANM

rng = np.random.RandomState(0)
x = rng.randn(500)
y = 0.5 * x + np.sin(x) + np.random.RandomState(1).randn(500) * 0.3
anm = ANM()
p_fwd, p_bwd = anm.cause_or_effect(x.reshape(-1, 1), y.reshape(-1, 1))
print(f"ANM p_forward (x→y): {p_fwd:.6f}")
print(f"ANM p_backward(y→x): {p_bwd:.6e}")
assert p_fwd > 0.05 and p_bwd < 0.05, "应判定 x→y"
results["ANM"] = {"p_forward": float(p_fwd), "p_backward": float(p_bwd),
                  "verdict": "x->y"}

print()
print("=" * 70)
print("示例 3: VAR-LiNGAM（2 变量滞后因果时序, lags=2, n=500）")
print("=" * 70)
# 真值结构: x2[t] 受 x2[t-1], x1[t-1], x1[t-2] 影响; 噪声非高斯（指数）
n_var = 500
e1 = np.random.RandomState(1).exponential(1.0, n_var) - 1.0
e2 = np.random.RandomState(2).exponential(1.0, n_var) - 1.0
x1 = np.zeros(n_var)
x2 = np.zeros(n_var)
x1[0], x2[0] = e1[0], e2[0]
for t in range(1, n_var):
    x1[t] = 0.6 * x1[t - 1] + e1[t]
    x2[t] = 0.7 * x2[t - 1] + 0.5 * x1[t - 1] + e2[t]
    if t >= 2:
        x2[t] += -0.3 * x1[t - 2]
X = np.column_stack([x1, x2])

t0 = time.time()
m3 = lingam.VARLiNGAM(lags=2)
m3.fit(X)
var_time = round(time.time() - t0, 3)
adj_lags = [a.round(3).tolist() for a in m3.adjacency_matrices_]
print(f"VAR-LiNGAM 耗时: {var_time}s; lags={len(m3.adjacency_matrices_) - 1}")
print("  lag-0（同期）矩阵（[i,j]=X_j 对 X_i 同期影响）:\n", np.array(adj_lags[0]))
print("  lag-1 矩阵:\n", np.array(adj_lags[1]))
print("  lag-2 矩阵:\n", np.array(adj_lags[2]))
# 验证: lag-1 x1->x2 与 lag-2 x1->x2 均应检出（系数显著非零且符号正确）
est_lag1_x1x2 = m3.adjacency_matrices_[1][1, 0]   # X1(lag1) -> X2
est_lag2_x1x2 = m3.adjacency_matrices_[2][1, 0]   # X1(lag2) -> X2
print(f"  检出: X1(lag1)->X2={est_lag1_x1x2:.3f}（真值 0.5）;  X1(lag2)->X2={est_lag2_x1x2:.3f}（真值 -0.3）")
assert est_lag1_x1x2 > 0.3, "lag-1 x1->x2 应检出为正"
assert est_lag2_x1x2 < 0, "lag-2 x1->x2 应检出海为负"
results["VAR-LiNGAM"] = {"time_s": var_time, "lags": 2,
                         "adjacency_matrices": adj_lags,
                         "lag1_X1->X2": round(float(est_lag1_x1x2), 3),
                         "lag2_X1->X2": round(float(est_lag2_x1x2), 3)}

# VAR-LiNGAM 滞后系数热力图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for k, (ax, name) in enumerate(zip(axes, ["lag 0 (same-time)", "lag 1", "lag 2"])):
    A = m3.adjacency_matrices_[k]
    im = ax.imshow(A, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_title(f"VAR-LiNGAM {name}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["x1", "x2"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["x1", "x2"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=list(axes), shrink=0.85)
fig.subplots_adjust(wspace=0.45)
fig.savefig(os.path.join(FIG_DIR, "04_var_lingam_lags.png"), dpi=200)
plt.close(fig)

print()
print("=" * 70)
print("示例 4: PNL 成对方向判断（后非线性 y=(x+x^3+e)^2, n=400, import 约 45s）")
print("=" * 70)
# 惰性导入（try 块内）：PNL 是重依赖（torch 非 causal-learn 必装 + import ~45s），
# 仅当本示例确实用到时才 import——批量任务只想跑 1-3 时不背 PNL 的启动开销；
# 且 torch 缺失（如 CI 装 requirements 后）自动降级跳过本示例，不阻断整脚本。
try:
    import torch  # PNL 的依赖；causal-learn 不强依赖 torch（CI 装 requirements 后无 torch）
    from causallearn.search.FCMBased.PNL.PNL import PNL   # 重依赖, import 慢
    _pnl_available = True
except ImportError as e:
    print(f"[跳过] PNL 需 torch，当前环境无（{e}）——CI/requirements 无 torch 时跳过，本地 pytorch env 正常")
    results["PNL"] = {"skipped": "torch not installed (causal-learn optional dep)"}
    _pnl_available = False

if _pnl_available:
    t0 = time.time()
    n_pnl = 400
    rng = np.random.RandomState(0)
    x = rng.randn(n_pnl, 1)
    x = x / x.std()
    e = rng.randn(n_pnl, 1)
    e = e / e.std()
    y = (x + x ** 3 + e) ** 2
    pnl = PNL()
    p_fwd, p_bwd = pnl.cause_or_effect(x, y)
    p_fwd, p_bwd = float(np.asarray(p_fwd).ravel()[0]), float(np.asarray(p_bwd).ravel()[0])
    pnl_time = round(time.time() - t0, 3)   # 含 import（首次约 45s+）
    print(f"PNL import+run 耗时: {pnl_time}s（首次含 ~45s import）")
    print(f"PNL p_forward (x→y): {p_fwd:.4f}")
    print(f"PNL p_backward(y→x): {p_bwd:.4f}")
    verdict = "x->y" if p_fwd > p_bwd else "y->x / 不可判"
    print(f"方向判定: {verdict}（p 大 = 接受该方向）")
    assert p_fwd > p_bwd, "PNL 应判 x->y（后非线性数据按 x 生成）"
    results["PNL"] = {"time_s_import_plus_run": pnl_time, "n": n_pnl,
                      "p_forward": round(float(p_fwd), 4), "p_backward": round(float(p_bwd), 4),
                      "verdict": verdict}

    # PNL 数据散点
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x.ravel(), y.ravel(), s=8, alpha=0.6, color="#2b6cb0")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("PNL data: y = (x + x^3 + e)^2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_pnl_scatter.png"), dpi=200)
    plt.close(fig)

print()
print("=" * 70)
print("示例 5: RCD 最小调用（带隐变量 LiNGAM 扩展；本数据无隐变量, 仅 API 演示）")
print("=" * 70)
# RCD（Repetitive Causal Discovery）面向含隐变量/混杂的数据; 本实验数据无隐变量,
# 只作最小调用演示 API 可用性（隐变量发现基准不在本批次范围）
t0 = time.time()
np.random.seed(42)   # RCD 内部随机不可控, 用全局 np.random 固定保证可复现
m5 = lingam.RCD()
m5.fit(data)   # data = 示例 1 的线性非高斯数据
rcd_time = round(time.time() - t0, 3)
print(f"RCD 耗时: {rcd_time}s")
print(f"RCD 非零系数数: {np.count_nonzero(np.abs(m5.adjacency_matrix_) > 0.1)}")
print("RCD adjacency_matrix_（round 3）:")
print(np.round(m5.adjacency_matrix_, 3))
print("RCD ancestors_list_:", [list(a) for a in m5.ancestors_list_])
results["RCD"] = {"time_s": rcd_time,
                  "note": "最小 API 调用（数据无隐变量, 非隐变量发现基准实验）",
                  "n_edges": int(np.count_nonzero(np.abs(m5.adjacency_matrix_) > 0.1))}

print()
print("=" * 70)
print("示例 6: CAM-UV 最小调用（未观测混杂; 父节点恢复 + UCP/UBP 不确定性报告）")
print("=" * 70)
# CAM-UV（CAM with Unobserved Confounders）: execute(X, alpha, num_explanatory_vals)
# 返回 (P, U): P[i]=Xi 的父节点索引; U=存在 UCP/UBP（不确定因果路径）的变量对
# 依赖 pygam（非 causal-learn 必装, 2026-08 已获用户批准安装）; n=800 控制 HSIC 耗时
# ⚠️ 惰性导入保护（同示例 4 PNL）：pygam 不在 requirements.txt——CI 装 requirements 后无 pygam，
#    `from ...CAMUV import execute` 内部 import pygam 会 ModuleNotFoundError → 必须 try/except
#    降级跳过，不阻断整脚本（只在本机装了 pygam 时真正执行）。
try:
    from causallearn.search.FCMBased.lingam.CAMUV import execute as camuv_execute
    _camuv_available = True
except ImportError as e:
    print(f"[跳过] CAM-UV 需 pygam，当前环境无（{e}）——CI/requirements 无 pygam 时跳过；"
          f"本地 pytorch env 已装 pygam 则正常")
    results["CAM-UV"] = {"skipped": "pygam not installed (causal-learn optional dep)"}
    _camuv_available = False

if _camuv_available:
    t0 = time.time()
    np.random.seed(42)
    data_camuv, truth_camuv = simulate_linear_nongaussian(n=800, seed=42)
    P, U = camuv_execute(data_camuv, 0.05, 3)
    camuv_time = round(time.time() - t0, 3)
    truth_parents = {i: sorted([p for p in range(5) if truth_camuv[p, i]]) for i in range(5)}
    est_parents = {i: sorted(P[i]) for i in range(5)}
    match = sum(1 for i in range(5) if set(P[i]) == set(truth_parents[i]))
    print(f"CAM-UV 耗时: {camuv_time}s（n=800, pygam 依赖）")
    print(f"真值父节点: {truth_parents}")
    print(f"CAM-UV 父节点: {est_parents}")
    print(f"父节点完全一致: {match}/5")
    print(f"UCP/UBP 不确定对 U: {sorted(sorted(u) for u in U)}（显式报告方向不确定性）")
    results["CAM-UV"] = {"time_s": camuv_time, "n": 800,
                         "parents_match": f"{match}/5",
                         "U_uncertain_pairs": [sorted(map(int, u)) for u in U],
                         "note": "最小调用; U 为 CAM-UV 报告的方向不确定对(UCP/UBP)"}

# 落盘
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
print(f"\n指标落盘: {OUT}")

print()
print("全部复现成功 [OK]")
