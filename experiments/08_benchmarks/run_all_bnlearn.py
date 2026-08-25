# -*- coding: utf-8 -*-
"""
experiments/08_benchmarks/run_all_bnlearn.py — bnlearn 13 数据集全量基准（IMPROVEMENTS #8）
============================================================================================
真实离散贝叶斯网络基准（bnlearn，10000 样本）全量跑 PC+chisq 与 BOSS（离散打分 BDeu），
真值 DAG -> CPDAG 对齐评估 SHD / Adjacency-PR / Arrow-PR，补「离散大图」场景实证。

数据源: D:\\win\\causal-learn\\tests\\TestData\\bnlearn_discrete_10000\\（官方 causal-learn 仓库自带，
本脚本兼容 smoke_asia.py 的通用路径约定；可 <数据目录> 参数覆盖；CI 无法访问则跳过并标注）。

两种运行模式:
  1. 无参数    -> 编排模式: 逐数据集/方法 spawn 子进程（带 wall-clock 超时，
                 BOSS+BDeu 大图分钟级，超时如实标失败，不阻塞其余），汇总落盘
                 bnlearn_all.json + 图表
  2. 有 3 参数  -> 单格模式 (被编排模式调用): run_all_bnlearn.py <dataset> <method> <data_dir>
                 跑一个 (数据集, 方法) 格, 结果写 results/metrics/_bnlearn_cell_<ds>_<m>.json
                 方法: pc_chisq | boss_bdeu

产出:
  - results/metrics/bnlearn_all.json               13 集 × 2 方法全指标
  - results/figs/bnlearn_<alarm|child|win95pts>.png   3 个大数据集估计 CPDAG
  - 本目录 README.md 结果表更新
"""
import sys, os, json, time, subprocess, shutil, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

DATASETS = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley",
            "child", "insurance", "water", "hailfinder", "hepar2", "win95pts"]
METHODS = ["pc_chisq", "boss_bdeu"]
# 每格 wall-clock 超时（s）。BOSS+BDeu 的离散打分（父配置分组）随图规模超线性变慢：
# sachs(11 节点) 已 28s，alarm(37) 实测 >10min 不收敛——因此按规模分级：
#   n<=27  (7 集):     BOSS 给足 1500s           （中小图，预期可完成）
#   28-40  (alarm/water): BOSS 一次性有界尝试 600s（实测锚点：37 节点不适在此量级止损）
#   n>40   (barley/hailfinder/hepar2/win95pts):  直接跳过并显式标注（>37 节点 BDeu 打分更慢，无必要重试）
# 节点数见 bnlearn 官方表（asia8/cancer5/earthquake5/survey6/sachs11/child20/insurance27/
# alarm37/water32/barley48/hailfinder56/hepar2/win95pts 70+/76）。
N_NODES = {"asia": 8, "cancer": 5, "earthquake": 5, "survey": 6, "sachs": 11,
           "child": 20, "insurance": 27, "alarm": 37, "water": 32, "barley": 48,
           "hailfinder": 56, "hepar2": 70, "win95pts": 76}
PC_TIMEOUT = 600               # PC+chisq 很快，任何规模都给足
BOSS_TIMEOUT = 1500            # 中小图（n<=27）
BOSS_TIMEOUT_MID = 600         # 中间图（28-40 节点）一次性有界尝试
BOSS_SKIP_LARGE = [d for d in DATASETS if N_NODES[d] > 40]  # 跳过清单（显式标注）
FIG_SETS = ["alarm", "child", "win95pts"]   # 挑 2-3 个较大数据集出图
OUT = os.path.join("results", "metrics", "bnlearn_all.json")
FIG_DIR = os.path.join("results", "figs")
CELL_DIR = os.path.join("results", "metrics", "_bnlearn_cells")


def cell_timeout(dataset, method):
    if method == "pc_chisq":
        return PC_TIMEOUT
    n = N_NODES.get(dataset, 0)
    if n <= 27:
        return BOSS_TIMEOUT
    return BOSS_TIMEOUT_MID

# smoke_asia.py 默认的相对路径约定
DEFAULT_SUB = os.path.join("..", "..", "..", "causal-learn", "tests", "TestData",
                           "bnlearn_discrete_10000")


def parse_truth_graph(path, n):
    """解析 truth_dag_graph/*.graph.txt（形如 '1. X1 --> X2'）-> 邻接矩阵 X1 编号=1。"""
    adj = np.zeros((n, n), dtype=int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "-->" in line:
                arrow = line.split("-->")
                u = arrow[0].strip().split()[-1]   # '1. X1' -> 'X1'
                v = arrow[1].strip().split()[0]    # 'X2'
                adj[int(u[1:]) - 1, int(v[1:]) - 1] = 1
    return adj


def run_cell(dataset, method, data_dir):
    """单格模式: 跑一个 (数据集, 方法), 结果 JSON 落盘 + 大数据集图对象序列化。"""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # BDeu 内部 pandas 分组弃用提示
    from scripts.evaluate import evaluate_graph
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import chisq
    from causallearn.search.PermutationBased.BOSS import boss

    data_path = os.path.join(data_dir, "data", f"{dataset}.txt")
    truth_path = os.path.join(data_dir, "truth_dag_graph", f"{dataset}.graph.txt")
    data = np.loadtxt(data_path, skiprows=1, dtype=int)
    n_nodes = data.shape[1]
    truth = parse_truth_graph(truth_path, n_nodes)

    t0 = time.time()
    if method == "pc_chisq":
        est = pc(data, 0.05, chisq, show_progress=False)
    else:  # boss_bdeu：离散打分（真实离散 CPD 语义）
        est = boss(data, score_func="local_score_BDeu", verbose=False)
    elapsed = round(time.time() - t0, 3)
    m = evaluate_graph(truth, est)
    m["time_s"] = elapsed
    m["n_nodes"] = int(n_nodes)
    m["truth_edges"] = int(truth.sum())
    if hasattr(est, "G"):
        m["est_edges"] = int(est.G.get_num_edges())
        G = est.G
    else:
        m["est_edges"] = int(est.get_num_edges())
        G = est
    m["method"] = method
    m["dataset"] = dataset

    # 大数据集图对象序列化（供编排模式出图）
    if dataset in FIG_SETS:
        os.makedirs(FIG_DIR, exist_ok=True)
        with open(os.path.join(FIG_DIR, f"bnlearn_{dataset}_{method}.pkl"), "wb") as f:
            pickle.dump(G, f)

    os.makedirs(CELL_DIR, exist_ok=True)
    with open(os.path.join(CELL_DIR, f"{dataset}_{method}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{dataset}/{method}: SHD={m['SHD']} n={n_nodes} e={truth.sum()} {elapsed}s",
          flush=True)
    return m


def load_cell(dataset, method):
    p = os.path.join(CELL_DIR, f"{dataset}_{method}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def main_orchestrator(data_dir):
    results = {}
    for ds in DATASETS:
        print(f"\n[数据集] {ds}", flush=True)
        results[ds] = {}
        for method in METHODS:
            if method == "boss_bdeu" and ds in BOSS_SKIP_LARGE:
                # 显式跳过 + 标注（见模块 docstring：>40 节点 BDeu 打分不实用）
                msg = f"skipped: BOSS+BDeu 对 >40 节点(实际 {N_NODES[ds]} 节点)真实图不实用（alarm 37 节点 600s 超时锚点）"
                print(f"  {method:<10} SKIPPED（{msg}）", flush=True)
                os.makedirs(CELL_DIR, exist_ok=True)
                with open(os.path.join(CELL_DIR, f"{ds}_{method}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"error": msg, "dataset": ds, "method": method,
                               "n_nodes": N_NODES[ds]},
                              f, ensure_ascii=False, indent=2, allow_nan=False)
                results[ds][method] = {"error": msg, "n_nodes": N_NODES[ds]}
                continue
            existing = load_cell(ds, method)   # 中断恢复：已跑过的格复用
            if existing is not None:
                print(f"  复用已有: {ds}/{method} SHD={existing.get('SHD')}", flush=True)
                results[ds][method] = existing
                continue
            cell_out = os.path.join(CELL_DIR, f"{ds}_{method}.json")
            if os.path.exists(cell_out):
                os.remove(cell_out)
            try:
                subprocess.run(
                    [sys.executable, os.path.abspath(__file__), ds, method, data_dir],
                    timeout=cell_timeout(ds, method), capture_output=True,
                )
                m = load_cell(ds, method)
                results[ds][method] = m if m else {"error": "subprocess 未产出结果文件"}
                if m:
                    print(f"  {method:<10} SHD={m['SHD']}  time={m['time_s']}s  "
                          f"n={m['n_nodes']} e={m['truth_edges']}", flush=True)
            except subprocess.TimeoutExpired:
                msg = f"timeout>{cell_timeout(ds, method)}s (BOSS+BDeu 大图不实用)"
                print(f"  {method:<10} TIMEOUT（{msg}），标记失败", flush=True)
                # 写回 cell 文件，重启/续跑不再重复尝试
                os.makedirs(CELL_DIR, exist_ok=True)
                with open(os.path.join(CELL_DIR, f"{ds}_{method}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"error": msg, "dataset": ds, "method": method},
                              f, ensure_ascii=False, indent=2, allow_nan=False)
                results[ds][method] = {"error": msg}

    # ---- 汇总 bnlearn_all.json ----
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "causal-learn tests/TestData/bnlearn_discrete_10000",
                   "n_samples": 10000, "cell_timeout_s": {"pc_chisq": PC_TIMEOUT,
                                                          "boss_bdeu": BOSS_TIMEOUT,
                                                          "boss_bdeu_mid": BOSS_TIMEOUT_MID},
                   "boss_skip_large": BOSS_SKIP_LARGE,
                   "methods": METHODS, "results": results},
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n指标已落盘: {OUT}  (allow_nan=False, 无 NaN 字面量)", flush=True)

    # ---- FIG_SETS 数据集的 PC+chisq 估计 CPDAG 出图（BOSS 失败不影响）----
    from scripts.plotting import plot_graph
    os.makedirs(FIG_DIR, exist_ok=True)
    for ds in FIG_SETS:
        pkl = os.path.join(FIG_DIR, f"bnlearn_{ds}_pc_chisq.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            G = pickle.load(f)
        png = os.path.join(FIG_DIR, f"bnlearn_{ds}_pc_chisq.png")
        plot_graph(G, png, title=f"bnlearn {ds} ({len(G.get_nodes())} nodes): PC+chisq CPDAG",
                   figsize=(14, 10))
        print(f"图落盘: {png}", flush=True)

    # 清理单元结果目录（避免污染 git）
    shutil.rmtree(CELL_DIR, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_cell(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
            os.path.join(os.path.dirname(__file__), DEFAULT_SUB))
        if not os.path.isdir(os.path.join(data_dir, "data")):
            sys.exit(f"找不到 bnlearn 数据目录: {data_dir}\n"
                     f"请先获取 causal-learn tests/TestData/bnlearn_discrete_10000/，或用参数指定")
        main_orchestrator(data_dir)