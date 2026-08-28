# -*- coding: utf-8 -*-
"""
experiments/08_benchmarks/compare_official.py — 官方 Test*.py 基准逐位对照（PC + GES + FCI）
============================================================================================
causal-learn 官方 tests/TestData/benchmark_returned_results/ 存有各方法的输出图矩阵基准。
本脚本用与官方 Test*.py **完全一致**的调用重跑，把输出 `G.graph` 与官方文件**逐位比对**
（MD5 级一致性），覆盖三大范式：

  Part A PC+chisq  bnlearn 13 集    官方调用 pc(data, 0.05, chisq, True, 0, -1)（stable, uc_rule=0, uc_priority=-1）
  Part B GES       合成 10 节点     官方调用 ges(data, score_func='local_score_BIC'/'local_score_BDeu', maxP=None, parameters=None)
  Part C FCI       bnlearn 13 集 + 合成 10 节点  官方调用 fci(data, chisq/fisherz, 0.05, verbose=False)

注: 官方基准来自旧 commit（TestPC=5918419 / TestGES=b51d788 / FCI 同期），若逐位不一致多为
版本漂移或参数默认值变化（如 PC 的 uc_priority，对齐官方精确调用后可 13/13）。**FCI 例外**：
其调用已与官方完全一致（depth=-1, max_path_length=-1, chisq/fisherz, 0.05），仍 9/14 差 1-7 格
——这是 0.1.4.8 与官方旧 commit 的 **PAG 定向规则版本漂移**（TestFCI.py 自注：不一致 ≠ 实现错误），
无参数可对齐，如实标注即可。

用法: python experiments/08_benchmarks/compare_official.py [bnlearn数据目录]
产出: results/metrics/compare_official.json（每方法 × 每数据集 match + 差异格数）
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq, fisherz

DATASETS = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley",
            "child", "insurance", "water", "hailfinder", "hepar2", "win95pts"]
OUT = os.path.join("results", "metrics", "compare_official.json")
DEFAULT_SUB = os.path.join("..", "..", "..", "causal-learn", "tests", "TestData",
                           "bnlearn_discrete_10000")


def _diff(gm, official):
    if gm.shape == official.shape:
        return int(np.sum(gm != official))
    return -1


def _report(name, ds, gm, official, t0):
    dt = round(time.time() - t0, 2)
    match = bool(np.array_equal(gm, official))
    nd = _diff(gm, official)
    print(f"  {name:<10} {ds:<12} {'[MATCH]' if match else '[DIFF]'} 差异格={nd:<4} "
          f"{gm.shape[0]}×{gm.shape[0]}  {dt}s", flush=True)
    return {"match": match, "n_diff_cells": nd, "nodes": gm.shape[0], "time_s": dt}


def part_a_pc(data_dir, bmr):
    """PC+chisq: bnlearn 13 集（官方精确签名 uc_priority=-1）。"""
    print("Part A: PC+chisq（bnlearn 13 集, 官方签名 pc(data,0.05,chisq,True,0,-1)）")
    res = {"call": "pc(data, 0.05, chisq, True, 0, -1)", "datasets": {}}
    for ds in DATASETS:
        off = os.path.join(bmr, f"{ds}_pc_chisq_0.05_stable_0_-1.txt")
        data = np.loadtxt(os.path.join(data_dir, "data", f"{ds}.txt"), skiprows=1, dtype=int)
        t0 = time.time()
        cg = pc(data, 0.05, chisq, True, 0, -1, show_progress=False)
        res["datasets"][ds] = _report("pc_chisq", ds, np.asarray(cg.G.graph, dtype=int),
                                      np.loadtxt(off, dtype=int), t0)
    res["matched"] = sum(1 for d in res["datasets"].values() if d["match"])
    print(f"  -> PC+chisq 对照: {res['matched']}/{len(DATASETS)}\n")
    return res


def part_b_ges(testdata_dir, bmr_parent):
    """GES: 合成 10 节点（BIC 连续 + BDeu 离散），官方签名。

    基准文件在 tests/TestData/benchmark_returned_results/（父目录，非 bnlearn 子目录）。
    """
    print("Part B: GES（合成 10 节点, 官方签名 ges(data, score_func, maxP=None, parameters=None)）")
    res = {"datasets": {}}
    for sf, dfile, bfile in [
        ("local_score_BIC", "data_linear_10.txt", "linear_10_ges_local_score_BIC_none_none.txt"),
        ("local_score_BDeu", "data_discrete_10.txt", "discrete_10_ges_local_score_BDeu_none_none.txt"),
    ]:
        data = np.loadtxt(os.path.join(testdata_dir, dfile), skiprows=1)
        t0 = time.time()
        G = ges(data, score_func=sf, maxP=None, parameters=None)["G"]
        res["datasets"][f"{sf}_{dfile.replace('data_','').replace('.txt','')}"] = _report(
            "ges", sf, np.asarray(G.graph, dtype=int),
            np.loadtxt(os.path.join(bmr_parent, bfile), dtype=int), t0)
    res["matched"] = sum(1 for d in res["datasets"].values() if d["match"])
    print(f"  -> GES 对照: {res['matched']}/{len(res['datasets'])}\n")
    return res


def part_c_fci(data_dir, bmr_parent, testdata_dir):
    """FCI: bnlearn 13 集（chisq）+ 合成 10 节点（fisherz），官方签名。

    基准文件在 tests/TestData/benchmark_returned_results/（父目录，非 bnlearn 子目录）。
    """
    print("Part C: FCI（bnlearn 13 集 chisq + linear_10 fisherz, 官方签名 fci(data, cit, 0.05, verbose=False)）")
    res = {"datasets": {}}
    for ds in DATASETS:
        off = os.path.join(bmr_parent, f"bnlearn_discrete_10000_{ds}_fci_chisq_0.05.txt")
        data = np.loadtxt(os.path.join(data_dir, "data", f"{ds}.txt"), skiprows=1, dtype=int)
        t0 = time.time()
        G, _edges = fci(data, chisq, 0.05, verbose=False)
        res["datasets"][f"bnlearn_{ds}"] = _report("fci_chisq", ds,
                                                   np.asarray(G.graph, dtype=int),
                                                   np.loadtxt(off, dtype=int), t0)
    # linear_10 fisherz
    off = os.path.join(bmr_parent, "linear_10_fci_fisherz_0.05.txt")
    data = np.loadtxt(os.path.join(testdata_dir, "data_linear_10.txt"), skiprows=1)
    t0 = time.time()
    G, _edges = fci(data, fisherz, 0.05, verbose=False)
    res["datasets"]["linear_10_fisherz"] = _report("fci_fisherz", "linear_10",
                                                   np.asarray(G.graph, dtype=int),
                                                   np.loadtxt(off, dtype=int), t0)
    res["matched"] = sum(1 for d in res["datasets"].values() if d["match"])
    print(f"  -> FCI 对照: {res['matched']}/{len(res['datasets'])}\n")
    return res


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), DEFAULT_SUB))
    if not os.path.isdir(os.path.join(data_dir, "data")):
        sys.exit(f"找不到 bnlearn 数据目录: {data_dir}")
    bmr = os.path.join(data_dir, "benchmark_returned_results")          # bnlearn 子目录（PC 基准）
    testdata_dir = os.path.dirname(data_dir)                            # tests/TestData
    bmr_parent = os.path.join(testdata_dir, "benchmark_returned_results")  # 父目录（GES/FCI 基准）

    out = {"note": "官方 Test*.py 基准逐位对照（三大范式: PC / GES / FCI），调用与官方完全一致"}
    out["pc_chisq_bnlearn"] = part_a_pc(data_dir, bmr)
    out["ges_synthetic10"] = part_b_ges(testdata_dir, bmr_parent)
    out["fci"] = part_c_fci(data_dir, bmr_parent, testdata_dir)
    out["summary"] = {k: v["matched"] for k, v in out.items() if isinstance(v, dict) and "matched" in v}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"指标已落盘: {OUT}  (allow_nan=False)\n汇总: {out['summary']}")


if __name__ == "__main__":
    main()