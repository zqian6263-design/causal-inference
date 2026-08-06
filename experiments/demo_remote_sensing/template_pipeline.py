# -*- coding: utf-8 -*-
"""
通用因果分析管道（template_pipeline.py）
========================================
数据 → 方法选择 → 运行 → 评估 → 图 → Markdown 报告, 一键完成。

用法:
  1. 复制本目录到任务目录
  2. 修改 template_data_gen.load_your_data() 换成你的数据
  3. 运行: cd causal-lab && python \
       experiments/10_templates/template_pipeline.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 本目录（template_data_gen）
import numpy as np

from template_data_gen import load_your_data as load_template_data, describe_data, quick_select_method
from scripts.evaluate import evaluate_graph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.ScoreBased.GES import ges

SEED = 42
OUT_DIR = os.path.join("results", "template_out")


def run_with_truth(data, truth_adj, methods):
    """合成数据: 有真值图 → 定量评估。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    report = ["# 因果分析报告（模板）\n", f"- 时间: {time.strftime('%Y-%m-%d %H:%M')}",
              f"- 数据: {data.shape[0]} 样本 × {data.shape[1]} 变量, seed={SEED}",
              "- 评估: SHD / Adjacency P-R / Arrow P-R（真值 CPDAG 对齐）\n",
              "| 方法 | SHD | adjP | adjR | arrP | arrR | 时间(s) |", "|---|---|---|---|---|---|---|"]
    all_m = {}
    for name, fn in methods:
        t0 = time.time()
        try:
            est = fn(data)
            m = evaluate_graph(truth_adj, est)
            m["time_s"] = round(time.time() - t0, 3)
            print(f"{name}: {m}")
            report.append(f"| {name} | {m['SHD']} | {m['adj_precision']} | {m['adj_recall']} | "
                          f"{m['arrow_precision']} | {m['arrow_recall']} | {m['time_s']} |")
            all_m[name] = m
        except Exception as e:
            print(f"{name}: FAILED {e}")
            report.append(f"| {name} | - | - | - | - | - | FAILED |")
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_m, f, indent=2)
    print(f"\n📄 报告: {os.path.join(OUT_DIR, 'report.md')}")
    print(f"📊 指标: {os.path.join(OUT_DIR, 'metrics.json')}")


def main():
    data, meta = load_template_data()
    info = describe_data(data, meta.get("feature_names"))
    recs = quick_select_method(info)
    print("\n按推荐运行:", [r[0] for r in recs[:2]])

    # 示例管道: 默认跑 PC+fisherz 和 GES+BIC（连续数据）
    methods = [
        ("PC+fisherz", lambda d: pc(d, 0.05, fisherz, show_progress=False)),
        ("GES+BIC", lambda d: ges(d)["G"]),
    ]
    # 若有真值图（合成数据）→ 定量评估; 否则仅输出图结构
    truth_adj = meta.get("truth_adj")
    if truth_adj is not None:
        run_with_truth(data, truth_adj, methods)
    else:
        for name, fn in methods:
            est = fn(data)
            print(f"\n{name} 边数: {est.G.get_num_edges() if hasattr(est, 'G') else est.get_num_edges()}")
            print("⚠️ 真实数据无真值图 → 建议 bootstrap 稳定性检验 + 领域知识核对")


if __name__ == "__main__":
    main()
