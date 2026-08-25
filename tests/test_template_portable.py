# -*- coding: utf-8 -*-
"""
tests/test_template_portable.py — 模板「复制即用」回归测试（IMP #7）
======================================================================
批次 D 把 10_templates 改为完全自包含（无仓库根依赖）；本测试把「复制到独立临时目录
能独立跑通」固化为回归测试，防止未来改动再破坏。

验证内容:
  1. 复制 experiments/10_templates 到独立临时目录（与仓库根隔离）
  2. 子进程跑 template_data_gen.py  -> exit=0 且输出含 [数据体检]/[建议方法]
  3. 子进程跑 template_pipeline.py  -> exit=0 且输出含 按推荐运行/真实数据无真值图/边数
     （模板示例数据无真值图 → 走「仅给出图结构」分支，这也正是「复制即用」默认路径）
  4. 额外：子进程跑一个给入真值图的小驱动，确认「定量评估」分支能产出 report.md +
     metrics.json（合法 JSON，无 NaN 字面量）——防未来改动破坏评估交付链路
  5. 清理临时目录

用法:
  cd causal-lab
  python tests/test_template_portable.py        # 非零退出码 = 回归失败
"""
import sys, os, shutil, subprocess, tempfile, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(ROOT, "experiments", "10_templates")
MARKERS_DATA_GEN = ["[数据体检]", "[建议方法]"]
MARKERS_PIPELINE = ["按推荐运行", "真实数据无真值图", "边数"]

# 给入真值图的小驱动：X1->X2->X3 链，调用模板的 run_with_truth 定量评估
DRIVER = '''# -*- coding: utf-8 -*-
import numpy as np
from template_data_gen import load_your_data
from template_pipeline import run_with_truth
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

data, meta = load_your_data()
truth = np.zeros((3, 3), dtype=int)
truth[0, 1] = truth[1, 2] = 1          # X1->X2->X3
run_with_truth(data, truth, [("PC+fisherz", lambda d: pc(d, 0.05, fisherz, show_progress=False))])
'''


def _run(cwd, script):
    """子进程跑模板脚本；返回 (exit_code, stdout)。"""
    proc = subprocess.run([sys.executable, script],
                          cwd=cwd, capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def _check_markers(output, name, markers):
    missing = [m for m in markers if m not in output]
    if missing:
        raise AssertionError(f"{name} 输出缺失关键标记 {missing}\n--- 输出前 800 字符 ---\n{output[:800]}")


def main():
    assert os.path.isdir(TEMPLATE_DIR), f"找不到模板目录: {TEMPLATE_DIR}"
    tmp = tempfile.mkdtemp(prefix="tpl_portable_")
    driver_path = os.path.join(tmp, "_tpl_eval_driver.py")
    try:
        for fn in os.listdir(TEMPLATE_DIR):
            src = os.path.join(TEMPLATE_DIR, fn)
            if os.path.isfile(src) and fn.endswith(".py"):
                shutil.copy2(src, os.path.join(tmp, fn))
        with open(driver_path, "w", encoding="utf-8") as f:
            f.write(DRIVER)
        print(f"[1/5] 已复制模板到独立临时目录: {tmp}")

        rc, out = _run(tmp, "template_data_gen.py")
        assert rc == 0, f"template_data_gen.py 退出码 {rc}\n{out[:800]}"
        _check_markers(out, "template_data_gen.py", MARKERS_DATA_GEN)
        print("[2/5] template_data_gen.py 独立跑通 (exit=0, 标记齐全)")

        rc, out = _run(tmp, "template_pipeline.py")
        assert rc == 0, f"template_pipeline.py 退出码 {rc}\n{out[:800]}"
        _check_markers(out, "template_pipeline.py", MARKERS_PIPELINE)
        print("[3/5] template_pipeline.py 独立跑通 (exit=0, 默认无真值图路径标记齐全)")

        rc, out = _run(tmp, "_tpl_eval_driver.py")
        assert rc == 0, f"_tpl_eval_driver.py 退出码 {rc}\n{out[:800]}"
        assert "[报告]" in out and "[指标]" in out, f"定量评估分支未打印 [报告]/[指标]\n{out[:800]}"
        print("[4/5] 定量评估分支（给入真值图）独立跑通 (exit=0)")

        metrics_path = os.path.join(tmp, "results", "template_out", "metrics.json")
        assert os.path.exists(metrics_path), f"未找到指标文件: {metrics_path}"
        with open(metrics_path, encoding="utf-8") as f:
            text = f.read()
        assert "NaN" not in text and "Infinity" not in text, "metrics.json 含 NaN/Infinity 字面量"
        json.loads(text)
        print(f"[5/5] 指标文件合法: {os.path.relpath(metrics_path, tmp)}（无 NaN/Infinity）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n模板回归测试通过 [OK]  —— 复制即用能力未损坏")


if __name__ == "__main__":
    main()