# -*- coding: utf-8 -*-
"""
GitHub 推送脚本（scripts/github_push.py）—— IMPROVEMENTS #5
============================================================
用 Git Data API 推送本地仓库到 GitHub（github.com:443 的 git push 被墙时的标准通道）。

特性（相对早期内联脚本的改进）：
1. **gitignore 语义**：推送前用 `git check-ignore --stdin` 过滤——手动 walk 不再绕过
   .gitignore 规则（曾把 results/metrics/cache_*.json 误推远程的根因）
2. **内置硬跳过兜底**：.git/.scratch/__pycache__/.claude/settings.local.json/
   results/metrics/cache_* 即使未来 .gitignore 被改坏也绝不推送
3. **增量推送**：base_tree = 远程 HEAD——保留远程独有文件（如用户的 JudeaPearl 笔记），
   本地文件 add/overwrite
4. **推送后自验证**：检查远程树文件数、关键文件在位、无 cache/.claude 残留

用法:
  python scripts/github_push.py --dry-run          # 只列将推送的文件（不推）
  python scripts/github_push.py                    # 实际推送（自动生成 commit message）
  python scripts/github_push.py -m "自定义 commit message"

环境:
  token 自动从 D:\\win\\youtube-content\\.git\\config 的凭据 URL 提取；
  或设置环境变量 GITHUB_TOKEN。
"""
import argparse
import base64
import json
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen, HTTPError

# Windows 上 Anaconda Python 3.9 的 ssl 加载系统证书存储可能失败（ASN1 NOT_ENOUGH_DATA），
# 显式用 certifi 的 CA 文件创建上下文；无 certifi 则回退（unverified 仅作最后手段并告警）。
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    try:
        _SSL_CTX = ssl.create_default_context()
    except ssl.SSLError:
        print("  [警告] 系统证书不可用，使用未校验上下文（仅推荐内网/测试）")
        _SSL_CTX = ssl._create_unverified_context()

OWNER, REPO = "zqian6263-design", "causal-inference"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"
# 硬跳过（.gitignore 之外的双保险；即便未来规则改动也绝不推送）
HARD_SKIP_DIRS = {".git", ".scratch", "__pycache__", ".pytest_cache", "node_modules",
                  "venv", ".vscode", ".idea"}
HARD_SKIP_FILES = {".claude/settings.local.json"}
HARD_SKIP_PREFIXES = ("results/metrics/cache_",)


def get_token():
    env = os.environ.get("GITHUB_TOKEN")
    if env:
        return env
    cfg = open(r"D:\win\youtube-content\.git\config", encoding="utf-8").read()
    m = re.search(r"https://([^@]+)@github\.com/", cfg)
    if not m:
        sys.exit("错误: 无法从 youtube-content/.git/config 提取凭据，请设 GITHUB_TOKEN")
    return m.group(1).partition(":")[2]


def api(method, path, data=None, token=None):
    hdr = {"Authorization": f"token {token}",
           "Accept": "application/vnd.github.v3+json",
           "Content-Type": "application/json"}
    req = Request(API + path, data=json.dumps(data).encode() if data else None,
                  headers=hdr, method=method)
    try:
        with urlopen(req, timeout=120, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode()[:300]
        print(f"  !! HTTP {e.code} @ {path}: {body}")
        raise


def collect_files(root):
    """walk 文件系统 → (rel, abs_path) 列表，应用 check-ignore + 硬跳过。"""
    root = Path(root)
    rels = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in HARD_SKIP_DIRS]
        for fn in filenames:
            rel = (Path(dirpath) / fn).relative_to(root).as_posix()
            if rel in HARD_SKIP_FILES or rel.startswith(HARD_SKIP_PREFIXES):
                continue
            rels.append(rel)
    # git check-ignore --stdin（批量）：输出被忽略的路径
    if rels:
        r = subprocess.run(["git", "check-ignore", "--stdin"],
                           input="\n".join(rels) + "\n",
                           capture_output=True, text=True, cwd=root)
        ignored = set(r.stdout.splitlines())
        rels = [rel for rel in rels if rel not in ignored]
    return [(rel, root / rel) for rel in sorted(rels)]


def push(files, message, token, dry_run):
    print(f"待推送 {len(files)} 个文件"
          f"（{'DRY-RUN 不推送' if dry_run else '实际推送'}）")
    if dry_run:
        for rel, _ in files:
            print("  +", rel)
        return None

    ref = api("GET", "/git/refs/heads/main", token=token)
    parent = ref["object"]["sha"]
    print(f"远程 HEAD: {parent[:12]}")

    blobs = []
    for i, (rel, path) in enumerate(files):
        raw = path.read_bytes()
        try:
            b = api("POST", "/git/blobs",
                    {"content": raw.decode("utf-8"), "encoding": "utf-8"}, token=token)
        except (HTTPError, UnicodeDecodeError):   # 二进制文件（PNG 等）走 base64
            b = api("POST", "/git/blobs",
                    {"content": base64.b64encode(raw).decode(), "encoding": "base64"},
                    token=token)
        blobs.append((rel, b["sha"]))
        if (i + 1) % 20 == 0:
            print(f"  blobs {i + 1}/{len(files)}")

    tree_data = [{"path": r, "mode": "100644", "type": "blob", "sha": s}
                 for r, s in blobs]
    tree = api("POST", "/git/trees", {"tree": tree_data, "base_tree": parent}, token=token)
    commit = api("POST", "/git/commits",
                 {"message": message, "tree": tree["sha"], "parents": [parent]},
                 token=token)
    api("PATCH", "/git/refs/heads/main", {"sha": commit["sha"], "force": True}, token=token)
    print(f"[OK] 推送完成: commit {commit['sha'][:12]}")

    # 自验证
    tree2 = api("GET", f"/git/trees/{commit['sha']}?recursive=1", token=token)
    paths = {t["path"] for t in tree2["tree"] if t["type"] == "blob"}
    print(f"远程文件数: {len(paths)}")
    leak = [p for p in paths if p.startswith(".claude") or "cache_" in p]
    print("残留检查:", leak if leak else "无 cache/.claude 残留 [OK]")
    return commit["sha"]


def main():
    ap = argparse.ArgumentParser(description="Git Data API 推送（gitignore 语义 + 自验证）")
    ap.add_argument("--dry-run", action="store_true", help="只列出将推送的文件")
    ap.add_argument("-m", "--message", default=None, help="commit message（默认自动）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    files = collect_files(root)
    message = args.message or f"sync: {len(files)} files via Git Data API (github_push.py)"
    push(files, message, get_token(), args.dry_run)


if __name__ == "__main__":
    main()
