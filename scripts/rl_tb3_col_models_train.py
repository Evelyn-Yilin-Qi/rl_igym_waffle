"""
固定场景 cylinder：依次用 5 个预训练 ``best_actor_*.pth`` 各跑一遍 ``rl_tb3_col.py``（每轮新进程）。

**不要用 ``isaac_python`` 运行本编排脚本**（见 ``rl_tb3_subprocess_runner`` 说明）。请用::

    python3 scripts/rl_tb3_col_models_train.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_scripts = str(Path(__file__).resolve().parent)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from rl_tb3_subprocess_runner import assert_plain_python_parent, run_isaac_child_blocking

MODELS = (
    "simple_fc_sft",
    "cnn_lstm_sft",
    "cnn_gru_sft",
    "fc_lstm_sft",
    "cnn_lstm_sft_nodoor",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_worker_python() -> str:
    override = (os.environ.get("ISAAC_PYTHON") or "").strip()
    if override:
        return override
    exe = Path(sys.executable).resolve()
    d = exe.parent
    for _ in range(16):
        if d == d.parent:
            break
        cand = d / "python.sh"
        if cand.is_file():
            return str(cand)
        d = d.parent
    return str(exe)


def main() -> None:
    assert_plain_python_parent()
    os.chdir(PROJECT_ROOT)
    py = resolve_worker_python()
    script = os.path.join(PROJECT_ROOT, "scripts", "rl_tb3_col.py")
    for model in MODELS:
        ck = os.path.join(PROJECT_ROOT, f"best_actor_{model}.pth")
        if not os.path.isfile(ck):
            print(f"[SKIP] 缺少预训练: {ck}")
            continue
        cmd = [py, script, "--model", model]
        print("\n" + "=" * 72)
        print("RUN:", " ".join(cmd))
        if py != sys.executable:
            print(f"(子进程启动器 != 当前 python；当前 sys.executable={sys.executable})")
        print("=" * 72)
        ret = run_isaac_child_blocking(cmd, cwd=PROJECT_ROOT)
        if ret != 0:
            raise SystemExit(f"子进程失败 exit={ret} model={model}")


if __name__ == "__main__":
    main()
