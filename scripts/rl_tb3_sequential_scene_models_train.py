"""
顺序执行 20 次场景 RL（``box`` / ``door`` / ``col`` / ``mixed`` × 5 个 ``--model``），严格阻塞：
上一条子进程完全退出并经过短暂冷却后，再启动下一条。

**不要用 ``isaac_python`` 运行本编排脚本**；请用系统 ``python3``，子进程再用 ``ISAAC_PYTHON`` /
``python.sh`` 起 ``rl_tb3_*.py``（见 ``rl_tb3_subprocess_runner``）。

仓库根目录::

    python3 scripts/rl_tb3_sequential_scene_models_train.py

冷却秒数：环境变量 ``RL_ORCHESTRATOR_COOLDOWN_SEC``（默认 3）。
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

SCENE_SCRIPTS = (
    "rl_tb3_box.py",
    "rl_tb3_door.py",
    "rl_tb3_col.py",
    "rl_tb3_mixed.py",
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
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    for scene_script in SCENE_SCRIPTS:
        for model in MODELS:
            ck = os.path.join(PROJECT_ROOT, f"best_actor_{model}.pth")
            if not os.path.isfile(ck):
                print(f"[SKIP] {scene_script} model={model} 缺少预训练: {ck}")
                continue
            script_path = os.path.join(scripts_dir, scene_script)
            cmd = [py, script_path, "--model", model]
            print("\n" + "=" * 72)
            print("RUN:", " ".join(cmd))
            if py != sys.executable:
                print(f"(子进程启动器 != 当前 python；当前 sys.executable={sys.executable})")
            print("=" * 72)
            ret = run_isaac_child_blocking(cmd, cwd=PROJECT_ROOT)
            if ret != 0:
                raise SystemExit(f"子进程失败 exit={ret} script={scene_script} model={model}")


if __name__ == "__main__":
    main()
