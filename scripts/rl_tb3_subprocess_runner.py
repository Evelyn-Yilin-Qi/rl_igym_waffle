"""
供 ``rl_tb3_*_models_train.py`` 等编排脚本使用：父进程应为「非 Isaac」解释器，
由子进程单独 ``python.sh`` / ``ISAAC_PYTHON`` 拉起 ``rl_tb3_*.py``，避免嵌套 Isaac 卡死。
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional


def assert_plain_python_parent() -> None:
    """若当前解释器已加载 Omniverse/Isaac 相关模块，立即退出并提示正确用法。"""
    bad = [m for m in sys.modules if m.startswith(("omni.", "carb.", "pxr."))]
    if bad:
        print(
            "[编排脚本] 当前进程已加载 Isaac/Omni（例如用了 ``isaac_python`` 启动本脚本），\n"
            "再 ``subprocess`` 起子 Isaac 极易卡死或资源不释放。\n\n"
            "请改用「系统」Python 只跑编排逻辑，由子进程去起仿真，例如：\n"
            "  cd <仓库根>\n"
            "  python3 scripts/rl_tb3_box_models_train.py\n\n"
            "子进程会自动使用环境变量 ``ISAAC_PYTHON``（推荐指向 ``python.sh``），\n"
            "或从 ``sys.executable`` 向上查找 ``python.sh``。\n"
            f"（已检测到模块前缀示例: {bad[:5]}…）\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


def run_isaac_child_blocking(cmd: List[str], *, cwd: str, env: Optional[Dict[str, str]] = None) -> int:
    """
    阻塞运行一条 Isaac 训练命令；返回后 ``gc.collect()``，并按环境变量稍作等待，
    便于 GPU/进程句柄释放（默认几秒，可用 ``RL_ORCHESTRATOR_COOLDOWN_SEC`` 调整）。
    """
    if env is None:
        env = os.environ.copy()
    close_fds = os.name != "nt"
    proc = subprocess.run(cmd, cwd=cwd, env=env, close_fds=close_fds)
    gc.collect()
    cooldown = float(os.environ.get("RL_ORCHESTRATOR_COOLDOWN_SEC", "3"))
    if cooldown > 0:
        time.sleep(cooldown)
    return int(proc.returncode)
