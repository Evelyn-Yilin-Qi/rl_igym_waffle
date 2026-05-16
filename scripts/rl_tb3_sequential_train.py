"""
顺序执行 3 场景 × 5 种预训练骨干的 PPO（严格一个接一个，不做多任务并行）。

与 `rl_tb3_matrix_train.py` 共用同一套 worker 与保存路径；编排器对每个组合
阻塞式 `subprocess.call`，上一任务结束后再启动下一任务。

用法（项目根目录，**系统 python3**，勿用 ``isaac_python`` 起编排进程）::

  python3 scripts/rl_tb3_sequential_train.py

若要用 ``rl_tb3_box.py`` / ``door`` / ``col`` 各跑 5 模型（共 15 次、ckpt 在
``ppo_sft_rl_tb3_v3``），请用 ``scripts/rl_tb3_sequential_scene_models_train.py``。

单任务仍用矩阵脚本:
  ./python.sh scripts/rl_tb3_matrix_train.py --worker --scene box --model simple_fc_sft
"""
from __future__ import annotations

import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from rl_tb3_matrix_train import orchestrator_main


if __name__ == "__main__":
    orchestrator_main()
