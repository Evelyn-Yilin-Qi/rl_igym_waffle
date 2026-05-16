#!/usr/bin/env bash
# 顺序跑 3×5 PPO（见 scripts/rl_tb3_sequential_train.py）。
#
# 编排进程必须用「非 Isaac」解释器（默认 python3），否则会与子 Isaac 冲突卡死。
# 子进程仍由 rl_tb3_matrix_train 根据 ISAAC_PYTHON / python.sh 拉起。
#
#   ./run_rl_tb3_sequential.sh
#
# 指定编排用解释器（例如 conda）::
#   export ORCHESTRATOR_PYTHON=/path/to/python3
#   ./run_rl_tb3_sequential.sh
#
# 子仿真入口（传给子进程）::
#   export ISAAC_PYTHON=/isaac-sim/python.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYBIN="${ORCHESTRATOR_PYTHON:-python3}"
exec "$PYBIN" "$ROOT/scripts/rl_tb3_sequential_train.py" "$@"
