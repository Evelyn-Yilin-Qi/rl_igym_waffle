#!/usr/bin/env bash
# 逐条顺序跑 20 次场景 RL（box / door / col / mixed × 5 model；每条独立 isaac 进程）。
# 用法（仓库根）::
#   chmod +x run_rl_tb3_scene_models_isaac_sequential.sh
#   ./run_rl_tb3_scene_models_isaac_sequential.sh
#
# 默认 Isaac 入口写死为 ``/isaac-sim/python.sh``；若你安装不在此路径，请任选覆盖::
#   export ISAAC_PYTHON_LAUNCHER=/你的/python.sh
#   或 export ISAAC_PYTHON=/你的/python.sh
#
# 任一步失败会 ``set -e`` 中止后续命令。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 默认 Isaac 入口（安装不在 /isaac-sim 时改此变量或设环境变量覆盖）
DEFAULT_ISAAC_PY="/isaac-sim/python.sh"
PY="${ISAAC_PYTHON_LAUNCHER:-${ISAAC_PYTHON:-$DEFAULT_ISAAC_PY}}"
if [[ "$PY" == */* ]]; then
  if [[ ! -x "$PY" ]]; then
    echo "Isaac 启动器不存在或不可执行: $PY" >&2
    echo "请修改本脚本内默认路径，或执行: export ISAAC_PYTHON_LAUNCHER=/你的/python.sh" >&2
    exit 127
  fi
else
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "未在 PATH 中找到启动器: $PY" >&2
    exit 127
  fi
fi

_run() {
  echo "================================================================================"
  echo "RUN: $PY $*"
  echo "================================================================================"
  "$PY" "$@"
}

# # --- box ---
# _run "$ROOT/scripts/rl_tb3_box.py" --model simple_fc_sft
# _run "$ROOT/scripts/rl_tb3_box.py" --model cnn_lstm_sft
# _run "$ROOT/scripts/rl_tb3_box.py" --model cnn_gru_sft
# _run "$ROOT/scripts/rl_tb3_box.py" --model fc_lstm_sft
# _run "$ROOT/scripts/rl_tb3_box.py" --model cnn_lstm_sft_nodoor

# # --- door ---
# _run "$ROOT/scripts/rl_tb3_door.py" --model simple_fc_sft
# _run "$ROOT/scripts/rl_tb3_door.py" --model cnn_lstm_sft
# _run "$ROOT/scripts/rl_tb3_door.py" --model cnn_gru_sft
# _run "$ROOT/scripts/rl_tb3_door.py" --model fc_lstm_sft
# _run "$ROOT/scripts/rl_tb3_door.py" --model cnn_lstm_sft_nodoor

# # --- cylinder（rl_tb3_col.py）---
# _run "$ROOT/scripts/rl_tb3_col.py" --model simple_fc_sft
# _run "$ROOT/scripts/rl_tb3_col.py" --model cnn_lstm_sft
# _run "$ROOT/scripts/rl_tb3_col.py" --model cnn_gru_sft
# _run "$ROOT/scripts/rl_tb3_col.py" --model fc_lstm_sft
# _run "$ROOT/scripts/rl_tb3_col.py" --model cnn_lstm_sft_nodoor

# --- mixed（四场景循环，多环境）---
_run "$ROOT/scripts/rl_tb3_mixed.py" --model simple_fc_sft
_run "$ROOT/scripts/rl_tb3_mixed.py" --model cnn_lstm_sft
_run "$ROOT/scripts/rl_tb3_mixed.py" --model cnn_gru_sft
_run "$ROOT/scripts/rl_tb3_mixed.py" --model fc_lstm_sft
_run "$ROOT/scripts/rl_tb3_mixed.py" --model cnn_lstm_sft_nodoor

echo "全部 20 条命令已顺序执行完毕。"
