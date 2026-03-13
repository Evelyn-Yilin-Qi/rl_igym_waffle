#!/bin/bash
# 复制模型文件到tb3_model_package目录

# 默认模型路径
DEFAULT_MODEL="../checkpoints/ppo_stage2_test_modular/ppo_final_20260312_16.pth"

# 如果提供了参数，使用参数作为模型路径
MODEL_PATH=${1:-$DEFAULT_MODEL}

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo ""
    echo "使用方法:"
    echo "  ./copy_model.sh [模型文件路径]"
    echo ""
    echo "示例:"
    echo "  ./copy_model.sh ../checkpoints/ppo_stage2_test_modular/ppo_final_20260312_16.pth"
    exit 1
fi