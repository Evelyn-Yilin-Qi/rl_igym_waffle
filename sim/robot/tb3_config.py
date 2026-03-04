"""
TB3 Robot Configuration Constants
TB3 Waffle Pi 机器人物理参数配置
"""
import os

# 机器人 USD 文件路径
# 使用相对于项目根目录的路径，支持不同工作目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TB3_USD = os.path.join(_PROJECT_ROOT, "assets", "turtlebot3_waffle_pi", "waffle_pi.usd")

# 轮子参数
WHEEL_RADIUS = 0.033  # 轮子半径 (m)
WHEEL_BASE = 0.288    # 轮距 (m)，左右轮中心距离（根据 URDF: 0.144 - (-0.144) = 0.288）

# 机器人质量参数
BASE_MASS = 2.2       # 底盘质量 (kg)
COM_X = -0.08         # 质心 X 偏移 (m)
COM_Z = -0.10         # 质心 Z 偏移 (m)

# 轮子关节控制参数
WHEEL_KP = 0.0        # 位置增益
WHEEL_KD = 120.0      # 速度增益
WHEEL_MAX_EFFORT = 80.0  # 最大力矩 (N·m)

# 机器人尺寸（用于碰撞检测）
ROBOT_HALF_LENGTH_X = 0.145  # 机器人半长 (m)
ROBOT_HALF_WIDTH_Y = 0.155   # 机器人半宽 (m)
ROBOT_COLLISION_RADIUS = max(ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y)  # 碰撞半径
