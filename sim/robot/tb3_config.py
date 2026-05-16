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

# 机器人控制上限（训练/归一化统一口径）
MAX_V = 1.0  # 线速度上限 (m/s)
MAX_W = 1.0  # 角速度上限 (rad/s)

# 机器人质量参数
BASE_MASS = 1.8       # 底盘质量 (kg)
# MassAPI 质心相对 base_link（m）。URDF base_link 惯性原点为 0,0,0；勿随意前移/下移，
# 否则易与后部万向轮载荷耦合，产生难以消除的直行偏航（常见为缓慢向一侧弯）。
COM_X = -0.08
COM_Z = -0.10

# base_link 主惯量 (kg·m²)，来自 assets/.../turtlebot3_waffle_pi.urdf（仅用对角项 ixx, iyy, izz）
BASE_LINK_DIAGONAL_INERTIA = (8.7002718e-3, 8.6195418e-3, 1.4612727e-2)

# 轮子关节控制参数（略降 KD、略提力矩上限，便于差速转向跟手；可按仿真再调）
WHEEL_KP = 0.0
WHEEL_KD = 75.0
WHEEL_MAX_EFFORT = 150.0

# 机器人尺寸（用于碰撞检测）
ROBOT_HALF_LENGTH_X = 0.145  # 机器人半长 (m)
ROBOT_HALF_WIDTH_Y = 0.155   # 机器人半宽 (m)
ROBOT_COLLISION_RADIUS = max(ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y)  # 碰撞半径
