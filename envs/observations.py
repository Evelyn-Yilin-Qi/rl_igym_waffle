"""
Observation Space Assembly
组装 44 维观察向量：LiDAR[36] + User Input[2] + Base Velocity[2] + Action History[4]
Base Velocity 是测量的当前线速度和角速度 (v, w)
"""
import numpy as np
import torch
from .user_intent import compute_user_intent_torch


def assemble_observations(
    robot_positions: np.ndarray,
    robot_orientations: np.ndarray,
    goal_positions: np.ndarray,
    env_origins: np.ndarray,
    lidar_ranges: np.ndarray,
    base_vel: np.ndarray,
    action_history: np.ndarray,
    max_v: float = 0.5,
    max_w: float = 1.0,
    lidar_max_range: float = 3.0,
    user_input: np.ndarray = None  # 可选：如果已计算则直接使用，避免重复计算
) -> np.ndarray:
    """
    组装 44 维观察向量
    
    Args:
        robot_positions: (N, 3) 机器人位置
        robot_orientations: (N, 4) 机器人朝向（四元数）
        goal_positions: (N, 3) 目标位置
        env_origins: (N, 3) 环境原点
        lidar_ranges: (N, 36) LiDAR 距离测量值
        base_vel: (N, 2) 归一化的测量基础速度 [v_norm, w_norm]
                  v: 测量的线速度，w: 测量的角速度
        action_history: (N, 4) 最近 2 个动作 [v_t-1, w_t-1, v_t-2, w_t-2]
        max_v: 最大线速度（用于归一化）
        max_w: 最大角速度（用于归一化）
        lidar_max_range: LiDAR 最大测量范围（用于归一化）
    
    Returns:
        obs: (N, 44) 观察向量
            - [0:36]: LiDAR 归一化距离 [0, 1]
            - [36:38]: User Input 单位向量 (ux, uy) ∈ [-1, 1], √(ux² + uy²) = 1
            - [38:40]: Base Velocity 归一化 (v_norm, w_norm) ∈ [-1, 1]
                      测量的当前线速度 v 和角速度 w
            - [40:44]: Action History 归一化 (v_t-1, w_t-1, v_t-2, w_t-2) ∈ [-1, 1]
    """
    num_envs = lidar_ranges.shape[0]
    
    # 1. LiDAR 归一化 [0, 1]
    lidar_normalized = np.clip(lidar_ranges / lidar_max_range, 0.0, 1.0).astype(np.float32)
    
    # 2. User Input 单位向量 (ux, uy) ∈ [-1, 1]
    if user_input is None:
        # 如果未传入，则内部计算
        _, user_input_unit, _ = compute_user_intent_torch(
            torch.from_numpy(robot_positions),
            torch.from_numpy(robot_orientations),
            torch.from_numpy(goal_positions),
            torch.from_numpy(env_origins),
            normalize=True
        )
        user_input = user_input_unit.numpy().astype(np.float32)  # (N, 2)
    else:
        # 使用传入的 user_input（确保是 float32）
        user_input = np.asarray(user_input, dtype=np.float32)
    
    # 3. Base Velocity 归一化（已传入，直接使用）
    # base_vel 已经是归一化的 [v_norm, w_norm]，测量的线速度 v 和角速度 w
    base_vel_norm = base_vel.copy()
    
    # 4. Action History 归一化
    action_history_normalized = action_history.copy()
    action_history_normalized[:, 0] /= max_v  # v 归一化
    action_history_normalized[:, 1] /= max_w   # w 归一化
    action_history_normalized = np.clip(action_history_normalized, -1.0, 1.0).astype(np.float32)
    
    # 拼接所有观察
    obs = np.concatenate([
        lidar_normalized,      # (N, 36)
        user_input,            # (N, 2)
        base_vel_norm,         # (N, 2)
        action_history_normalized  # (N, 2)
    ], axis=1).astype(np.float32)
    
    return obs


def get_base_velocity_from_tensor(velocities: torch.Tensor, max_v: float = 0.5, max_w: float = 1.0) -> np.ndarray:
    """
    从速度张量获取归一化的测量基础速度
    
    Base Velocity 是测量的当前线速度和角速度，用于作为 RL 模型的输入观察
    TB3 差速驱动机器人只有线速度 v 和角速度 w，没有 vx/vy 的概念
    
    Args:
        velocities: (N, 6) 速度张量 [vx, vy, vz, wx, wy, wz]
                   对于 TB3：vx 是线速度 v（机器人局部坐标系 x 方向），vy=0，wz 是角速度 w
        max_v: 最大线速度（用于归一化）
        max_w: 最大角速度（用于归一化）
    
    Returns:
        base_vel: (N, 2) 归一化的测量基础速度 [v_norm, w_norm]
                  v_norm: 归一化的测量线速度 ∈ [-1, 1]
                  w_norm: 归一化的测量角速度 ∈ [-1, 1]
    """
    # 提取测量的线速度 v 和角速度 w
    # 对于 TB3：vx 就是线速度 v（机器人局部坐标系 x 方向是前进方向）
    # wz 就是角速度 w（绕 z 轴旋转）
    v = velocities[:, 0].cpu().numpy()  # 线速度
    w = velocities[:, 5].cpu().numpy()  # 角速度
    
    # 归一化
    v_norm = np.clip(v / max_v, -1.0, 1.0)
    w_norm = np.clip(w / max_w, -1.0, 1.0)
    
    base_vel = np.stack([v_norm, w_norm], axis=1).astype(np.float32)
    return base_vel
