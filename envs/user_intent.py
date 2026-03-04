"""
User Intent Vector Calculation
计算用户意图向量 (ux, uy)，严格为单位向量：√(ux² + uy²) = 1, ux, uy ∈ [-1, 1]
"""
import numpy as np
import torch


def compute_user_intent_env(simulator, normalize: bool = True):
    """
    计算用户意图向量（从环境坐标系到机器人局部坐标系）
    
    Args:
        simulator: Isaac Gym 模拟器对象，包含机器人位置、朝向和目标位置
        normalize: 是否归一化为单位向量（默认 True）
    
    Returns:
        If normalize=True:
            dir_vec: (N, 2) 原始方向向量
            dir_unit: (N, 2) 单位向量 (ux, uy)，满足 √(ux² + uy²) = 1, ux, uy ∈ [-1, 1]
            dist: (N,) 距离
        If normalize=False:
            dir_vec: (N, 2) 原始方向向量
            dist: (N,) 距离
    """
    # 获取机器人位置和目标位置（环境坐标系）
    robot_positions = simulator.robot_positions  # (N, 3)
    robot_orientations = simulator.robot_orientations  # (N, 4) quaternion
    goal_positions = simulator.target_positions  # (N, 3)
    env_origins = simulator.env_origins  # (N, 3)
    
    # 转换到环境局部坐标系
    robot_local = robot_positions - env_origins
    goal_local = goal_positions - env_origins
    
    # 计算方向向量（环境坐标系）
    dir_vec = (goal_local[:, :2] - robot_local[:, :2]).astype(np.float32)
    dist = np.linalg.norm(dir_vec, axis=1)
    
    if not normalize:
        return dir_vec, dist
    
    # 归一化为单位向量
    norm = np.maximum(dist, 1e-8)
    dir_unit = (dir_vec / norm[:, None]).astype(np.float32)
    
    return dir_vec, dir_unit, dist


def compute_user_intent_torch(robot_positions, robot_orientations, goal_positions, env_origins, normalize: bool = True):
    """
    PyTorch 版本的用户意图向量计算
    
    Args:
        robot_positions: (N, 3) 机器人位置
        robot_orientations: (N, 4) 机器人朝向（四元数）
        goal_positions: (N, 3) 目标位置
        env_origins: (N, 3) 环境原点
        normalize: 是否归一化为单位向量
    
    Returns:
        If normalize=True:
            dir_vec: (N, 2) 原始方向向量
            dir_unit: (N, 2) 单位向量 (ux, uy)
            dist: (N,) 距离
        If normalize=False:
            dir_vec: (N, 2) 原始方向向量
            dist: (N,) 距离
    """
    # 转换到环境局部坐标系
    robot_local = robot_positions - env_origins
    goal_local = goal_positions - env_origins
    
    # 计算方向向量（环境坐标系）
    dir_vec = goal_local[:, :2] - robot_local[:, :2]
    dist = torch.norm(dir_vec, dim=1)
    
    if not normalize:
        return dir_vec, dist
    
    # 归一化为单位向量
    norm = torch.clamp(dist, min=1e-8)
    dir_unit = dir_vec / norm.unsqueeze(1)
    
    return dir_vec, dir_unit, dist
