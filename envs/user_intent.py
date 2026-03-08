"""
User Intent Vector Calculation
计算用户意图向量 (ux, uy)，严格为单位向量：√(ux² + uy²) = 1, ux, uy ∈ [-1, 1]
"""
import numpy as np
import torch

def convert_to_ego_frame(dir_vec_env, yaw):
    """
    修正版：将环境坐标系向量转换为自车坐标系（X前，Y左）
    增加中间值打印，方便调试
    """
    # 确保输入是float32，避免精度问题
    dir_vec_env = dir_vec_env.float()
    yaw = yaw.float()
    
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # 打印关键中间值（调试用）
    # print(f"  旋转参数 - cos(θ): {cos_yaw.item():.6f}, sin(θ): {sin_yaw.item():.6f}")
    # print(f"  环境向量输入: {dir_vec_env.cpu().numpy()}")
    
    # 正确的旋转公式（逐元素计算，避免维度错误）
    dx_env = dir_vec_env[:, 0]
    dy_env = dir_vec_env[:, 1]
    
    dx_ego = dx_env * cos_yaw + dy_env * sin_yaw
    dy_ego = -dx_env * sin_yaw + dy_env * cos_yaw
    
    dir_vec_ego = torch.stack([dx_ego, dy_ego], dim=1)
    # print(f"  旋转后向量: {dir_vec_ego.cpu().numpy()}")
    
    return dir_vec_ego

def compute_user_intent_torch(
    robot_positions: torch.Tensor,
    robot_yaw: torch.Tensor,
    goal_positions: torch.Tensor,
    env_origins: torch.Tensor,
    normalize: bool = True
):
    """
    修正版：计算用户意图向量（增加全流程打印，修复精度问题）
    """
    # 1. 强制转换为float32，避免精度问题
    robot_positions = robot_positions.float()
    goal_positions = goal_positions.float()
    env_origins = env_origins.float()
    robot_yaw = robot_yaw.float()
    
    # 2. 计算环境局部坐标系（打印中间值）
    robot_local = robot_positions - env_origins
    goal_local = goal_positions - env_origins
    # print(f"\n=== 基础坐标计算 ===")
    # print(f"机器人局部坐标: {robot_local.cpu().numpy()}")
    # print(f"目标局部坐标: {goal_local.cpu().numpy()}")
    
    # 3. 环境坐标系方向向量（原始值，未归一化）
    dir_vec_env = goal_local[:, :2] - robot_local[:, :2]
    # print(f"原始环境方向向量: {dir_vec_env.cpu().numpy()}")
    
    # 4. 计算距离（避免除零的安全方式）
    dist = torch.sqrt(torch.sum(dir_vec_env **2, dim=1))  # 替代torch.norm，更稳定
    # print(f"到目标的距离: {dist.item():.6f}")
    
    # 5. 转换到自车坐标系
    # print(f"\n=== 自车坐标系转换 ===")
    dir_vec_ego = convert_to_ego_frame(dir_vec_env, robot_yaw)
    
    # 6. 安全归一化（核心修复：先判断距离是否为0）
    if normalize:
        # print(f"\n=== 归一化处理 ===")
        # 仅对距离>0的向量归一化（避免除零）
        mask = dist > 1e-6
        dir_vec_ego_norm = dir_vec_ego.clone()
        
        # 对有效向量归一化
        if mask.any():
            norm = dist[mask].unsqueeze(1)  # 保持维度 (N,1)
            dir_vec_ego_norm[mask] = dir_vec_ego[mask] / norm
        # 对距离为0的向量，直接设为[0,0]
        dir_vec_ego_norm[~mask] = 0.0
        
        # print(f"  归一化前自车向量: {dir_vec_ego.cpu().numpy()}")
        # print(f"  归一化后自车向量: {dir_vec_ego_norm.cpu().numpy()}")
        dir_vec_ego = dir_vec_ego_norm
    
    return dir_vec_env, dir_vec_ego, dist

