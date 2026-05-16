"""
User Intent Vector Calculation

工程统一定义（必须保持一致）：
- user_intent 使用自车坐标系（ego/body frame）
- ux（第0维）: 小车正前方为正方向（forward +X）
- uy（第1维）: 小车左侧90度为正方向（left +Y）
- 默认归一化后为单位向量，模长约为1（目标与车重合时为[0, 0]）
"""
import numpy as np
import torch

def convert_to_ego_frame(dir_vec_env, yaw):
    """
    将环境坐标系方向向量转换为自车坐标系方向向量。

    输入:
    - dir_vec_env: 环境坐标系中的方向向量（机器人位置 -> 目标位置）
    - yaw: 机器人偏航角（环境坐标系）

    输出:
    - dir_vec_ego: 自车坐标系向量 [ux, uy]
      * ux > 0 表示目标在车前方
      * uy > 0 表示目标在车左侧
    """
    # 确保输入是float32，避免精度问题
    dir_vec_env = dir_vec_env.float()
    yaw = yaw.float()
    
    cos_yaw = torch.cos(yaw).cuda()
    sin_yaw = torch.sin(yaw).cuda()
    
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
    计算用户意图向量。

    Returns:
    - dir_vec_env: 环境坐标系方向向量（未归一化）
    - dir_vec_ego: 自车坐标系方向向量（默认归一化，作为模型输入）
    - dist: 机器人到目标距离
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
    
    # 5. 转换到自车坐标系（X前，Y左）
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
    
    # 注意：第二个返回值 dir_vec_ego 才是训练中 user_intent 的标准输入口径
    return dir_vec_env, dir_vec_ego, dist

