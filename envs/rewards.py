"""
Reward Function Calculator (论文 Method 2)
实现所有奖励组件，并返回各组件值用于 Tensorboard 记录
"""
import torch
import numpy as np
from .user_intent import compute_user_intent_torch


class RewardCalculator:
    """
    奖励函数计算器，实现论文 Method 2 的所有组件
    """
    def __init__(self, num_envs: int, device: str, **kwargs):
        self.num_envs = num_envs
        self.device = device
        
        # vx Tracking 参数
        self.ra = kwargs.get("ra", 0.5)
        self.rl = kwargs.get("rl", -0.5)
        
        # Heading 参数
        self.rh = kwargs.get("rh", -0.5)
        self.phi_thresh = kwargs.get("phi_thresh", 0.2)  # radians
        
        # Obstacles 参数
        self.rc = kwargs.get("rc", -1.0)
        self.rcrit = kwargs.get("rcrit", -1.0)
        self.rcol = kwargs.get("rcol", -100.0)
        self.d_col = kwargs.get("d_col", 0.12)
        self.d_crit = kwargs.get("d_crit", 0.35)
        
        # Smoothing 参数（使用 ras 和 rc）
        self.ras = kwargs.get("ras", -0.02)
        # 注意：rc 同时用于障碍物（临界区）和二阶平滑，使用同一个值
        self.rc_smooth = self.rc  # 二阶平滑使用与障碍物相同的 rc
        
        # Centripetal 参数（可选）
        self.enable_centripetal = kwargs.get("enable_centripetal", False)
        self.w_centripetal = kwargs.get("w_centripetal", 0.0)
        
        # 速度限制（用于归一化）
        self.max_v = kwargs.get("max_v", 0.5)
        self.max_w = kwargs.get("max_w", 1.0)
        
        # 用于存储各组件奖励（用于 Tensorboard）
        self.reward_components = {}
    
    def compute_rewards(
        self,
        lidar_ranges: torch.Tensor,
        robot_positions: torch.Tensor,
        robot_velocities: torch.Tensor,
        robot_orientations: torch.Tensor,
        goal_positions: torch.Tensor,
        env_origins: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor,
        user_intent_env: torch.Tensor = None  # 可选：如果已计算则直接使用，避免重复计算 (N, 2)
    ) -> torch.Tensor:
        """
        计算总奖励
        
        Args:
            lidar_ranges: (N, 36) LiDAR 距离测量
            robot_positions: (N, 3) 机器人位置
            robot_velocities: (N, 6) 机器人速度 [vx, vy, vz, wx, wy, wz]
            robot_orientations: (N, 4) 机器人朝向（四元数）
            goal_positions: (N, 3) 目标位置
            env_origins: (N, 3) 环境原点
            actions: (N, 2) 当前动作 [v, w]
            action_history: (N, 4) 动作历史 [v_t-1, w_t-1, v_t-2, w_t-2]
        
        Returns:
            total_reward: (N,) 总奖励
        """
        # 如果未传入 user_intent，则计算一次（用于 heading 和 vx_tracking）
        if user_intent_env is None:
            _, user_intent_env, _ = compute_user_intent_torch(
                robot_positions, robot_orientations, goal_positions, env_origins, normalize=True
            )
        
        # 计算各组件奖励
        r_obstacles = self._compute_obstacles_reward(lidar_ranges)
        r_heading = self._compute_heading_reward(
            robot_positions, robot_orientations, goal_positions, env_origins, user_intent_env=user_intent_env
        )
        r_vx_tracking = self._compute_vx_tracking_reward(
            robot_positions, robot_velocities, robot_orientations, goal_positions, env_origins, user_intent_env=user_intent_env
        )
        r_smoothing_1st = self._compute_smoothing_1st(actions, action_history)
        r_smoothing_2nd = self._compute_smoothing_2nd(actions, action_history)
        r_centripetal = self._compute_centripetal_reward(robot_velocities)
        
        # 存储各组件（用于 Tensorboard）
        self.reward_components = {
            "obstacles": r_obstacles,
            "heading": r_heading,
            "vx_tracking": r_vx_tracking,
            "smoothing_1st": r_smoothing_1st,
            "smoothing_2nd": r_smoothing_2nd,
            "centripetal": r_centripetal
        }
        
        # 总奖励
        total_reward = (
            r_obstacles + r_heading + r_vx_tracking + 
            r_smoothing_1st + r_smoothing_2nd + r_centripetal
        )
        
        return total_reward
    
    def _compute_obstacles_reward(self, lidar_ranges: torch.Tensor) -> torch.Tensor:
        """
        障碍物奖励（分段二次函数，根据论文 Method 2）
        
        R_obs = {
            rc + rcrit * (dcrit - d)^2,  if d < dcrit
            rcol,                         if d < dcol
            0,                            else
        }
        """
        # 找到最小距离（最近障碍物）
        d_min = torch.min(lidar_ranges, dim=1)[0]  # (N,)
        
        # 分段函数
        mask_collision = d_min < self.d_col
        mask_critical = (d_min >= self.d_col) & (d_min < self.d_crit)
        mask_safe = d_min >= self.d_crit
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 碰撞区：rcol
        reward[mask_collision] = self.rcol
        
        # 临界区：rc + rcrit * (dcrit - d)^2
        if mask_critical.any():
            d_critical = d_min[mask_critical]
            reward[mask_critical] = self.rc + self.rcrit * (self.d_crit - d_critical) ** 2
        
        # 安全区：0（无奖励）
        reward[mask_safe] = 0.0
        
        return reward
    
    def _compute_heading_reward(
        self,
        robot_positions: torch.Tensor,
        robot_orientations: torch.Tensor,
        goal_positions: torch.Tensor,
        env_origins: torch.Tensor,
        user_intent_env: torch.Tensor = None  # 可选：如果已计算则直接使用
    ) -> torch.Tensor:
        """
        航向奖励（带阈值的二次函数）
        
        R_heading = {
            rh * phi^2,    if |phi| > phi_thresh
            0,              otherwise
        }
        
        phi = atan2(u_y, u_x) - robot_yaw
        """
        # 使用传入的 user_intent 或重新计算
        if user_intent_env is None:
            _, u_env, _ = compute_user_intent_torch(
                robot_positions, robot_orientations, goal_positions, env_origins, normalize=True
            )
        else:
            u_env = user_intent_env
        u_x_env = u_env[:, 0]
        u_y_env = u_env[:, 1]
        
        # 目标航向角（环境坐标系）
        theta_target = torch.atan2(u_y_env, u_x_env)
        
        # 机器人当前航向角（从四元数提取）
        # 四元数转欧拉角（只取 yaw）
        q = robot_orientations  # (N, 4) [w, x, y, z]
        yaw = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        )
        
        # 航向误差
        phi = theta_target - yaw
        #  wrap to [-π, π]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        
        # 应用阈值
        abs_phi = torch.abs(phi)
        is_above_thresh = abs_phi > self.phi_thresh
        
        reward = torch.where(
            is_above_thresh,
            self.rh * (abs_phi ** 2),
            torch.tensor(0.0, device=self.device)
        )
        
        return reward
    
    def _compute_vx_tracking_reward(
        self,
        robot_positions: torch.Tensor,
        robot_velocities: torch.Tensor,
        robot_orientations: torch.Tensor,
        goal_positions: torch.Tensor,
        env_origins: torch.Tensor,
        user_intent_env: torch.Tensor = None  # 可选：如果已计算则直接使用
    ) -> torch.Tensor:
        """
        vx 跟踪奖励（指数函数）
        
        R_vx = ra * exp(rl * (vx_norm - ux)^2)
        
        其中：
        - vx_norm = vx / max_v (归一化)
        - ux 是用户意图在机器人局部 x 轴上的投影
        """
        # 使用传入的 user_intent 或重新计算
        if user_intent_env is None:
            _, u_env, _ = compute_user_intent_torch(
                robot_positions, robot_orientations, goal_positions, env_origins, normalize=True
            )
        else:
            u_env = user_intent_env
        u_x_env = u_env[:, 0]
        u_y_env = u_env[:, 1]
        
        # 机器人当前航向角
        q = robot_orientations
        theta = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        )
        
        # 将用户意图从环境坐标系转换到机器人局部坐标系
        # ux = u_x_env * cos(theta) + u_y_env * sin(theta)
        ux = u_x_env * torch.cos(theta) + u_y_env * torch.sin(theta)
        
        # 归一化 vx
        vx = robot_velocities[:, 0]
        vx_norm = vx / self.max_v
        
        # 计算误差平方
        error_sq = (vx_norm - ux) ** 2
        
        # 指数奖励
        reward = self.ra * torch.exp(self.rl * error_sq)
        
        return reward
    
    def _compute_smoothing_1st(
        self,
        actions: torch.Tensor,
        action_history: torch.Tensor
    ) -> torch.Tensor:
        """
        一阶平滑奖励
        
        R_smooth1 = weight * (a_t - a_{t-1})^2
        """
        a_t = actions  # (N, 2) [v, w]
        a_t_minus_1 = action_history[:, :2]  # (N, 2) [v_t-1, w_t-1]
        
        diff = a_t - a_t_minus_1
        reward = self.ras * torch.sum(diff ** 2, dim=1)
        
        return reward
    
    def _compute_smoothing_2nd(
        self,
        actions: torch.Tensor,
        action_history: torch.Tensor
    ) -> torch.Tensor:
        """
        二阶平滑奖励
        
        R_smooth2 = rc * (a_t - 2*a_{t-1} + a_{t-2})^2
        """
        a_t = actions  # (N, 2)
        a_t_minus_1 = action_history[:, :2]  # (N, 2)
        a_t_minus_2 = action_history[:, 2:]  # (N, 2)
        
        diff = a_t - 2.0 * a_t_minus_1 + a_t_minus_2
        reward = self.rc_smooth * torch.sum(diff ** 2, dim=1)
        
        return reward
    
    def _compute_centripetal_reward(self, robot_velocities: torch.Tensor) -> torch.Tensor:
        """
        向心加速度惩罚（可选）
        
        R_centripetal = -w_centripetal * |v * w|
        """
        if not self.enable_centripetal:
            return torch.zeros(self.num_envs, device=self.device)
        
        v = robot_velocities[:, 0]  # 线速度
        w = robot_velocities[:, 5]  # 角速度（绕 z 轴）
        
        centripetal = -self.w_centripetal * torch.abs(v * w)
        
        return centripetal
