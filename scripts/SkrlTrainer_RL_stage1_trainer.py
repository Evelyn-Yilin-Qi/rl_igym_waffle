"""
RL Stage 1 训练脚本 - 使用 skrl Trainer
基于 3_RL_stage1_test.py，改用 skrl 的 Trainer 管理训练循环
4个empty场景，使用RL策略训练，包含完整的PPO训练流程
"""
import os
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

# 添加项目根目录到 Python 路径（必须在导入其他模块之前）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omni.isaac.kit import SimulationApp

# 初始化 Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.physx import get_physx_scene_query_interface
from pxr import Gf

# 导入 sim 模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)

# 导入观察模块
from envs.observations import assemble_observations, get_base_velocity_from_tensor
from envs.user_intent import compute_user_intent_torch
from envs.rewards import RewardCalculator

# 导入 RL 组件
from models.cnn_lstm_policy_base import WaffleCNNLSTMPolicyBase
from agents.ppo_agent_base import PPOAgentBase
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from gym import spaces

# 导入 Tensorboard
from torch.utils.tensorboard import SummaryWriter


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """从配置文件加载参数"""
    # 如果路径是相对路径，基于项目根目录
    if not os.path.isabs(cfg_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        cfg_path = os.path.join(project_root, cfg_path)
    cfg = OmegaConf.load(cfg_path)
    return cfg


def compute_lidar_ranges(robot_positions, robot_orientations, env_origins, 
                         lidar_num_rays, lidar_max_range, scene_manager):
    """
    计算真实的 LiDAR 范围（使用 ray casting）
    
    Args:
        robot_positions: (N, 3) 机器人世界坐标位置
        robot_orientations: (N, 4) 机器人朝向（四元数 wxyz）
        env_origins: (N, 3) 环境原点
        lidar_num_rays: LiDAR 射线数量
        lidar_max_range: LiDAR 最大测量范围
        scene_manager: SceneManager 实例，用于获取障碍物信息
    
    Returns:
        lidar_ranges: (N, lidar_num_rays) LiDAR 距离测量值
    """
    num_envs = robot_positions.shape[0]
    lidar_ranges = np.ones((num_envs, lidar_num_rays), dtype=np.float32) * lidar_max_range
    
    # 获取 PhysX 场景查询接口
    physx_interface = get_physx_scene_query_interface()
    
    # LiDAR 高度（机器人中心高度）
    lidar_height = 0.1  # 10cm，略高于地面
    
    for env_id in range(num_envs):
        robot_pos = robot_positions[env_id]
        robot_rot = robot_orientations[env_id]
        env_origin = env_origins[env_id]
        
        # 计算机器人局部位置（用于边界碰撞检测）
        robot_pos_local = np.array([
            robot_pos[0] - env_origin[0],
            robot_pos[1] - env_origin[1]
        ])
        
        # 从四元数提取 yaw 角
        w, x, y, z = robot_rot[0], robot_rot[1], robot_rot[2], robot_rot[3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 生成 LiDAR 射线方向（在机器人局部坐标系中，均匀分布在 360 度）
        angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)
        
        for ray_idx, angle in enumerate(angles):
            # 射线方向（在机器人局部坐标系中）
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # 转换到世界坐标系（考虑机器人朝向）
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ray_dir_world = np.array([
                ray_dir_local[0] * cos_yaw - ray_dir_local[1] * sin_yaw,
                ray_dir_local[0] * sin_yaw + ray_dir_local[1] * cos_yaw,
                0.0
            ])
            
            # 射线起点（机器人位置 + LiDAR 高度）
            ray_origin = Gf.Vec3f(
                float(robot_pos[0]),
                float(robot_pos[1]),
                float(robot_pos[2] + lidar_height)
            )
            
            # 射线方向（归一化的方向向量）
            ray_dir_norm = np.linalg.norm(ray_dir_world)
            if ray_dir_norm > 1e-6:
                ray_dir_normalized = ray_dir_world / ray_dir_norm
            else:
                ray_dir_normalized = np.array([1.0, 0.0, 0.0])
            
            ray_dir = Gf.Vec3f(
                float(ray_dir_normalized[0]),
                float(ray_dir_normalized[1]),
                float(ray_dir_normalized[2])
            )
            
            # 执行 ray cast
            hit = physx_interface.raycast_closest(
                ray_origin, 
                ray_dir, 
                float(lidar_max_range),
                bothSides=False
            )
            
            if hit:
                distance = hit.get('distance', lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
                # 检查是否与边界墙碰撞
                ray_end_local = robot_pos_local + ray_dir_local[:2] * lidar_max_range
                
                if scene_manager.check_boundary_collision(env_id, ray_end_local):
                    env_half = scene_manager.env_size * 0.5
                    dist_to_boundary = lidar_max_range
                    
                    # 检查四个边界
                    if ray_dir_local[0] > 0:
                        t = (env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[0] < 0:
                        t = (-env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    if ray_dir_local[1] > 0:
                        t = (env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[1] < 0:
                        t = (-env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    lidar_ranges[env_id, ray_idx] = min(dist_to_boundary, lidar_max_range)
                else:
                    lidar_ranges[env_id, ray_idx] = lidar_max_range
    
    return lidar_ranges


class WaffleDriveEnv:
    """
    自定义环境类，用于 skrl Trainer
    封装环境交互逻辑，让 Trainer 可以调用
    """
    def __init__(self, world, robots, scene_manager, reward_calculator, diff_ctrl,
                 env_origins, goal_pos, markers, left_idx, right_idx,
                 lidar_num_rays, lidar_max_range, max_v, max_w,
                 reset_dist, TIMEOUT_SECONDS, physics_dt, rng, device):
        self.world = world
        self.robots = robots
        self.scene_manager = scene_manager
        self.reward_calculator = reward_calculator
        self.diff_ctrl = diff_ctrl
        self.env_origins = env_origins
        self.goal_pos = goal_pos
        self.markers = markers
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.lidar_num_rays = lidar_num_rays
        self.lidar_max_range = lidar_max_range
        self.max_v = max_v
        self.max_w = max_w
        self.reset_dist = reset_dist
        self.TIMEOUT_SECONDS = TIMEOUT_SECONDS
        self.physics_dt = physics_dt
        self.rng = rng
        self.device = device
        
        self.num_envs = len(env_origins)
        
        # 定义观察和动作空间（用于 skrl Trainer）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(44,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32
        )
        
        # 训练状态
        self.active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self.cooldown_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.episode_start_time = np.zeros((self.num_envs,), dtype=np.float32)
        self.current_time = 0.0
        
        # 动作历史（用于观察）
        self.action_history = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.prev_measured_v = np.zeros((self.num_envs,), dtype=np.float32)
        self.prev_measured_w = np.zeros((self.num_envs,), dtype=np.float32)
        
        # 统计信息
        self.episode_rewards = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        
        # 初始重置
        self._reset_all()
    
    def _reset_all(self):
        """重置所有环境"""
        for i in range(self.num_envs):
            spawn_pos, spawn_yaw = self.scene_manager.get_robot_spawn_config(i, rng=self.rng)
            spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
            
            idx = np.array([i], dtype=np.int32)
            self.robots.set_world_poses(
                positions=spawn_pos.reshape(1, 3),
                orientations=spawn_rot.reshape(1, 4),
                indices=idx,
            )
            self.robots.set_velocities(
                velocities=np.zeros((1, 6), dtype=np.float32),
                indices=idx,
            )
            self.robots.set_joint_velocities(
                velocities=np.zeros((1, self.robots.num_dof), dtype=np.float32),
                indices=idx,
            )
            self.robots.set_joint_velocity_targets(
                np.zeros((1, self.robots.num_dof), dtype=np.float32),
                indices=idx,
            )
            
            self.goal_pos[i] = self.scene_manager.get_goal_config(i, rng=self.rng)
            self.markers[i].set_world_pose(position=self.goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
            self.episode_start_time[i] = 0.0
            self.action_history[i] = 0.0
            self.prev_measured_v[i] = 0.0
            self.prev_measured_w[i] = 0.0
    
    def reset(self, env_ids=None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        for i in env_ids:
            self.scene_manager.reset_scene_obstacles(np.array([i]), rng=self.rng)
            spawn_pos, spawn_yaw = self.scene_manager.get_robot_spawn_config(i, rng=self.rng)
            spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
            
            idx = np.array([i], dtype=np.int32)
            self.robots.set_world_poses(
                positions=spawn_pos.reshape(1, 3),
                orientations=spawn_rot.reshape(1, 4),
                indices=idx,
            )
            self.robots.set_velocities(
                velocities=np.zeros((1, 6), dtype=np.float32),
                indices=idx,
            )
            self.robots.set_joint_velocities(
                velocities=np.zeros((1, self.robots.num_dof), dtype=np.float32),
                indices=idx,
            )
            self.robots.set_joint_velocity_targets(
                np.zeros((1, self.robots.num_dof), dtype=np.float32),
                indices=idx,
            )
            
            self.goal_pos[i] = self.scene_manager.get_goal_config(i, rng=self.rng)
            self.markers[i].set_world_pose(position=self.goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
            self.episode_start_time[i] = self.current_time
            self.action_history[i] = 0.0
            self.prev_measured_v[i] = 0.0
            self.prev_measured_w[i] = 0.0
            
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
    
    def step(self, actions):
        """
        执行一步环境交互
        
        Args:
            actions: (num_envs, 2) 归一化的动作 [v, w]，范围 [-1, 1]
        
        Returns:
            obs: (num_envs, 44) 观察
            rewards: (num_envs,) 奖励
            dones: (num_envs,) 是否结束
            infos: dict 额外信息
        """
        # 转换为 numpy
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.array(actions)
        
        # 冷却期处理：强制 inactive 环境的动作为 0
        actions_clamped = actions_np.copy()
        actions_clamped[~self.active_mask.cpu().numpy()] = 0.0
        
        # 反归一化动作（从 [-1, 1] 到 [v, w]）
        v_cmd = actions_clamped[:, 0] * self.max_v
        w_cmd = actions_clamped[:, 1] * self.max_w
        
        # 转换为轮速并应用到机器人
        targets = np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32)
        for i in range(self.num_envs):
            action = self.diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, self.left_idx] = float(action.joint_velocities[0])
            targets[i, self.right_idx] = float(action.joint_velocities[1])
        
        self.robots.set_joint_velocity_targets(targets)
        self.world.step(render=True)
        
        # 更新仿真时间
        self.current_time += self.physics_dt
        
        # 获取机器人状态
        pos, rot = self.robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        x_local = (pos[:, 0] - self.env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - self.env_origins[:, 1]).astype(np.float32)
        goal_offsets = self.goal_pos[:, :2] - self.env_origins[:, :2]
        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)
        
        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        
        robot_velocities_np = self.robots.get_velocities()
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]
        
        # 计算奖励
        robot_pos_torch = torch.from_numpy(pos).float().to(self.device)
        robot_rot_torch = torch.from_numpy(rot).float().to(self.device)
        goal_pos_torch = torch.from_numpy(self.goal_pos).float().to(self.device)
        env_origins_torch = torch.from_numpy(self.env_origins).float().to(self.device)
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(self.device)
        
        # 计算 user_intent
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch,
            robot_rot_torch,
            goal_pos_torch,
            env_origins_torch,
            normalize=True
        )
        
        # 计算 LiDAR
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=self.env_origins,
            lidar_num_rays=self.lidar_num_rays,
            lidar_max_range=self.lidar_max_range,
            scene_manager=self.scene_manager
        )
        
        # 转换 actions 为 tensor（用于奖励计算）
        actions_torch = torch.from_numpy(actions_np).float().to(self.device)
        
        rewards_torch = self.reward_calculator.compute_rewards(
            lidar_ranges=torch.from_numpy(lidar_ranges).float().to(self.device),
            robot_positions=robot_pos_torch,
            robot_velocities=robot_velocities_torch,
            robot_orientations=robot_rot_torch,
            goal_positions=goal_pos_torch,
            env_origins=env_origins_torch,
            actions=actions_torch,
            action_history=torch.from_numpy(self.action_history).float().to(self.device),
            user_intent_env=user_intent_env
        )
        rewards = rewards_torch.detach().cpu().numpy()
        
        # 检查重置条件
        reset_timeout = (self.current_time - self.episode_start_time) >= self.TIMEOUT_SECONDS
        reached = dist <= self.reset_dist
        collision = np.zeros((self.num_envs,), dtype=bool)
        for i in range(self.num_envs):
            if self.scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]])):
                collision[i] = True
        
        to_reset = reset_timeout | reached | collision
        dones = to_reset.astype(np.float32)
        
        # 更新统计信息
        self.episode_rewards += torch.from_numpy(rewards).float().to(self.device)
        self.episode_lengths += 1
        
        # 处理重置和冷却期
        reset_env_ids = np.nonzero(to_reset)[0]
        
        if len(reset_env_ids) > 0:
            # 标记为 inactive，启动冷却期
            self.active_mask[reset_env_ids] = False
            self.cooldown_steps[reset_env_ids] = 10  # COOLDOWN_STEPS
            
            # 重置统计信息
            self.episode_rewards[reset_env_ids] = 0.0
            self.episode_lengths[reset_env_ids] = 0
            
            # 停止机器人
            self.robots.set_joint_velocity_targets(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((self.num_envs, 6), dtype=np.float32)
            self.robots.set_velocities(v_zero)
            self.robots.set_joint_velocities(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            
            # 重置环境
            self.reset(reset_env_ids)
            
            # 稳定步骤
            for _ in range(8):
                self.robots.set_joint_velocity_targets(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
                self.robots.set_velocities(v_zero)
                self.robots.set_joint_velocities(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
                self.world.step(render=True)
        
        # 更新冷却计数器
        self.cooldown_steps[~self.active_mask] -= 1
        cooldown_finished = (self.cooldown_steps <= 0) & (~self.active_mask)
        if cooldown_finished.any():
            self.active_mask[cooldown_finished] = True
        
        # 计算下一步观察
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=self.env_origins,
            lidar_num_rays=self.lidar_num_rays,
            lidar_max_range=self.lidar_max_range,
            scene_manager=self.scene_manager
        )
        
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=self.max_v,
            max_w=self.max_w
        )
        
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch,
            robot_rot_torch,
            goal_pos_torch,
            env_origins_torch,
            normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)
        
        next_obs = assemble_observations(
            robot_positions=pos,
            robot_orientations=rot,
            goal_positions=self.goal_pos,
            env_origins=self.env_origins,
            lidar_ranges=lidar_ranges,
            base_vel=base_vel_np,
            action_history=self.action_history,
            max_v=self.max_v,
            max_w=self.max_w,
            lidar_max_range=self.lidar_max_range,
            user_input=user_intent_np
        )
        
        # 更新动作历史
        self.action_history[:, 2:] = self.action_history[:, :2].copy()
        self.action_history[:, 0] = self.prev_measured_v.copy()
        self.action_history[:, 1] = self.prev_measured_w.copy()
        self.prev_measured_v = current_measured_v.copy()
        self.prev_measured_w = current_measured_w.copy()
        
        # 准备返回
        obs_torch = torch.from_numpy(next_obs).float().to(self.device)
        rewards_masked = rewards.copy()
        rewards_masked[~self.active_mask.cpu().numpy()] = 0.0
        dones_masked = dones.copy()
        dones_masked[~self.active_mask.cpu().numpy()] = 0.0
        
        infos = {
            "episode_rewards": self.episode_rewards.cpu().numpy(),
            "episode_lengths": self.episode_lengths.cpu().numpy(),
            "active_mask": self.active_mask.cpu().numpy(),
        }
        
        return obs_torch, torch.from_numpy(rewards_masked).float().to(self.device), \
               torch.from_numpy(dones_masked).float().to(self.device), infos
    
    def get_obs(self):
        """获取当前观察"""
        pos, rot = self.robots.get_world_poses()
        
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=self.env_origins,
            lidar_num_rays=self.lidar_num_rays,
            lidar_max_range=self.lidar_max_range,
            scene_manager=self.scene_manager
        )
        
        robot_velocities_np = self.robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=self.max_v,
            max_w=self.max_w
        )
        
        robot_pos_torch = torch.from_numpy(pos).float()
        robot_rot_torch = torch.from_numpy(rot).float()
        goal_pos_torch = torch.from_numpy(self.goal_pos).float()
        env_origins_torch = torch.from_numpy(self.env_origins).float()
        
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch,
            robot_rot_torch,
            goal_pos_torch,
            env_origins_torch,
            normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)
        
        obs = assemble_observations(
            robot_positions=pos,
            robot_orientations=rot,
            goal_positions=self.goal_pos,
            env_origins=self.env_origins,
            lidar_ranges=lidar_ranges,
            base_vel=base_vel_np,
            action_history=self.action_history,
            max_v=self.max_v,
            max_w=self.max_w,
            lidar_max_range=self.lidar_max_range,
            user_input=user_intent_np
        )
        
        return torch.from_numpy(obs).float().to(self.device)


def main():
    # ==================== 配置参数 ====================
    # 训练参数
    TOTAL_STEPS = 10000  # 总采样步数
    CHECKPOINT_INTERVAL = 1000  # Checkpoint 保存间隔
    CHECKPOINT_DIR = "checkpoints/stage1_trainer"
    TENSORBOARD_DIR = "runs/stage1_trainer"
    
    # 创建必要的目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    
    # ==================== 加载配置 ====================
    cfg = load_config()
    
    # ==================== 从配置读取环境参数 ====================
    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    env_gap = 2.0
    env_spacing = env_size + env_gap
    reset_dist = float(cfg.env.resetDist)
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    show_visual_walls = bool(cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 60.0
    
    # ==================== 从配置读取物理参数 ====================
    physics_dt = float(cfg.sim.dt)  # 0.025，对应40Hz
    render_dt = 1.0 / 60.0
    
    # ==================== 从配置读取机器人限制 ====================
    max_v = float(cfg.env.robot_limits.max_v)
    max_w = float(cfg.env.robot_limits.max_w)
    
    # ==================== 从配置读取 LiDAR 参数 ====================
    lidar_num_rays = int(cfg.env.lidar.num_rays)
    lidar_max_range = float(cfg.env.lidar.max_range)
    
    # ==================== 从配置读取奖励函数参数 ====================
    reward_cfg = cfg.env.rewards
    ra = float(reward_cfg.get("ra", 0.5))
    rl = float(reward_cfg.get("rl", -0.5))
    rh = float(reward_cfg.get("rh", -0.5))
    phi_thresh = float(reward_cfg.get("phi_thresh", 0.2))
    ras = float(reward_cfg.get("ras", -0.02))
    rc = float(reward_cfg.get("rc", -1.0))
    rcrit = float(reward_cfg.get("rcrit", -1.0))
    rcol = float(reward_cfg.get("rcol", -100.0))
    d_col = float(reward_cfg.get("d_col", 0.12))
    d_crit = float(reward_cfg.get("d_crit", 0.35))
    
    # ==================== 从配置读取训练参数 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_cfg_path = os.path.join(project_root, "cfg/train/WaffleDrivePPO.yaml")
    train_cfg = OmegaConf.load(train_cfg_path)
    rollouts = int(train_cfg.agent.rollouts)  # 8
    
    # ==================== 初始化随机数生成器 ====================
    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ==================== 1. 创建 World 和物理环境 ====================
    print("=" * 80)
    print("🌍 创建 World 和物理环境...")
    print("=" * 80)
    
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # ==================== 2. 计算环境原点 ====================
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
    
    # ==================== 3. 加载机器人模型 ====================
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)
    
    # 等待场景稳定
    for _ in range(180):
        world.step(render=True)
    
    # ==================== 4. 创建机器人视图 ====================
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()
    
    # ==================== 5. 配置机器人物理属性 ====================
    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)
    
    # ==================== 6. 配置轮子关节 ====================
    left_idx, right_idx = configure_wheel_joints(robots)
    if left_idx is None or right_idx is None:
        print("[ERROR] Could not find wheel joints in dof_names.")
        print(robots.dof_names)
        simulation_app.close()
        return
    
    # ==================== 7. 创建差速控制器 ====================
    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=max_v,
        max_angular_speed=max_w,
    )
    
    # ==================== 8. 创建场景管理器 ====================
    scene_types = [SCENE_EMPTY] * num_envs
    show_visual_walls_list = [show_visual_walls] * num_envs
    
    scene_manager = SceneManager(
        num_envs=num_envs,
        env_size=env_size,
        env_origins=env_origins,
        stage=stage
    )
    scene_manager.wall_thickness = wall_thickness
    scene_manager.wall_height = wall_height
    scene_manager.create_scene_obstacles(
        scene_types=scene_types,
        show_visual_walls=show_visual_walls_list
    )
    
    # ==================== 9. 确定设备 ====================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # ==================== 10. 创建奖励计算器 ====================
    reward_calculator = RewardCalculator(
        num_envs=num_envs,
        device=device,
        ra=ra, rl=rl, rh=rh, phi_thresh=phi_thresh,
        ras=ras, rc=rc, rcrit=rcrit, rcol=rcol,
        d_col=d_col, d_crit=d_crit,
        max_v=max_v, max_w=max_w
    )
    
    # ==================== 11. 初始化目标位置和标记 ====================
    goal_pos = np.zeros((num_envs, 3), dtype=np.float32)
    markers = []
    for i in range(num_envs):
        goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
        m = world.scene.add(
            VisualSphere(
                prim_path=f"/World/envs/env_{i}/GoalMarker",
                name=f"goal_marker_{i}",
                position=goal_pos[i].tolist(),
                radius=0.05,
                color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        markers.append(m)
    
    # ==================== 12. 创建 RL 模型 ====================
    print("\n" + "=" * 80)
    print("🧠 创建 RL 模型...")
    print("=" * 80)
    
    num_observations = 44
    num_actions = 2
    
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(num_observations,), dtype=np.float32
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0,
        shape=(num_actions,), dtype=np.float32
    )
    
    models = {
        "policy": WaffleCNNLSTMPolicyBase(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=True
        ),
        "value": WaffleCNNLSTMPolicyBase(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=True
        )
    }
    
    print("✅ 模型创建完成")
    
    # ==================== 13. 创建 Memory ====================
    print("\n" + "=" * 80)
    print("💾 创建 Memory...")
    print("=" * 80)
    
    memory_size = rollouts * num_envs
    
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=num_envs,
        device=device,
        replacement=True
    )
    
    print("✅ Memory 创建完成")
    
    # ==================== 14. 创建 PPO Agent ====================
    print("\n" + "=" * 80)
    print("🤖 创建 PPO Agent...")
    print("=" * 80)
    
    full_cfg = OmegaConf.create({
        "task": cfg,
        "train": train_cfg
    })
    
    agent = PPOAgentBase(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=full_cfg
    )
    
    print("✅ PPO Agent 创建完成")
    
    # ==================== 15. 创建环境 ====================
    print("\n" + "=" * 80)
    print("🌐 创建环境...")
    print("=" * 80)
    
    env = WaffleDriveEnv(
        world=world,
        robots=robots,
        scene_manager=scene_manager,
        reward_calculator=reward_calculator,
        diff_ctrl=diff_ctrl,
        env_origins=env_origins,
        goal_pos=goal_pos,
        markers=markers,
        left_idx=left_idx,
        right_idx=right_idx,
        lidar_num_rays=lidar_num_rays,
        lidar_max_range=lidar_max_range,
        max_v=max_v,
        max_w=max_w,
        reset_dist=reset_dist,
        TIMEOUT_SECONDS=TIMEOUT_SECONDS,
        physics_dt=physics_dt,
        rng=rng,
        device=device
    )
    
    print("✅ 环境创建完成")
    
    # ==================== 16. 创建 Tensorboard ====================
    print("\n" + "=" * 80)
    print("📊 初始化 Tensorboard...")
    print("=" * 80)
    
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    print("✅ Tensorboard 初始化完成")
    
    # ==================== 17. 创建 Trainer ====================
    print("\n" + "=" * 80)
    print("🚂 创建 Trainer...")
    print("=" * 80)
    
    # 定义训练回调函数
    def train_callback(trainer, env, agent, timestep, timesteps):
        """训练回调函数，用于记录 Tensorboard"""
        if timestep % rollouts == 0:
            infos = env.step.__self__ if hasattr(env.step, '__self__') else None
            if infos is None:
                # 尝试从环境获取统计信息
                if hasattr(env, 'episode_rewards'):
                    avg_reward = float(env.episode_rewards[env.active_mask].mean().item()) if env.active_mask.any() else 0.0
                    avg_length = float(env.episode_lengths[env.active_mask].float().mean().item()) if env.active_mask.any() else 0.0
                    writer.add_scalar("Episode/Reward", avg_reward, timestep)
                    writer.add_scalar("Episode/Length", avg_length, timestep)
    
    # skrl SequentialTrainer 不接受 timesteps 关键字参数，需放入 cfg.trainer.total_steps
    if "trainer" not in full_cfg:
        full_cfg.trainer = {}
    full_cfg.trainer["total_steps"] = TOTAL_STEPS
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        agents_scope=[0],
        cfg=full_cfg,
        callbacks={"train": train_callback}
    )
    
    print("✅ Trainer 创建完成")
    
    # ==================== 18. 开始训练 ====================
    print("\n" + "=" * 80)
    print("🎯 开始训练...")
    print("=" * 80)
    print(f"  - Total Steps: {TOTAL_STEPS}")
    print(f"  - Checkpoint Interval: {CHECKPOINT_INTERVAL}")
    print(f"  - Rollouts: {rollouts}")
    print("=" * 80 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ==================== 19. 训练结束 ====================
        print("\n" + "=" * 80)
        print("✅ 训练完成！")
        print("=" * 80)
        
        # 保存最终 Checkpoint
        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
        agent.save(final_checkpoint_path)
        print(f"💾 Final checkpoint saved: {final_checkpoint_path}")
        
        # 关闭 Tensorboard
        writer.close()
        print(f"📊 Tensorboard logs saved to: {TENSORBOARD_DIR}")
        
        # 关闭应用
        simulation_app.close()
        print("👋 应用已关闭")


if __name__ == "__main__":
    main()
