"""
RL Stage 1 PPO测试脚本 - 升级奖励函数
集成障碍物奖惩、航向惩罚、动作平滑惩罚，适配轮椅碰撞模型
GPU加速版本
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from omegaconf import OmegaConf
import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print('wait for client')
from omni.isaac.kit import SimulationApp

# 配置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 设备配置 ====================
# 自动检测GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 设备配置 ===")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("未检测到GPU，使用CPU运行")

# 初始化Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.physx import get_physx_scene_query_interface
from pxr import Gf

# 导入自定义模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)
from envs.observations import assemble_observations, get_base_velocity_from_tensor
from envs.user_intent import compute_user_intent_torch

def pretrain_actor_to_simple_control(
    actor, 
    max_v, 
    max_w, 
    k_v=0.5,  # 你的线速度系数
    k_yaw=2.0,  # 你的角速度系数
    num_samples=100000  # 预训练样本数
):
    """
    预训练Actor网络，让其输出匹配你的简单速度控制逻辑：
    v_cmd = k_v * dist（裁剪到0~max_v）
    w_cmd = k_yaw * yaw_err（裁剪到-max_w~max_w）
    """
    # 1. 生成覆盖全场景的训练样本（重点包含后方目标）
    # 环境原点范围（匹配你的env_origins）
    env_origin_x = np.random.uniform(-10, 10, num_samples)
    env_origin_y = np.random.uniform(-10, 10, num_samples)
    
    # 小车局部位置（x_local/y_local）：覆盖-5~5（环境内任意位置）
    x_local = np.random.uniform(-5, 5, num_samples)
    y_local = np.random.uniform(-5, 5, num_samples)
    
    # 目标局部位置（gx_local/gy_local）：覆盖-5~5（包含后方目标）
    gx_local = np.random.uniform(-5, 5, num_samples)
    gy_local = np.random.uniform(-5, 5, num_samples)
    
    # 小车航向角（覆盖0~2π，包含任意朝向）
    yaw = np.random.uniform(-np.pi, np.pi, num_samples)
    
    # 2. 按你的逻辑计算目标动作（v_cmd/w_cmd）
    target_v = []
    target_w = []
    for i in range(num_samples):
        # 计算到目标的距离和角度误差（完全复刻你的逻辑）
        ex = gx_local[i] - x_local[i]
        ey = gy_local[i] - y_local[i]
        dist = np.sqrt(ex * ex + ey * ey)
        
        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw[i])
        
        # 你的速度控制公式
        v_cmd = np.clip(k_v * dist, 0.0, max_v)
        w_cmd = np.clip(k_yaw * yaw_err, -max_w, max_w)
        
        # 归一化到[-1,1]（匹配Actor网络输出范围）
        v_norm = v_cmd / max_v
        w_norm = w_cmd / max_w
        
        target_v.append(v_norm)
        target_w.append(w_norm)
    
    # 转换为tensor（移到训练设备）
    target_actions = torch.FloatTensor(np.column_stack([target_v, target_w])).to(device)
    
    # 3. 构造匹配你obs结构的输入样本
    obs_samples = []
    for i in range(num_samples):
        # 模拟你的obs结构：LiDAR(36) + UserIntent(2) + BaseVel(2) + ActionHistory(2)
        # 重点：把dist和yaw_err放在obs的关键位置（和你实际代码一致）
        lidar = np.ones(36) * 0.0  # 模拟LiDAR数据（全为最大距离）
        user_intent = np.array([0.0, 0.0])  # 模拟用户意图
        base_vel = np.array([0.0, 0.0])  # 模拟基础速度
        action_hist = np.array([0.0, 0.0])  # 模拟动作历史
        
        # 计算当前样本的dist和yaw_err（和目标动作一致）
        ex = gx_local[i] - x_local[i]
        ey = gy_local[i] - y_local[i]
        dist = np.sqrt(ex * ex + ey * ey)
        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw[i])
        
        # 组装obs（关键：把dist和yaw_err放在和你实际代码相同的位置）
        # 假设你的obs中：第38位=dist，第39位=yaw_err（根据assemble_observations调整）
        obs = np.hstack([
            lidar,          # 0-35: LiDAR
            user_intent,    # 36-37: 用户意图
            base_vel,       # 40-41: 基础速度
            action_hist     # 42-43: 动作历史
        ])
        obs_samples.append(obs)
    
    # 转换为tensor（移到训练设备）
    obs_samples = torch.FloatTensor(np.array(obs_samples)).to(device)
    
    # 4. 监督学习训练Actor网络（匹配你的速度控制逻辑）
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    actor.train()  # 切换到训练模式
    
    # 训练50轮（足够拟合线性逻辑）
    for epoch in range(50):
        # 前向传播：获取Actor输出的均值（忽略log_std）
        mean, _ = actor(obs_samples)
        
        # 计算损失（让网络输出匹配目标动作）
        loss = loss_fn(mean, target_actions)
        
        # 反向传播更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练进度
        if epoch % 10 == 0:
            print(f"预训练Epoch {epoch:2d} | 损失值: {loss.item():.6f}")

# ==================== PPO核心网络定义 ====================
class Actor(nn.Module):
    """PPO策略网络（输出动作分布的均值和标准差）"""
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        # 动作标准差限制（避免方差过大/过小）
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class Critic(nn.Module):
    """PPO价值网络（输出状态价值）"""
    def __init__(self, obs_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value_layer(x)
        return value

class PPO:
    """PPO算法核心类"""
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        lamda=0.95,
        clip_eps=0.2,
        k_epochs=3,
        batch_size=64
    ):
        # 网络初始化并移到指定设备
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 超参数
        self.gamma = gamma  # 折扣因子
        self.lamda = lamda  # GAE系数
        self.clip_eps = clip_eps  # 裁剪系数
        self.k_epochs = k_epochs  # 每次更新迭代次数
        self.batch_size = batch_size  # 批次大小
        self.entropy_coef = 0.0001
        
        # 经验缓存
        self.buffer = {
            "obs": [], "acts": [], "rews": [], 
            "next_obs": [], "dones": [], "log_probs": []
        }

    def select_action(self, obs):
        """根据观察选择动作（带探索）"""
        # 将观察数据移到指定设备
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        # 动作裁剪（限制在合理范围）
        action = torch.clamp(action, -1.0, 1.0)
        
        # 返回时转回CPU和numpy（因为后续计算用numpy）
        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().numpy()

    def store_transition(self, obs, act, rew, next_obs, done, log_prob):
        """存储经验"""
        self.buffer["obs"].append(obs)
        self.buffer["acts"].append(act)
        self.buffer["rews"].append(rew)
        self.buffer["next_obs"].append(next_obs)
        self.buffer["dones"].append(done)
        self.buffer["log_probs"].append(log_prob)

    def compute_gae(self, rewards, dones, values, next_values):
        """计算GAE（广义优势估计）"""
        advantages = []
        advantage = 0.0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = td_error + self.gamma * self.lamda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        returns = np.array(advantages) + np.array(values)
        # 优势归一化
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns

    def update(self):
        """更新PPO策略"""
        # 转换为tensor并移到指定设备
        obs = torch.FloatTensor(np.array(self.buffer["obs"])).to(device)
        acts = torch.FloatTensor(np.array(self.buffer["acts"])).to(device)
        rews = np.array(self.buffer["rews"])
        next_obs = torch.FloatTensor(np.array(self.buffer["next_obs"])).to(device)
        dones = np.array(self.buffer["dones"])
        old_log_probs = torch.FloatTensor(np.array(self.buffer["log_probs"])).to(device)
        
        # 计算价值和下一步价值
        values = self.critic(obs).detach().cpu().numpy().flatten()
        next_values = self.critic(next_obs).detach().cpu().numpy().flatten()
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(rews, dones, values, next_values)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # 多次迭代更新
        for _ in range(self.k_epochs):
            # 打乱数据
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            # 分批更新
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # 计算当前策略的log_prob
                mean, log_std = self.actor(obs[batch_idx])
                std = log_std.exp()
                dist = Normal(mean, std)
                current_log_probs = dist.log_prob(acts[batch_idx]).sum(dim=1)
                
                # 计算比率
                ratio = torch.exp(current_log_probs - old_log_probs[batch_idx])
                
                # 裁剪的优势损失
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch_idx]
                entropy = dist.entropy().sum(dim=1)  # 新增：计算每个动作的熵
                # 2. 策略损失 = 原损失 - 熵系数*熵（减号是因为要最大化熵，最小化损失）
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()  # 修改这行
                
                # 更新策略网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # 更新价值网络
                critic_loss = nn.MSELoss()(self.critic(obs[batch_idx]).flatten(), returns[batch_idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        # 清空缓存
        for key in self.buffer.keys():
            self.buffer[key].clear()

# ==================== 工具函数 ====================
def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """加载配置文件"""
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend([
            os.path.join(PROJECT_ROOT, cfg_path),
            os.path.abspath(cfg_path),
            os.path.join(os.getcwd(), cfg_path),
        ])

    found = None
    for path in candidates:
        if os.path.exists(path):
            found = os.path.abspath(path)
            break

    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")

    cfg = OmegaConf.load(found)
    return cfg

def compute_lidar_ranges(robot_positions, robot_orientations, env_origins, 
                         lidar_num_rays, lidar_max_range, scene_manager):
    """计算LiDAR距离（射线检测）"""
    num_envs = robot_positions.shape[0]
    lidar_ranges = np.ones((num_envs, lidar_num_rays), dtype=np.float32) * lidar_max_range
    
    physx_interface = get_physx_scene_query_interface()
    lidar_height = 0.1  # LiDAR安装高度
    
    for env_id in range(num_envs):
        robot_pos = robot_positions[env_id]
        robot_rot = robot_orientations[env_id]
        env_origin = env_origins[env_id]
        
        robot_pos_local = np.array([
            robot_pos[0] - env_origin[0],
            robot_pos[1] - env_origin[1]
        ])
        
        # 提取偏航角
        w, x, y, z = robot_rot
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 生成LiDAR射线角度
        angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)
        
        for ray_idx, angle in enumerate(angles):
            # 局部射线方向
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # 转换到世界坐标系
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ray_dir_world = np.array([
                ray_dir_local[0] * cos_yaw - ray_dir_local[1] * sin_yaw,
                ray_dir_local[0] * sin_yaw + ray_dir_local[1] * cos_yaw,
                0.0
            ])
            
            # 射线起点和方向
            ray_origin = Gf.Vec3f(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + lidar_height))
            ray_dir_norm = np.linalg.norm(ray_dir_world)
            ray_dir_normalized = ray_dir_world / ray_dir_norm if ray_dir_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            ray_dir = Gf.Vec3f(*ray_dir_normalized)
            
            # 射线检测
            hit = physx_interface.raycast_closest(ray_origin, ray_dir, float(lidar_max_range), False)
            if hit:
                distance = hit.get('distance', lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
                # 检查边界碰撞
                ray_end_local = robot_pos_local + ray_dir_local[:2] * lidar_max_range
                if scene_manager.check_boundary_collision(env_id, ray_end_local):
                    env_half = scene_manager.env_size * 0.5
                    dist_to_boundary = lidar_max_range
                    
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
    
    return lidar_ranges

def check_obstacle_collision(lidar_ranges, dcol=0.2, dcrit=0.5):
    """
    检查障碍物碰撞（胶囊形模型）
    Args:
        lidar_ranges: (num_envs, lidar_num_rays) LiDAR测距数据
        dcol: 碰撞阈值（m）
        dcrit: 临界阈值（m），dcrit > dcol
    Returns:
        collision_flag: (num_envs,) 是否碰撞（d < dcrit）
        min_dist: (num_envs,) 最近障碍物距离
    """
    num_envs = lidar_ranges.shape[0]
    collision_flag = np.zeros(num_envs, dtype=bool)
    min_dist = np.min(lidar_ranges, axis=1)  # 每个环境的最近障碍物距离
    
    # 胶囊形模型修正：沿前进方向扩展碰撞检测范围
    # 轮椅长度方向（前进）的安全距离修正
    forward_rays = lidar_ranges[:, :int(lidar_ranges.shape[1]/4)]  # 前90° LiDAR射线
    forward_min_dist = np.min(forward_rays, axis=1)
    min_dist = np.minimum(min_dist, forward_min_dist * 1.2)  # 胶囊形长度修正
    
    # 判定碰撞
    collision_flag = min_dist < dcrit
    
    return collision_flag, min_dist

def compute_obstacle_penalty(min_dist, dcol=0.2, dcrit=0.5, rc=-0.5, rcrit=-2.0):
    """
    计算障碍物惩罚
    Args:
        min_dist: (num_envs,) 最近障碍物距离
        dcol: 碰撞阈值
        dcrit: 临界阈值
        rc: 基础惩罚系数
        rcrit: 二次惩罚系数
    Returns:
        penalty: (num_envs,) 障碍物惩罚值
    """
    num_envs = len(min_dist)
    penalty = np.zeros(num_envs, dtype=np.float32)
    
    # 仅对 dcol < d < dcrit 的情况施加惩罚
    mask = (min_dist > dcol) & (min_dist < dcrit)
    penalty[mask] = rc + rcrit * (dcrit - min_dist[mask]) ** 2
    
    return penalty

def compute_heading_penalty(yaw_err, phi_thresh=np.pi/12, rh=-0.1):
    """
    计算航向惩罚（降低认知负荷）
    Args:
        yaw_err: (num_envs,) 航向角误差（Φ）
        phi_thresh: 航向角阈值（rad），默认15°
        rh: 惩罚系数
    Returns:
        penalty: (num_envs,) 航向惩罚值
    """
    num_envs = len(yaw_err)
    penalty = np.zeros(num_envs, dtype=np.float32)
    
    # 仅当|Φ| > Φthresh时施加惩罚
    abs_err = np.abs(yaw_err)
    mask = abs_err > phi_thresh
    penalty[mask] = rh * (abs_err[mask] ** 2)
    
    return penalty

def compute_smoothness_penalty(current_act, prev_act, prev_prev_act, ras1=-0.05, ras2=-0.02):
    """
    计算动作平滑惩罚（一阶+二阶）
    Args:
        current_act: (num_envs, 2) 当前动作 [v, w]（归一化）
        prev_act: (num_envs, 2) 上一动作
        prev_prev_act: (num_envs, 2) 前两动作
        ras1: 一阶平滑惩罚系数
        ras2: 二阶平滑惩罚系数
    Returns:
        penalty: (num_envs,) 总平滑惩罚
    """
    num_envs = current_act.shape[0]
    
    # 一阶平滑惩罚：|a_t - a_t-1|²
    first_order = np.sum((current_act - prev_act) ** 2, axis=1)
    
    # 二阶平滑惩罚：|a_t - 2*a_t-1 + a_t-2|²（二阶差分）
    second_order = np.sum((current_act - 2*prev_act + prev_prev_act) ** 2, axis=1)
    
    # 总惩罚
    penalty = ras1 * first_order + ras2 * second_order
    
    return penalty

def compute_total_reward(dist, yaw_err, v_cmd, collision, reached, timeout,
                         min_dist, current_act, prev_act, prev_prev_act, w_cmd):
    """
    计算多环境的总奖励（适配数组输入）
    Args:
        dist: (n,) 每个环境到目标的距离
        yaw_err: (n,) 每个环境的航向角误差（Φ）
        v_cmd: (n,) 每个环境的线速度指令
        collision: (n,) 每个环境是否障碍物碰撞（bool）
        reached: (n,) 每个环境是否到达目标（bool）
        timeout: (n,) 每个环境是否超时（bool）
        min_dist: (n,) 每个环境的最近障碍物距离
        current_act: (n, 2) 每个环境的当前归一化动作 [v, w]
        prev_act: (n, 2) 每个环境的上一归一化动作
        prev_prev_act: (n, 2) 每个环境的前两归一化动作
    Returns:
        total_reward: (n,) 每个环境的总奖励
    """
    # 确保输入为numpy数组（兼容列表输入）
    dist = np.asarray(dist, dtype=np.float32)
    yaw_err = np.asarray(yaw_err, dtype=np.float32)
    v_cmd = np.asarray(v_cmd, dtype=np.float32)
    collision = np.asarray(collision, dtype=bool)
    reached = np.asarray(reached, dtype=bool)
    timeout = np.asarray(timeout, dtype=bool)
    min_dist = np.asarray(min_dist, dtype=np.float32)
    current_act = np.asarray(current_act, dtype=np.float32)
    prev_act = np.asarray(prev_act, dtype=np.float32)
    prev_prev_act = np.asarray(prev_prev_act, dtype=np.float32)
    
    n = len(dist)  # 环境数量
    total_reward = np.zeros(n, dtype=np.float32)

    # 距离奖励
    dist_reward = -dist * 0.5  # 距离越小，奖励越高
    # 目标奖励/超时惩罚（数组索引赋值）
    goal_reward = np.zeros(n, dtype=np.float32)
    goal_reward[reached] = 10.0
    timeout_penalty = np.zeros(n, dtype=np.float32)
    timeout_penalty[timeout] = -8.0 - 1.0 * dist[timeout]

    static_pen = -1 * ((abs(v_cmd) < 0.1) & (abs(w_cmd) < 0.2) )
    # rw_wv = 0.05* abs(w_cmd) + 0.05 * abs(v_cmd)
    
    # 2. 障碍物惩罚（碰撞时奖励为0，直接跳过后续计算）
    non_collision_mask = ~collision  # 非碰撞环境掩码
    obstacle_pen = np.zeros(n, dtype=np.float32)
    if np.any(non_collision_mask):
        # 仅对非碰撞环境计算障碍物惩罚
        dcol = 0.2
        dcrit = 0.5
        rc = -0.5
        rcrit = -2.0
        
        # 筛选非碰撞环境的最近距离
        min_dist_non_collision = min_dist[non_collision_mask]
        # 计算dcol < d < dcrit的环境掩码
        danger_mask = (min_dist_non_collision > dcol) & (min_dist_non_collision < dcrit)
        # 计算障碍物惩罚
        obstacle_pen_non_collision = np.zeros_like(min_dist_non_collision)
        obstacle_pen_non_collision[danger_mask] = rc + rcrit * (dcrit - min_dist_non_collision[danger_mask]) ** 2
        # 赋值回总惩罚数组
        obstacle_pen[non_collision_mask] = obstacle_pen_non_collision
    
    # 3. 航向惩罚（仅对非碰撞环境计算）
    heading_pen = np.zeros(n, dtype=np.float32)
    if np.any(non_collision_mask):
        phi_thresh = np.pi/12  # 15°阈值
        rh = -0.2
        
        # 筛选非碰撞环境的航向误差
        yaw_err_non_collision = yaw_err[non_collision_mask]
        abs_err = np.abs(yaw_err_non_collision)
        # 仅当|Φ| > Φthresh时施加惩罚
        heading_mask = abs_err > phi_thresh
        heading_pen_non_collision = np.zeros_like(yaw_err_non_collision)
        heading_pen_non_collision[heading_mask] = rh * (abs_err[heading_mask] ** 2)
        # 赋值回总惩罚数组
        heading_pen[non_collision_mask] = heading_pen_non_collision
    
    # 4. 动作平滑惩罚（仅对非碰撞环境计算）
    smooth_pen = np.zeros(n, dtype=np.float32)
    if np.any(non_collision_mask):
        ras1 = -0.02  # 一阶平滑惩罚系数
        ras2 = -0.02  # 二阶平滑惩罚系数
        
        # 筛选非碰撞环境的动作
        current_act_non_collision = current_act[non_collision_mask]
        prev_act_non_collision = prev_act[non_collision_mask]
        prev_prev_act_non_collision = prev_prev_act[non_collision_mask]
        
        # 一阶平滑惩罚：|a_t - a_t-1|²（按动作维度求和）
        first_order = np.sum((current_act_non_collision - prev_act_non_collision) ** 2, axis=1)
        # 二阶平滑惩罚：|a_t - 2*a_t-1 + a_t-2|²
        second_order = np.sum((current_act_non_collision - 2*prev_act_non_collision + prev_prev_act_non_collision) ** 2, axis=1)
        # 总平滑惩罚
        smooth_pen_non_collision = ras1 * first_order + ras2 * second_order
        # 赋值回总惩罚数组
        smooth_pen[non_collision_mask] = smooth_pen_non_collision
    
    # 5. 总奖励计算（仅非碰撞环境累加奖励）
    total_reward[non_collision_mask] = (
        # rw_wv[non_collision_mask] +
        static_pen[non_collision_mask] +
        dist_reward[non_collision_mask] +
        goal_reward[non_collision_mask] +
        timeout_penalty[non_collision_mask] +
        obstacle_pen[non_collision_mask] +
        heading_pen[non_collision_mask] +
        smooth_pen[non_collision_mask]
    )

    return total_reward


def get_obs(robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, max_v, max_w, action_history):
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        
        # 局部坐标计算
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        
        # 目标偏移计算
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        
        # 到目标的距离和方位角误差
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)


        
        # ==================== 观察数据采集 ====================
        # 1. LiDAR数据
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        # 2. 基础速度
        robot_velocities_np = robots.get_velocities()
        # 移到GPU计算（如果可用）
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=max_v,
            max_w=max_w
        )  # 转回CPU用于后续numpy计算
        
        # 3. 当前测量速度
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]
        
        # 4. 用户意图
        # 张量移到GPU
        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        robot_rot_torch = torch.from_numpy(rot).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch, torch.tensor(yaw), goal_pos_torch, env_origins_torch, normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)  # 转回CPU
        
        # 5. 组装观察向量
        obs = assemble_observations(
            robot_positions=pos,
            robot_orientations=rot,
            goal_positions=goal_pos,
            env_origins=env_origins,
            lidar_ranges=lidar_ranges,
            base_vel=base_vel_np,
            action_history=action_history,
            max_v=max_v,
            max_w=max_w,
            lidar_max_range=lidar_max_range,
            user_input=user_intent_np
        )
        return obs

# ==================== 主函数 ====================
def main():
    # 加载配置
    cfg = load_config()
    
    # 配置参数
    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    env_gap = 2.0
    env_spacing = env_size + env_gap
    reset_dist = float(cfg.env.resetDist)
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    show_visual_walls = bool(cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 60.0
    
    # 物理和机器人参数
    physics_dt = float(cfg.sim.dt)
    render_dt = 1.0 / 30.0
    max_v = float(cfg.env.robot_limits.max_v)
    max_w = float(cfg.env.robot_limits.max_w)
    
    # LiDAR参数
    lidar_num_rays = int(cfg.env.lidar.num_rays)
    lidar_max_range = float(cfg.env.lidar.max_range)
    
    # 碰撞模型参数（轮椅胶囊形）
    DCOL = 0.2  # 碰撞阈值（20cm）
    DCRIT = 0.5  # 临界阈值（50cm）
    
    # 随机数初始化
    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 初始化仿真世界
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # 环境原点
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
    
    # 加载机器人
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)
    
    # 等待场景稳定
    for _ in range(180):
        world.step(render=True)
    
    # 创建机器人视图
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()
    
    # 配置机器人物理属性
    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)
    
    # 配置轮子关节
    left_idx, right_idx = configure_wheel_joints(robots)
    if left_idx is None or right_idx is None:
        print("[ERROR] Could not find wheel joints")
        simulation_app.close()
        return
    
    # 差速控制器
    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=max_v,
        max_angular_speed=max_w,
    )
    
    # 场景管理器
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
    scene_manager.create_scene_obstacles(scene_types=scene_types, show_visual_walls=show_visual_walls_list)
    
    # 目标位置和标记
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
    
    # 初始化变量
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    action_history = np.zeros((num_envs, 2), dtype=np.float32)
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
    
    # 动作历史缓存（用于平滑惩罚）
    prev_act = np.zeros((num_envs, 2), dtype=np.float32)  # a_t-1
    prev_prev_act = np.zeros((num_envs, 2), dtype=np.float32)  # a_t-2
    
    # 重置所有机器人
    for i in range(num_envs):
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        idx = np.array([i], dtype=np.int32)
        robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=idx,
        )
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        episode_start_time[i] = 0.0
    
    # ==================== 初始化PPO ====================
    obs_dim = 36 + 2 + 2 + 2  # LiDAR(36) + UserIntent(2) + BaseVel(2) + ActionHistory(2)
    act_dim = 2  # 线速度v、角速度w（归一化到[-1,1]）
    ppo = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=3e-4,
        gamma=0.95,
        lamda=0.90,
        clip_eps=0.2,
        k_epochs=3,
        batch_size=64
    )

    
    # 训练参数
    update_freq = 64  # 每64步更新一次策略
    step_count = -1
    reward_buffer = deque(maxlen=100)  # 奖励缓存，用于监控训练
    
    # 主循环
    current_time = 0.0
    last_obs_print = time.time()
    last_debug_print = time.time()
    
    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1
        
        # 获取机器人状态
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        
        # 局部坐标计算
        car_local = pos[:, :2] - env_origins[:, :2]  # (num_envs, 2)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        
        # 目标偏移计算
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        
        # 到目标的距离和方位角误差
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)


        
        # ==================== 观察数据采集 ====================
        # 1. LiDAR数据
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        # 2. 基础速度
        robot_velocities_np = robots.get_velocities()
        # 移到GPU计算（如果可用）
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=max_v,
            max_w=max_w
        )  # 转回CPU用于后续numpy计算
        
        # 3. 当前测量速度
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]
        
        # 4. 用户意图
        # 张量移到GPU
        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        robot_rot_torch = torch.from_numpy(rot).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch, torch.tensor(yaw), goal_pos_torch, env_origins_torch, normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)  # 转回CPU
        
        # 5. 组装观察向量
        obs = assemble_observations(
            robot_positions=pos,
            robot_orientations=rot,
            goal_positions=goal_pos,
            env_origins=env_origins,
            lidar_ranges=lidar_ranges,
            base_vel=base_vel_np,
            action_history=action_history,
            max_v=max_v,
            max_w=max_w,
            lidar_max_range=lidar_max_range,
            user_input=user_intent_np
        )
        
        # ==================== 障碍物碰撞检测 ====================
        obstacle_collision, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        
        # ==================== PPO动作选择 ====================
        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)  # 归一化动作
        
        for i in range(num_envs):
            # 碰撞时强制停止
            if obstacle_collision[i]:
                act = np.array([0.0, 0.0])
                log_prob = 0.0
            else:
                # PPO选择动作（归一化的[-1,1]）
                if step_count % 1 == 0:
                    act, log_prob = ppo.select_action(obs[i])
            
            log_probs[i] = log_prob
            current_act[i] = act
            
            # 反归一化到实际速度范围
            v_cmd[i] = act[0] * max_v
            w_cmd[i] = act[1] * max_w
            
            # 速度裁剪
            v_cmd[i] = np.clip(v_cmd[i], 0.0, max_v)
            w_cmd[i] = np.clip(w_cmd[i], -max_w, max_w)

            # w_cmd = np.ones(num_envs, dtype=np.float32)*1 左转
            # w_cmd = np.ones(num_envs, dtype=np.float32)*-1 右转
        
        # ==================== 动作下发 ====================
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])

        action_history = np.concatenate([[v_cmd], [w_cmd]], axis=0).T
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
        # ==================== 重置条件检查 ====================
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        reached = dist <= reset_dist
        
        boundary_collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            boundary_collision[i] = scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]]))
        
        # 总重置条件：障碍物碰撞 + 边界碰撞 + 超时 + 到达目标
        to_reset = obstacle_collision | boundary_collision | reset_timeout | reached
        
        # ==================== 奖励计算与经验存储 ====================
        next_obs = get_obs(robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, max_v, max_w, action_history)
        rewards = compute_total_reward(
            dist=dist,
            yaw_err=yaw_err,
            v_cmd=v_cmd,
            collision=obstacle_collision,
            reached=reached,
            timeout=reset_timeout,
            min_dist=min_dist,
            current_act=current_act,
            prev_act=prev_act,
            prev_prev_act=prev_prev_act,
            w_cmd=w_cmd
        )
        
        for i in range(num_envs):
            reward_buffer.append(rewards[i])
            
            # 存储经验（动作归一化）
            ppo.store_transition(
                obs=obs[i],
                act=current_act[i],
                rew=rewards[i],
                next_obs=next_obs[i],
                done=to_reset[i],
                log_prob=log_probs[i]
            )
        
        # ==================== 策略更新 ====================
        if step_count % update_freq == 0 and len(ppo.buffer["obs"]) >= ppo.batch_size:
            ppo.update()
            avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
            print(f"\n=== PPO策略更新 ===")
            print(f"更新步数: {step_count}, 最近100步平均奖励: {avg_reward:.2f}")
            print(f'yaw:{yaw}')
            print(f'goal local:{goal_local}')
            print(f'robot local:{car_local}')
            print(f'角速度：{w_cmd}')
            print(f'线速度：{v_cmd}')
            print(f'目标距离：{dist}')
            print(f'用户意图：{user_intent_env}')
            print(f'角度差：{yaw_err}')
            # 打印GPU使用情况（如果使用GPU）
            # if torch.cuda.is_available():
            #     print(f"GPU内存使用: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        
        # ==================== 机器人重置 ====================
        if np.any(to_reset):
            reset_ids = np.nonzero(to_reset)[0]
            
            # ==================== 处理关节速度目标值 ====================
            # 获取当前所有机器人的关节速度目标值（完整Tensor）
            # current_joint_vel_targets = robots.get_joint_velocity_targets()  # 形状: (num_envs, num_dof)
            # # 仅将需要重置的机器人的关节速度目标值设为0
            # current_joint_vel_targets[reset_ids] = 0.0
            # # 一次性设置所有机器人的关节速度目标值（未重置的保持原值）
            # robots.set_joint_velocity_targets(current_joint_vel_targets)
            
            # ==================== 处理根节点速度 ====================
            # 获取当前所有机器人的根节点速度（完整Tensor）
            current_base_vels = robots.get_velocities()  # 形状: (num_envs, 6)
            # 仅将需要重置的机器人的根节点速度设为0
            current_base_vels[reset_ids] = 0.0
            # 一次性设置所有机器人的根节点速度（未重置的保持原值）
            robots.set_velocities(current_base_vels)

            # ==================== 处理关节实时速度 ====================
            # 如果需要强制停止关节（硬停止），补充以下代码
            current_joint_vels = robots.get_joint_velocities()  # 形状: (num_envs, num_dof)
            current_joint_vels[reset_ids] = 0.0
            robots.set_joint_velocities(current_joint_vels)
            
            # 重置每个环境
            for i in reset_ids:
                # 重置场景和机器人
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
                spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
                idx = np.array([i], dtype=np.int32)
                
                robots.set_world_poses(
                    positions=spawn_pos.reshape(1, 3),
                    orientations=spawn_rot.reshape(1, 4),
                    indices=idx,
                )
                robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
                robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                
                # 重置目标
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                
                # 重置计时和状态
                episode_start_time[i] = current_time
                action_history[i] = 0.0
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
                
                # 重置动作历史（平滑惩罚用）
                prev_act[i] = np.zeros(2)
                prev_prev_act[i] = np.zeros(2)
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # 更新动作历史
        # action_history[:, :2] = action_history[:, :2].copy()
        # action_history[:, 0] = prev_measured_v.copy()
        # action_history[:, 1] = prev_measured_w.copy()
        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        
        # 更新平滑惩罚用的动作历史
        prev_prev_act = prev_act.copy()
        prev_act = current_act.copy()
        
        # ==================== 打印输出 ====================
        now = time.time()
        # 每隔1秒打印观察
        if now - last_obs_print >= 1.0:
            obs_env0 = obs[0]
            non_lidar_obs = obs_env0[36:44]
            lidar_first2 = obs_env0[0:2]
            
            last_obs_print = now
        
        # 每0.5秒打印调试信息
        if now - last_debug_print > 0.5:
            last_debug_print = now
        
        time.sleep(0.001)
    
    simulation_app.close()

if __name__ == "__main__":
    main()