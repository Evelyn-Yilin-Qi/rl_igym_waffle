"""
RL Stage 1 PPO测试脚本 - 模块化版本 v2
使用模块化架构：core_network, algorithms, rewards, process_settings/env_setup
训练逻辑与 stage1_test_modular.py 完全一致，但使用 EnvironmentSetup 类简化初始化
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# 配置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 设备配置 ====================
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

from sim.scenes import (
    SCENE_EMPTY, SCENE_BOX,
    yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)
from envs.observations import (
    assemble_observations, 
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
    get_obs
)
from envs.user_intent import compute_user_intent_torch

# 导入模块化组件
from core_network import create_policy, create_value
from algorithms import create_algorithm
from rewards import RewardComposer
from process_settings.env_setup import EnvironmentSetup
from utils.config_utils import get_enabled_component

# ==================== 工具函数 ====================
def load_config(cfg_path):
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


def save_model(rl_agent, checkpoint_dir, tag):
    """保存模型"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")

# ==================== 主函数 ====================
def main():
    # 加载配置（模块化配置）
    env_cfg = load_config("cfg/WaffleDrive.yaml")  # 环境和物理配置
    network_cfg = load_config("config/network_config.yaml")  # 核心网络配置
    algo_cfg = load_config("config/algorithm_config.yaml")  # RL算法配置
    reward_cfg = load_config("config/reward_config.yaml")  # 奖励函数配置
    
    # 配置参数（从env_cfg读取）
    num_envs = 16  # 固定16个空场景，作为Stage1测试
    reset_dist = float(env_cfg.env.resetDist)
    show_visual_walls = bool(env_cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 60.0
    
    # 物理和机器人参数
    physics_dt = float(env_cfg.sim.dt)
    max_v = float(env_cfg.env.robot_limits.max_v)
    max_w = float(env_cfg.env.robot_limits.max_w)
    
    # LiDAR参数
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    
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
    
    # ==================== 使用 EnvironmentSetup 初始化环境 ====================
    scene_types = [SCENE_EMPTY] * num_envs  # 所有环境都是空场景
    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=show_visual_walls,  # bool值，所有环境统一设置
        reset_scene_obstacles=False  # Stage1不需要重置障碍物
    )
    
    # 一次性初始化所有环境
    setup_result = env_setup.setup_all()
    world = setup_result['world']
    robots = setup_result['robots']
    scene_manager = setup_result['scene_manager']
    diff_ctrl = setup_result['diff_ctrl']
    left_idx = setup_result['left_idx']
    right_idx = setup_result['right_idx']
    env_origins = setup_result['env_origins']
    goal_pos = setup_result['goal_pos']
    markers = setup_result['markers']
    
    # 初始化变量
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    action_history = np.zeros((num_envs, 4), dtype=np.float32)  # [v_t-1, w_t-1, v_t-2, w_t-2]
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
    
    # ==================== 初始化模块化组件 ====================
    # 1. 创建核心网络
    network_type, network_params = get_enabled_component(network_cfg.networks)
    network_params = network_params.copy() if network_params else {}
    
    # 自动设置所有架构通用的参数（这些不应该在配置文件中重复）
    network_params['obs_dim'] = 44  # 固定44维观察
    network_params['act_dim'] = 2   # 固定2维动作
    network_params['lidar_dim'] = lidar_num_rays  # 从环境配置读取
    network_params['state_dim'] = 8  # 固定：UserIntent(2) + BaseVel(2) + ActionHistory(4)
    
    policy = create_policy(network_type, **network_params)
    value = create_value(network_type, **network_params)
    
    # 2. 创建RL算法
    algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
    rl_agent = create_algorithm(
        algorithm_type,
        policy=policy,
        value=value,
        config=algorithm_params,
        device=device
    )
    
    # 3. 创建奖励组合器
    reward_composer = RewardComposer(reward_cfg.rewards)
    
    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)
    
    # 训练参数
    update_freq = 64  # 每64步更新一次策略
    step_count = -1
    reward_buffer = deque(maxlen=100)  # 奖励缓存，用于监控训练
    total_epochs = 50           # 总共训练50轮（每轮一次ppo.update）
    save_every = 10             # 每10轮保存一次checkpoint
    update_count = 0            # 已完成的更新轮数
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage1_test_modular_v2")
    writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "runs", "ppo_stage1_test_modular_v2"))
    reward_component_sums = {}
    reward_component_count = 0
    
    # 主循环
    current_time = 0.0
    last_obs_print = time.time()
    last_debug_print = time.time()

    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)
    
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
        obstacle_collision, critical_flag, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        
        # ==================== RL动作选择 ====================
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
                # RL选择动作（归一化的[-1,1]）
                if step_count % 10 == 0:
                    act, log_prob = rl_agent.select_action(obs[i])
                    temp_cur_act[i] = act
                    temp_logprob[i] = log_prob
                else:
                    # 持续控制
                    act = temp_cur_act[i]
                    obs_cur = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                    mean, log_std = rl_agent.policy(obs_cur)
                    std = log_std.exp()
                    dist = Normal(mean, std)
                    log_prob = dist.log_prob(
                        torch.tensor(act).unsqueeze(0).to(device)
                    ).sum().detach().cpu().numpy()
            
            log_probs[i] = log_prob
            current_act[i] = act
            
            # 反归一化到实际速度范围
            v_cmd[i] = act[0] * max_v
            w_cmd[i] = act[1] * max_w
            
            # 速度裁剪
            v_cmd[i] = np.clip(v_cmd[i], 0.0, max_v)
            w_cmd[i] = np.clip(w_cmd[i], -max_w, max_w)
        
        # ==================== 动作下发 ====================
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])

        # 更新动作历史：旧的 t-1 -> t-2，新的 v_cmd/w_cmd -> t-1
        action_history[:, 2:] = action_history[:, 0:2]  # 旧的 t-1 -> t-2
        action_history[:, 0] = v_cmd
        action_history[:, 1] = w_cmd
        
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
        next_obs = get_obs(robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, max_v, max_w, action_history, device=device)
        
        # 使用模块化奖励组合器计算奖励
        # 注意：传入测量速度用于离心力计算（如果启用）
        rewards, reward_info = reward_composer.compute(
            dist=dist,
            yaw_err=yaw_err,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            v_measured=current_measured_v,  # 实际测量的线速度（用于离心力计算）
            w_measured=current_measured_w,  # 实际测量的角速度（用于离心力计算）
            collision=obstacle_collision,
            reached=reached,
            timeout=reset_timeout,
            min_dist=min_dist,
            current_act=current_act,
            prev_act=prev_act,
            prev_prev_act=prev_prev_act
        )
        
        # 累计奖励分量用于TB统计
        reward_component_count += len(rewards)
        for k, v in reward_info.items():
            if k not in reward_component_sums:
                reward_component_sums[k] = 0.0
            reward_component_sums[k] += float(np.sum(v))
        
        for i in range(num_envs):
            reward_buffer.append(rewards[i])
            
            # 存储经验（动作归一化）
            rl_agent.store_transition(
                obs=obs[i],
                act=current_act[i],
                rew=rewards[i],
                next_obs=next_obs[i],
                done=to_reset[i],
                log_prob=log_probs[i]
            )
        
        # ==================== 策略更新 ====================
        if step_count % update_freq == 0 and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size:
            buffer_len = len(rl_agent.buffer["obs"])
            train_info = rl_agent.update()
            update_count += 1
            avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
            print(f"\n=== PPO策略更新 ===")
            print(f"更新步数: {step_count}, 已完成轮数: {update_count}/{total_epochs}, 最近100步平均奖励: {avg_reward:.2f}")
            print(f'Actor Loss: {train_info["actor_loss"]:.4f}, Critic Loss: {train_info["critic_loss"]:.4f}')
            print(f'Entropy: {train_info["entropy"]:.4f}, Approx KL: {train_info["approx_kl"]:.4f}')
            print(f'yaw:{yaw}')
            print(f'goal local:{goal_local}')
            print(f'robot local:{car_local}')
            print(f'角速度：{w_cmd}')
            print(f'线速度：{v_cmd}')
            print(f'目标距离：{dist}')
            print(f'用户意图：{user_intent_env}')
            print(f'角度差：{yaw_err}')

            # TensorBoard记录：训练loss信息
            writer.add_scalar("train/actor_loss", train_info["actor_loss"], update_count)
            writer.add_scalar("train/critic_loss", train_info["critic_loss"], update_count)
            writer.add_scalar("train/entropy", train_info["entropy"], update_count)
            writer.add_scalar("train/approx_kl", train_info["approx_kl"], update_count)
            
            # TensorBoard记录：奖励分量均值、总奖励、rollout信息
            if reward_component_count > 0:
                for k, v_sum in reward_component_sums.items():
                    writer.add_scalar(f"reward/{k}", v_sum / reward_component_count, update_count)
            writer.add_scalar("reward/total_mean", avg_reward, update_count)
            writer.add_scalar("rollout/steps_per_update", update_freq, update_count)
            writer.add_scalar("rollout/buffer_size", buffer_len, update_count)
            writer.add_scalar("rollout/step_count", step_count, update_count)

            # 重置累计器
            for k in reward_component_sums.keys():
                reward_component_sums[k] = 0.0
            reward_component_count = 0
            
            # 定期保存checkpoint
            if update_count % save_every == 0:
                save_model(rl_agent, checkpoint_dir, f"epoch_{update_count:03d}")

            # 达到总轮数，保存最终模型并退出
            if update_count >= total_epochs:
                # 添加时间戳后缀（精确到日期和小时）
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H")
                save_model(rl_agent, checkpoint_dir, f"final_{timestamp}")
                break
        
        # ==================== 机器人重置 ====================
        if np.any(to_reset):
            reset_ids = np.nonzero(to_reset)[0]
            
            # ==================== 处理根节点速度 ====================
            current_base_vels = robots.get_velocities()  # 形状: (num_envs, 6)
            current_base_vels[reset_ids] = 0.0
            robots.set_velocities(current_base_vels)

            # ==================== 处理关节实时速度 ====================
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
                action_history[i] = np.zeros(4, dtype=np.float32)  # 重置为4维零向量 [v_t-1, w_t-1, v_t-2, w_t-2]
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
    writer.close()

if __name__ == "__main__":
    main()
