      
"""
SFT + PPO training script (modular validation version).

Goal:
- Keep behavior aligned with scripts/sft_rl_tb3_V3.py
- Move SFT-related logic into top-level sft/ package
"""
import os
import sys
import time
import random
from collections import deque
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print('wait for client')
import numpy as np
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import carb  # 用于键盘输入监听
from datetime import datetime
# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== Device ===")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# Isaac Sim app
simulation_app = SimulationApp({"headless": False})
import omni.appwindow

from sim.scenes import (
    SCENE_EMPTY,
    SCENE_CYLINDER,
    SCENE_DOOR,
    SCENE_BOX,
    yaw_from_quat_wxyz,
    quat_wxyz_from_yaw,
    wrap_to_pi,
)

scenes = [SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR]
base_dir = "./data"
for scene in scenes:
    scene_path = os.path.join(base_dir, scene)
    os.makedirs(scene_path, exist_ok=True)

from sim.robot import MAX_V, MAX_W
from envs.observations import (
    assemble_observations,
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
    get_obs,
)
from envs.user_intent import compute_user_intent_torch

from process_settings.env_setup import EnvironmentSetup
from core_network import create_policy, create_value
from algorithms import create_algorithm
from rewards import RewardComposer
from utils.config_utils import get_enabled_component
from sft import (
    load_config,
    get_network_params_by_name,
    RuleBasedController,
    train_supervised_policy,
    print_phase_status,
    print_reward_breakdown,
)

# ========== 【修改 1】全局变量改为按键状态计数（支持多键） ==========
key_state = {
    carb.input.KeyboardInput.W: 0,
    carb.input.KeyboardInput.S: 0,
    carb.input.KeyboardInput.A: 0,
    carb.input.KeyboardInput.D: 0,
}

key_name_map = {
    carb.input.KeyboardInput.W: "W",
    carb.input.KeyboardInput.S: "S",
    carb.input.KeyboardInput.A: "A",
    carb.input.KeyboardInput.D: "D"
}


def on_keyboard_event(event, *args, **kwargs):
    """【修改 2】支持多键同时按下的键盘事件回调"""
    if event.input not in key_state:
        return

    # 按下：计数+1
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        key_state[event.input] = 1
        print(f"【按住】{key_name_map[event.input]}")

    # 松开：计数-1
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        key_state[event.input] = 0
        print(f"【松开】{key_name_map[event.input]}")



def save_model(rl_agent, checkpoint_dir, tag):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")

def save_sample(data_list, test_scene_type):
    obs_list = np.array([item["obs"] for item in data_list])  # shape: (N, 40)
    v_list = np.array([item["v"] for item in data_list])
    w_list = np.array([item["w"] for item in data_list])
    test_scene_type_list = np.array([item["test_scene_type"] for item in data_list])
    time_str = datetime.now().strftime("%H_%M_%S_%f")
    # 保存到文件
    np.savez(
        f"data/{test_scene_type}/{time_str}.npz",
        obs=obs_list,
        v=v_list,
        w=w_list,
        test_scene_type=test_scene_type_list
    )

def main():
    # ❗❗❗单环境快速测试场景切换区（只改这里）
    # ❗num_envs 必须固定为 1，不要在这个脚本里改成多环境
    # ❗❗❗❗❗❗可选场景：SCENE_EMPTY / SCENE_CYLINDER / SCENE_DOOR / SCENE_BOX
    test_scene_type = SCENE_BOX
    num_envs = 1
    scene_types = [test_scene_type]

    # Config split (old script used single `cfg`)
    env_cfg = OmegaConf.create(
        {
            "env": {
                "numEnvs": num_envs,
                "resetDist": 0.2,
                "lidar": {"num_rays": 360, "max_range": 3.0},
                "scene": {
                    "env_size": 6.0,
                    "wall_thickness": 0.08,
                    "wall_height": 1.2,
                    "show_visual_walls": False,
                },
            },
            "sim": {"dt": 0.025},
        }
    )
    network_cfg = load_config(PROJECT_ROOT, "config/network_config.yaml")
    algo_cfg = load_config(PROJECT_ROOT, "config/algorithm_config.yaml")
    reward_cfg = load_config(PROJECT_ROOT, "config/reward_config.yaml")

    # Environment/training params
    num_envs = int(env_cfg.env.numEnvs)
    reset_dist = float(env_cfg.env.resetDist)
    # Align with stage training behavior:
    # EMPTY scene -> no visual walls, non-EMPTY scenes -> visual walls enabled.
    # SceneManager handles this automatically when passing None.
    show_visual_walls = None
    TIMEOUT_SECONDS = 1200.0
    physics_dt = float(env_cfg.sim.dt)
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    DCOL = 0.2
    DCRIT = 0.5

    data_list = []

    # SFT settings (aligned with stage1_rl_sft.py / stage2_rl_sft.py)
    enable_sft_warmup = True          # old: ENABLE_SFT
    sft_collection_active = enable_sft_warmup  # old runtime flag: in_supervised_phase
    sft_collect_total_steps = 200000  # old: SUPERVISED_STEPS
    sft_opt_epochs = 50               # old: SUPERVISED_EPOCHS

    # Reproducibility
    seed = random.randint(0, 10000)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Scene types are configured by `scene_types` above.
    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=show_visual_walls,
        reset_scene_obstacles=True,
    )
    setup = env_setup.setup_all()
    world = setup["world"]
    robots = setup["robots"]
    scene_manager = setup["scene_manager"]
    diff_ctrl = setup["diff_ctrl"]
    left_idx = setup["left_idx"]
    right_idx = setup["right_idx"]
    env_origins = setup["env_origins"]
    goal_pos = setup["goal_pos"]
    markers = setup["markers"]

    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    action_history = np.zeros((num_envs, 4), dtype=np.float32)  # v_t-1,w_t-1,v_t-2,w_t-2
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
    prev_act = np.zeros((num_envs, 2), dtype=np.float32)
    prev_prev_act = np.zeros((num_envs, 2), dtype=np.float32)

    for i in range(num_envs):
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        idx = np.array([i], dtype=np.int32)
        robots.set_world_poses(positions=spawn_pos.reshape(1, 3), orientations=spawn_rot.reshape(1, 4), indices=idx)
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        episode_start_time[i] = 0.0

    # Network: force SFT architecture while keeping config-managed params
    configured_network_type, _ = get_enabled_component(network_cfg.networks)
    network_params = get_network_params_by_name(network_cfg, "simple_fc_sft")
    if configured_network_type != "simple_fc_sft":
        print(f"[INFO] network '{configured_network_type}' is enabled in config. Forcing 'simple_fc_sft' for this script.")
    network_type = "simple_fc_sft"
    network_params.update(
        {
            "obs_dim": 44,
            "act_dim": 2,
            "lidar_dim": lidar_num_rays,
            "state_dim": 8,  # UserIntent(2)+BaseVel(2)+ActionHistory(4)
        }
    )
    policy = create_policy(network_type, **network_params)
    value = create_value(network_type, **network_params)

    algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
    rl_agent = create_algorithm(
        algorithm_type,
        policy=policy,
        value=value,
        config=algorithm_params,
        device=device,
    )
    reward_composer = RewardComposer(reward_cfg.rewards)


    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)

    # RL settings (aligned with stage1_rl_sft.py / stage2_rl_sft.py)
    ppo_update_interval_steps = 2048      # old: update_freq
    ckpt_save_every_update_epochs = 60    # old: save_every
    ppo_total_update_epochs = 300         # old: total_epochs
    ppo_update_epoch_count = 0            # old: update_count
    step_count = -1
    current_time = 0.0
    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    supervised_obs = []
    supervised_actions = []
    total_obstacle_collision = 0

    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_sft_rl_tb3_v3")
    writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "runs", "ppo_sft_rl_tb3_v3"))

    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)

    print("\n=== Start SFT RL V3 (Modular) ===")
    print(
        f"num_envs={num_envs}, test_scene={test_scene_type}, use_supervised={enable_sft_warmup}, "
        f"supervised_steps={sft_collect_total_steps}"
    )
        # 注册键盘回调
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    if keyboard is None:
        raise RuntimeError("无法获取键盘设备，无法进行控制")

    # 订阅键盘事件
    keyboard_sub = input_interface.subscribe_to_keyboard_events(
        keyboard, on_keyboard_event
    )

    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1


        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        car_local = np.stack([x_local, y_local], axis=1)
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)

        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager,
        )
        robot_velocities_np = robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(robot_velocities_torch, max_v=max_v, max_w=max_w)
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]

        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_ego, _ = compute_user_intent_torch(
            robot_pos_torch,
            torch.tensor(yaw),
            goal_pos_torch,
            env_origins_torch,
            normalize=True,
        )
        user_intent_np = user_intent_ego.cpu().numpy().astype(np.float32)

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
            user_input=user_intent_np,
        )

        obstacle_collision, _, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        total_obstacle_collision += int(np.sum(obstacle_collision))

        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)

        if True:
            for i in range(num_envs):
                if obstacle_collision[i]:
                    v = 0.0
                    w = 0.0
                else:
                    v = 0.0
                    w = 0.0

                    if key_state[carb.input.KeyboardInput.W]:
                        v += 1.0
                    if key_state[carb.input.KeyboardInput.S]:
                        v -= 1.0
                    if key_state[carb.input.KeyboardInput.A]:
                        w += 1.0
                    if key_state[carb.input.KeyboardInput.D]:
                        w -= 1.0
                    
                v_cmd[i] = v
                w_cmd[i] = w
                current_act[i, 0] = v / max_v
                current_act[i, 1] = w / max_w
                log_probs[i] = 0.0
                if not obstacle_collision[i]:
                    supervised_obs.append(obs[i])
                    supervised_actions.append(current_act[i])

        data_list.append({
            "obs": obs,
            'v':v,
            'w':w,
            'test_scene_type': test_scene_type})
        

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])
        
        action_history[:, 2:] = action_history[:, :2]
        action_history[:, 0] = v_cmd
        action_history[:, 1] = w_cmd

        robots.set_joint_velocity_targets(targets)
        world.step(render=True)

        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        reached = dist <= reset_dist

        boundary_collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            boundary_collision[i] = scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]]))
        to_reset = obstacle_collision | boundary_collision | reset_timeout | reached

        if reached:
            save_sample(data_list, test_scene_type)
            data_list = []

        next_obs = get_obs(
            robots,
            env_origins,
            goal_pos,
            lidar_num_rays,
            lidar_max_range,
            scene_manager,
            max_v,
            max_w,
            action_history,
            device=device,
        )

        rewards, reward_info = reward_composer.compute(
            dist=dist,
            yaw_err=yaw_err,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            v_measured=current_measured_v,
            w_measured=current_measured_w,
            collision=obstacle_collision,
            reached=reached,
            timeout=reset_timeout,
            min_dist=min_dist,
            current_act=current_act,
            prev_act=prev_act,
            prev_prev_act=prev_prev_act,
        )

        reward_component_count += len(rewards)
        for k, v in reward_info.items():
            if k not in reward_component_sums:
                reward_component_sums[k] = 0.0
            reward_component_sums[k] += float(np.sum(v))

        if not sft_collection_active:
            for i in range(num_envs):
                reward_buffer.append(rewards[i])
                rl_agent.store_transition(
                    obs=obs[i],
                    act=current_act[i],
                    rew=rewards[i],
                    next_obs=next_obs[i],
                    done=to_reset[i],
                    log_prob=log_probs[i],
                )



        avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
        print_phase_status(
            train_mode=("supervised" if sft_collection_active else "ppo"),
            step_count=step_count,
            supervised_steps=sft_collect_total_steps,
            supervised_obs_count=len(supervised_obs),
            update_count=ppo_update_epoch_count,
            total_epochs=ppo_total_update_epochs,
            avg_reward=avg_reward,
        )
        print(f"当前奖励: {rewards[0]:.2f}")
        print(f"yaw:{yaw}")
        print(f"goal local:{goal_local}")
        print(f"robot local:{car_local}")
        print(f"角速度：{w_cmd}")
        print(f"线速度：{v_cmd}")
        print(f"基础速度：{base_vel_np}")
        print(f"目标距离：{dist}")
        print(f"用户意图：{user_intent_np}")
        print(f"角度差：{yaw_err}")
        print(f"总碰撞次数：{total_obstacle_collision}")
        print_reward_breakdown(reward_info, env_idx=0)

        if np.any(to_reset):
            data_list = []
            reset_ids = np.nonzero(to_reset)[0]
            current_base_vels = robots.get_velocities()
            current_base_vels[reset_ids] = 0.0
            robots.set_velocities(current_base_vels)
            current_joint_vels = robots.get_joint_velocities()
            current_joint_vels[reset_ids] = 0.0
            robots.set_joint_velocities(current_joint_vels)

            for i in reset_ids:
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
                spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
                idx = np.array([i], dtype=np.int32)
                robots.set_world_poses(positions=spawn_pos.reshape(1, 3), orientations=spawn_rot.reshape(1, 4), indices=idx)
                robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
                robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                episode_start_time[i] = current_time
                action_history[i] = np.zeros(4, dtype=np.float32)
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
                prev_act[i] = np.zeros(2)
                prev_prev_act[i] = np.zeros(2)

            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)

        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        prev_prev_act = prev_act.copy()
        prev_act = current_act.copy()

        time.sleep(0.001)

    simulation_app.close()
    writer.close()
    input_interface.unsubscribe_from_keyboard_events(keyboard, keyboard_sub)
    simulation_app.close()


if __name__ == "__main__":
    main()

    