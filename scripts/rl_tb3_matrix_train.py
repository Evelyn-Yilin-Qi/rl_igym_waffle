"""
3 场景 × 5 种预训练骨干 的 PPO 矩阵训练（每个组合独立子进程 + 独立保存目录）。

编排为严格顺序：对每个 (scene, model) 阻塞式 subprocess.call，无并行多开。
若希望入口名更直观，可运行 `scripts/rl_tb3_sequential_train.py`（行为相同）。

单场景依次训 5 模型：``scripts/rl_tb3_box_models_train.py`` 等（顺序子进程调用
``rl_tb3_box.py`` / ``rl_tb3_door.py`` / ``rl_tb3_col.py`` 并带 ``--model``，与全矩阵本模块独立）。

编排入口（无 ``--worker``）请用「系统」Python，不要用 ``isaac_python`` 起本进程，否则易与子
Isaac 冲突卡死；子进程仍由 ``ISAAC_PYTHON`` / ``python.sh`` 拉起 worker::

  python3 scripts/rl_tb3_matrix_train.py

单 worker 调试仍用 Isaac 启动器::

  ./python.sh scripts/rl_tb3_matrix_train.py --worker --scene box --model simple_fc_sft

编排器子进程会自动从 ``sys.executable`` 向上查找 ``python.sh``；若布局非标准，请设置
环境变量 ``ISAAC_PYTHON`` 指向 ``python.sh``（或你的 ``isaac_python`` 实际脚本）。

预训练权重（与 train_actor.py 一致）:
  best_actor_{model}.pth  位于项目根目录

产出:
  checkpoints/ppo_matrix/{scene}_{model}/ppo_epoch_*.pth, ppo_final_*.pth
  runs/ppo_matrix/{scene}_{model}/
  训练结束后额外保存 policy 权重: checkpoints/ppo_matrix/{scene}_{model}/policy_after_rl.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_subprocess_python() -> str:
    """用于拉起 worker 子进程的启动器。

    编排进程若直接跑在 Kit 自带的 ``.../kit/python/bin/python3`` 上，
    ``sys.executable`` 再 spawn 一份同样的二进制时，往往没有 Isaac 的完整引导，
    会出现 ``No module named 'omni.isaac.core'``。与手动 ``./python.sh *.py`` 一致，
    应优先使用安装根目录下的 ``python.sh``。

    可在环境中设置 ``ISAAC_PYTHON``（指向 ``python.sh`` 或等价启动器）覆盖自动推断。
    """
    override = (os.environ.get("ISAAC_PYTHON") or "").strip()
    if override:
        return override
    exe = Path(sys.executable).resolve()
    d = exe.parent
    for _ in range(16):
        if d == d.parent:
            break
        cand = d / "python.sh"
        if cand.is_file():
            return str(cand)
        d = d.parent
    return str(exe)


SCENES = ["box", "door", "cylinder"]

MODELS = [
    "simple_fc_sft",
    "cnn_lstm_sft",
    "cnn_gru_sft",
    "fc_lstm_sft",
    "cnn_lstm_sft_nodoor",
]


def orchestrator_main_one_scene(scene_key: str) -> None:
    """固定一个场景，按 MODELS 顺序依次 spawn worker（每个子进程一次完整 Isaac 生命周期）。"""
    _scripts = str(Path(__file__).resolve().parent)
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from rl_tb3_subprocess_runner import assert_plain_python_parent, run_isaac_child_blocking

    assert_plain_python_parent()
    if scene_key not in SCENES:
        raise ValueError(f"scene_key must be one of {SCENES}, got {scene_key!r}")
    py = resolve_subprocess_python()
    script = str(Path(__file__).resolve())
    for model in MODELS:
        ck = PROJECT_ROOT / f"best_actor_{model}.pth"
        if not ck.is_file():
            print(f"[SKIP] 缺少预训练: {ck}")
            continue
        cmd = [
            py,
            script,
            "--worker",
            "--scene",
            scene_key,
            "--model",
            model,
        ]
        print("\n" + "=" * 72)
        print("RUN:", " ".join(cmd))
        if py != sys.executable:
            print(f"(worker 启动器 != sys.executable；sys.executable={sys.executable})")
        print("=" * 72)
        ret = run_isaac_child_blocking(cmd, cwd=str(PROJECT_ROOT))
        if ret != 0:
            raise SystemExit(f"子进程失败 exit={ret} scene={scene_key} model={model}")


def orchestrator_main():
    for scene_key in SCENES:
        orchestrator_main_one_scene(scene_key)


def worker_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scene", required=True, choices=["box", "door", "cylinder"])
    ap.add_argument("--model", required=True, choices=MODELS)
    args = ap.parse_args()

    # ----- Isaac 必须在子进程内首初始化 -----
    import random
    from collections import deque
    from datetime import datetime

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from omni.isaac.kit import SimulationApp
    from torch.utils.tensorboard import SummaryWriter

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from algorithms import create_algorithm
    from core_network import create_policy, create_value
    from envs.observations import (
        assemble_observations,
        check_obstacle_collision,
        compute_lidar_ranges,
        get_base_velocity_from_tensor,
        get_obs,
    )
    from envs.user_intent import compute_user_intent_torch
    from process_settings.env_setup import EnvironmentSetup
    from rewards import RewardComposer
    from sim.robot import MAX_V, MAX_W
    from sim.scenes import (
        SCENE_BOX,
        SCENE_CYLINDER,
        SCENE_DOOR,
        quat_wxyz_from_yaw,
        wrap_to_pi,
        yaw_from_quat_wxyz,
    )
    from sft import (
        RuleBasedController,
        get_network_params_by_name,
        load_config,
        print_phase_status,
        print_reward_breakdown,
        train_supervised_policy,
    )
    from utils.config_utils import get_enabled_component

    SCENE_OBJ = {"box": SCENE_BOX, "door": SCENE_DOOR, "cylinder": SCENE_CYLINDER}[args.scene]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[worker] device={device} scene={args.scene} model={args.model}")

    simulation_app = SimulationApp({"headless": False})

    def save_model(rl_agent, checkpoint_dir, tag):
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
        rl_agent.save(save_path)
        print(f"[Checkpoint] Saved => {save_path}")

    test_scene_type = SCENE_OBJ
    num_envs = 1
    scene_types = [test_scene_type]

    env_cfg = OmegaConf.create(
        {
            "env": {
                "numEnvs": num_envs,
                "resetDist": 0.8,
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
    network_cfg = load_config(str(PROJECT_ROOT), "config/network_config.yaml")
    algo_cfg = load_config(str(PROJECT_ROOT), "config/algorithm_config.yaml")
    reward_cfg = load_config(str(PROJECT_ROOT), "config/reward_config.yaml")

    reset_dist = float(env_cfg.env.resetDist)
    show_visual_walls = None
    TIMEOUT_SECONDS = 600.0
    physics_dt = float(env_cfg.sim.dt)
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    DCOL = 0.2
    DCRIT = 0.5

    sft_collection_active = False
    sft_collect_total_steps = 200000
    sft_opt_epochs = 50

    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    action_history = np.zeros((num_envs, 4), dtype=np.float32)
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

    network_type = args.model
    network_params = get_network_params_by_name(network_cfg, "simple_fc_sft")
    network_params.update(
        {
            "obs_dim": 364,
            "act_dim": 2,
            "lidar_dim": lidar_num_rays,
            "state_dim": 4,
        }
    )
    if network_type == "simple_fc_sft":
        network_params["hidden_dim"] = int(network_params.get("hidden_dim", 128))

    policy = create_policy(network_type, **network_params)
    actor_ckpt = PROJECT_ROOT / f"best_actor_{network_type}.pth"
    policy.load_state_dict(torch.load(str(actor_ckpt), map_location="cpu"))
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

    rule_controller = RuleBasedController(
        max_v=max_v,
        max_w=max_w,
        safe_dist=1.0,
        collision_dist=DCOL,
        v_gain=0.4,
        w_gain=1.8,
        obstacle_v_gain=0.3,
        obstacle_w_gain=2.2,
        num_rays=lidar_num_rays,
    )

    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)

    ppo_update_interval_steps = 4096
    ckpt_save_every_update_epochs = 10
    ppo_total_update_epochs = 500
    ppo_update_epoch_count = 0
    step_count = -1
    current_time = 0.0
    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    supervised_obs = []
    supervised_actions = []
    total_obstacle_collision = 0

    run_tag = f"{args.scene}_{args.model}"
    checkpoint_dir = os.path.join(str(PROJECT_ROOT), "checkpoints", "ppo_matrix", run_tag)
    writer = SummaryWriter(log_dir=os.path.join(str(PROJECT_ROOT), "runs", "ppo_matrix", run_tag))
    print(f"[I/O] checkpoints => {checkpoint_dir}\n[I/O] tensorboard => {writer.log_dir}")

    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)

    print(f"\n=== Matrix RL | scene={args.scene} | model={args.model} ===")

    try:
        while simulation_app.is_running():
            current_time += physics_dt
            step_count += 1

            pos, rot = robots.get_world_poses()
            yaw = yaw_from_quat_wxyz(rot)
            x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
            y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
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

            if sft_collection_active and step_count < sft_collect_total_steps:
                for i in range(num_envs):
                    if obstacle_collision[i]:
                        v = 0.0
                        w = 0.0
                    else:
                        v, w = rule_controller.compute_velocity_commands(
                            lidar_ranges=lidar_ranges[i],
                            user_intent=user_intent_np[i],
                            base_vel=base_vel_np[i],
                        )
                    v_cmd[i] = v
                    w_cmd[i] = w
                    current_act[i, 0] = v / max_v
                    current_act[i, 1] = w / max_w
                    log_probs[i] = 0.0
                    if not obstacle_collision[i]:
                        supervised_obs.append(obs[i])
                        supervised_actions.append(current_act[i])
            else:
                if sft_collection_active:
                    final_loss = train_supervised_policy(
                        rl_agent,
                        np.array(supervised_obs, dtype=np.float32),
                        np.array(supervised_actions, dtype=np.float32),
                        device=device,
                        epochs=sft_opt_epochs,
                    )
                    save_model(rl_agent, checkpoint_dir, "supervised_final")
                    print(f"[SFT] done, final loss = {final_loss}")
                    sft_collection_active = False
                    print("=== Switched to PPO finetuning ===")

                for i in range(num_envs):
                    if obstacle_collision[i]:
                        act = np.array([0.0, 0.0], dtype=np.float32)
                        log_prob = 0.0
                    else:
                        act, log_prob = rl_agent.select_action(obs[i])
                        temp_cur_act[i] = act
                        temp_logprob[i] = log_prob

                    log_probs[i] = log_prob
                    current_act[i] = act
                    v_cmd[i] = np.clip(act[0] * max_v, 0.0, max_v)
                    w_cmd[i] = np.clip(act[1] * max_w, -max_w, max_w)

            targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
            for i in range(num_envs):
                action = diff_ctrl.forward(
                    command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32)
                )
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
                boundary_collision[i] = scene_manager.check_boundary_collision(
                    i, np.array([x_local[i], y_local[i]])
                )
            to_reset = obstacle_collision | boundary_collision | reset_timeout | reached

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

                if (
                    step_count % ppo_update_interval_steps == 0
                    and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size
                ):
                    train_info = rl_agent.update()
                    ppo_update_epoch_count += 1
                    avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
                    writer.add_scalar("train/actor_loss", train_info["actor_loss"], ppo_update_epoch_count)
                    writer.add_scalar("train/critic_loss", train_info["critic_loss"], ppo_update_epoch_count)
                    writer.add_scalar("train/entropy", train_info["entropy"], ppo_update_epoch_count)
                    writer.add_scalar("train/approx_kl", train_info["approx_kl"], ppo_update_epoch_count)
                    writer.add_scalar("reward/total_mean", avg_reward, ppo_update_epoch_count)
                    if reward_component_count > 0:
                        for k, v_sum in reward_component_sums.items():
                            writer.add_scalar(
                                f"reward/{k}", v_sum / reward_component_count, ppo_update_epoch_count
                            )
                    for k in reward_component_sums.keys():
                        reward_component_sums[k] = 0.0
                    reward_component_count = 0

                    if ppo_update_epoch_count % ckpt_save_every_update_epochs == 0:
                        save_model(rl_agent, checkpoint_dir, f"epoch_{ppo_update_epoch_count:03d}")
                    if ppo_update_epoch_count >= ppo_total_update_epochs:
                        save_model(rl_agent, checkpoint_dir, f"final_{datetime.now().strftime('%Y%m%d_%H')}")
                        torch.save(
                            rl_agent.policy.state_dict(),
                            os.path.join(checkpoint_dir, "policy_after_rl.pth"),
                        )
                        break

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
            print_reward_breakdown(reward_info, env_idx=0)

            if np.any(to_reset):
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
                    robots.set_world_poses(
                        positions=spawn_pos.reshape(1, 3),
                        orientations=spawn_rot.reshape(1, 4),
                        indices=idx,
                    )
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

            if ppo_update_epoch_count >= ppo_total_update_epochs:
                break

            time.sleep(0.001)
    finally:
        writer.close()
        simulation_app.close()

    print(f"[worker] 完成 scene={args.scene} model={args.model}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker_main()
    else:
        orchestrator_main()


if __name__ == "__main__":
    main()
