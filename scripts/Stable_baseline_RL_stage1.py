"""
Stable-Baselines3 版 Stage 1 训练脚本
- 保持与 2_RL_stage1_test.py 相同的仿真、场景加载、重置逻辑
- 保持观察与动作含义不变：obs 维度 44，动作 [-1, 1]^2 反归一化为 [v, w]
- 使用自定义 VecEnv 封装 Isaac 多环境（单进程多机器人）
"""
import os
import sys
import time
from typing import List, Optional, Tuple, Union, Any

import numpy as np
import torch
from gym import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import configure as sb3_logger_configure

# 先把项目根目录放进路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Isaac Sim 必须最先初始化
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.physx import get_physx_scene_query_interface
from pxr import Gf

from sim.robot import (
    TB3_USD,
    WHEEL_RADIUS,
    WHEEL_BASE,
    apply_massapi_all_tb3,
    configure_wheel_joints,
)
from sim.scenes import (
    SCENE_EMPTY,
    SceneManager,
    yaw_from_quat_wxyz,
    quat_wxyz_from_yaw,
    wrap_to_pi,
)
from envs.observations import assemble_observations, get_base_velocity_from_tensor
from envs.user_intent import compute_user_intent_torch
from envs.rewards import RewardCalculator


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, cfg_path)
    return OmegaConf.load(cfg_path)


def compute_lidar_ranges(robot_positions, robot_orientations, env_origins,
                         lidar_num_rays, lidar_max_range, scene_manager):
    """与 2_RL_stage1_test.py 保持一致的射线 LiDAR 计算"""
    num_envs = robot_positions.shape[0]
    lidar_ranges = np.ones((num_envs, lidar_num_rays), dtype=np.float32) * lidar_max_range
    physx_interface = get_physx_scene_query_interface()
    lidar_height = 0.1

    for env_id in range(num_envs):
        robot_pos = robot_positions[env_id]
        robot_rot = robot_orientations[env_id]
        env_origin = env_origins[env_id]

        robot_pos_local = np.array([robot_pos[0] - env_origin[0], robot_pos[1] - env_origin[1]])

        w, x, y, z = robot_rot[0], robot_rot[1], robot_rot[2], robot_rot[3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)

        for ray_idx, angle in enumerate(angles):
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0.0])
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ray_dir_world = np.array([
                ray_dir_local[0] * cos_yaw - ray_dir_local[1] * sin_yaw,
                ray_dir_local[0] * sin_yaw + ray_dir_local[1] * cos_yaw,
                0.0
            ])

            ray_origin = Gf.Vec3f(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + lidar_height))

            ray_dir_norm = np.linalg.norm(ray_dir_world)
            if ray_dir_norm > 1e-6:
                ray_dir_normalized = ray_dir_world / ray_dir_norm
            else:
                ray_dir_normalized = np.array([1.0, 0.0, 0.0])
            ray_dir = Gf.Vec3f(float(ray_dir_normalized[0]), float(ray_dir_normalized[1]), float(ray_dir_normalized[2]))

            hit = physx_interface.raycast_closest(ray_origin, ray_dir, float(lidar_max_range), bothSides=False)

            if hit:
                distance = hit.get("distance", lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
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
                else:
                    lidar_ranges[env_id, ray_idx] = lidar_max_range

    return lidar_ranges


class IsaacWaffleVecEnv(VecEnv):
    """单进程多环境 VecEnv，兼容 SB3，封装当前 Isaac 多机器人场景"""

    def __init__(
        self,
        world: World,
        robots: ArticulationView,
        scene_manager: SceneManager,
        reward_calculator: RewardCalculator,
        diff_ctrl: DifferentialController,
        env_origins: np.ndarray,
        goal_pos: np.ndarray,
        markers: List[VisualSphere],
        left_idx: int,
        right_idx: int,
        lidar_num_rays: int,
        lidar_max_range: float,
        max_v: float,
        max_w: float,
        reset_dist: float,
        timeout_seconds: float,
        physics_dt: float,
        rng: np.random.Generator,
        device: str,
    ):
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
        self.timeout_seconds = timeout_seconds
        self.physics_dt = physics_dt
        self.rng = rng
        self.device = device

        self.num_envs = len(env_origins)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 状态缓存
        self.current_time = 0.0
        self.episode_start_time = np.zeros((self.num_envs,), dtype=np.float32)
        self.action_history = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.prev_measured_v = np.zeros((self.num_envs,), dtype=np.float32)
        self.prev_measured_w = np.zeros((self.num_envs,), dtype=np.float32)

        # 立刻重置
        self._reset_all()

        super().__init__(num_envs=self.num_envs, observation_space=self.observation_space, action_space=self.action_space)

    # ------------- VecEnv 接口 -------------
    def reset(self) -> np.ndarray:
        self._reset_all()
        obs = self._compute_obs()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        # 1. 反归一化动作
        actions_np = np.clip(np.asarray(self._actions), -1.0, 1.0)
        v_cmd = actions_np[:, 0] * self.max_v
        w_cmd = actions_np[:, 1] * self.max_w

        # 2. 转换为轮速并应用
        targets = np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32)
        for i in range(self.num_envs):
            action = self.diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, self.left_idx] = float(action.joint_velocities[0])
            targets[i, self.right_idx] = float(action.joint_velocities[1])
        self.robots.set_joint_velocity_targets(targets)

        # 3. 仿真步进
        self.world.step(render=True)
        self.current_time += self.physics_dt

        # 4. 获取状态
        pos, rot = self.robots.get_world_poses()
        robot_velocities_np = self.robots.get_velocities()
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]

        # 5. 奖励计算
        robot_pos_torch = torch.from_numpy(pos).float().to(self.device)
        robot_rot_torch = torch.from_numpy(rot).float().to(self.device)
        goal_pos_torch = torch.from_numpy(self.goal_pos).float().to(self.device)
        env_origins_torch = torch.from_numpy(self.env_origins).float().to(self.device)
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(self.device)

        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch, robot_rot_torch, goal_pos_torch, env_origins_torch, normalize=True
        )

        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=self.env_origins,
            lidar_num_rays=self.lidar_num_rays,
            lidar_max_range=self.lidar_max_range,
            scene_manager=self.scene_manager,
        )

        rewards_torch = self.reward_calculator.compute_rewards(
            lidar_ranges=torch.from_numpy(lidar_ranges).float().to(self.device),
            robot_positions=robot_pos_torch,
            robot_velocities=robot_velocities_torch,
            robot_orientations=robot_rot_torch,
            goal_positions=goal_pos_torch,
            env_origins=env_origins_torch,
            actions=torch.from_numpy(actions_np).float().to(self.device),
            action_history=torch.from_numpy(self.action_history).float().to(self.device),
            user_intent_env=user_intent_env,
        )
        rewards = rewards_torch.detach().cpu().numpy()

        # 6. Done 判定
        x_local = (pos[:, 0] - self.env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - self.env_origins[:, 1]).astype(np.float32)
        goal_offsets = self.goal_pos[:, :2] - self.env_origins[:, :2]
        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)
        dist = np.sqrt((gx_local - x_local) ** 2 + (gy_local - y_local) ** 2).astype(np.float32)

        reset_timeout = (self.current_time - self.episode_start_time) >= self.timeout_seconds
        reached = dist <= self.reset_dist
        collision = np.zeros((self.num_envs,), dtype=bool)
        for i in range(self.num_envs):
            if self.scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]])):
                collision[i] = True

        dones = (reset_timeout | reached | collision).astype(np.bool_)

        # 7. 重置 dones 的环境
        if dones.any():
            self._reset_envs(np.nonzero(dones)[0])
            # 重置后回传重置奖励为 0
            rewards[dones] = 0.0

        # 8. 更新动作历史
        self.action_history[:, 2:] = self.action_history[:, :2].copy()
        self.action_history[:, 0] = self.prev_measured_v.copy()
        self.action_history[:, 1] = self.prev_measured_w.copy()
        self.prev_measured_v = current_measured_v.copy()
        self.prev_measured_w = current_measured_w.copy()

        # 9. 计算下一个观察
        obs = self._compute_obs()

        # SB3 需要 list[dict]，每个 env 一个 info
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards.astype(np.float32), dones.astype(np.bool_), infos

    def close(self):
        simulation_app.close()

    # ============= VecEnv 抽象方法补全（SB3 需要） =============
    def get_attr(self, attr_name: str, indices: Optional[Union[int, List[int]]] = None) -> List[Any]:
        idxs = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in idxs]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, List[int]]] = None) -> None:
        idxs = self._get_indices(indices)
        for _ in idxs:
            setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args, indices: Optional[Union[int, List[int]]] = None, **method_kwargs
    ) -> List[Any]:
        idxs = self._get_indices(indices)
        results = []
        for _ in idxs:
            if hasattr(self, method_name):
                results.append(getattr(self, method_name)(*method_args, **method_kwargs))
            else:
                results.append(None)
        return results

    def env_is_wrapped(self, wrapper_class, indices: Optional[Union[int, List[int]]] = None) -> List[bool]:
        idxs = self._get_indices(indices)
        return [False for _ in idxs]

    # ------------- 内部工具函数 -------------
    def _reset_all(self):
        self.episode_start_time[:] = 0.0
        self.action_history[:] = 0.0
        self.prev_measured_v[:] = 0.0
        self.prev_measured_w[:] = 0.0
        for i in range(self.num_envs):
            self._reset_single(i, reset_goal=True, reset_time=True)
        # 稳定若干步
        for _ in range(8):
            self.robots.set_joint_velocity_targets(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            self.robots.set_velocities(np.zeros((self.num_envs, 6), dtype=np.float32))
            self.robots.set_joint_velocities(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            self.world.step(render=True)
        self.current_time = 0.0

    def _reset_envs(self, ids: np.ndarray):
        for i in ids:
            self._reset_single(int(i), reset_goal=True, reset_time=True)
        for _ in range(4):
            self.robots.set_joint_velocity_targets(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            self.robots.set_velocities(np.zeros((self.num_envs, 6), dtype=np.float32))
            self.robots.set_joint_velocities(np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32))
            self.world.step(render=True)

    def _reset_single(self, i: int, reset_goal: bool = True, reset_time: bool = True):
        self.scene_manager.reset_scene_obstacles(np.array([i]), rng=self.rng)
        spawn_pos, spawn_yaw = self.scene_manager.get_robot_spawn_config(i, rng=self.rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))

        idx = np.array([i], dtype=np.int32)
        self.robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=idx,
        )
        self.robots.set_velocities(velocities=np.zeros((1, 6), dtype=np.float32), indices=idx)
        self.robots.set_joint_velocities(velocities=np.zeros((1, self.robots.num_dof), dtype=np.float32), indices=idx)
        self.robots.set_joint_velocity_targets(np.zeros((1, self.robots.num_dof), dtype=np.float32), indices=idx)

        if reset_goal:
            self.goal_pos[i] = self.scene_manager.get_goal_config(i, rng=self.rng)
            self.markers[i].set_world_pose(position=self.goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
        if reset_time:
            self.episode_start_time[i] = self.current_time
        self.action_history[i] = 0.0
        self.prev_measured_v[i] = 0.0
        self.prev_measured_w[i] = 0.0

    def _compute_obs(self) -> np.ndarray:
        pos, rot = self.robots.get_world_poses()
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=self.env_origins,
            lidar_num_rays=self.lidar_num_rays,
            lidar_max_range=self.lidar_max_range,
            scene_manager=self.scene_manager,
        )
        robot_velocities_np = self.robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
        base_vel_np = get_base_velocity_from_tensor(robot_velocities_torch, max_v=self.max_v, max_w=self.max_w)

        robot_pos_torch = torch.from_numpy(pos).float()
        robot_rot_torch = torch.from_numpy(rot).float()
        goal_pos_torch = torch.from_numpy(self.goal_pos).float()
        env_origins_torch = torch.from_numpy(self.env_origins).float()

        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch, robot_rot_torch, goal_pos_torch, env_origins_torch, normalize=True
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
            user_input=user_intent_np,
        )
        return obs.astype(np.float32)


def build_world_and_env(device: str):
    cfg = load_config()

    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    env_gap = 2.0
    env_spacing = env_size + env_gap
    reset_dist = float(cfg.env.resetDist)
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    show_visual_walls = bool(cfg.env.scene.show_visual_walls)
    timeout_seconds = 60.0
    physics_dt = float(cfg.sim.dt)
    render_dt = 1.0 / 60.0
    max_v = float(cfg.env.robot_limits.max_v)
    max_w = float(cfg.env.robot_limits.max_w)
    lidar_num_rays = int(cfg.env.lidar.num_rays)
    lidar_max_range = float(cfg.env.lidar.max_range)

    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage

    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing

    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    for _ in range(180):
        world.step(render=True)

    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()

    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)

    left_idx, right_idx = configure_wheel_joints(robots)
    if left_idx is None or right_idx is None:
        raise RuntimeError("Could not find wheel joints in dof_names.")

    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=max_v,
        max_angular_speed=max_w,
    )

    scene_types = [SCENE_EMPTY] * num_envs
    show_visual_walls_list = [show_visual_walls] * num_envs
    scene_manager = SceneManager(num_envs=num_envs, env_size=env_size, env_origins=env_origins, stage=stage)
    scene_manager.wall_thickness = wall_thickness
    scene_manager.wall_height = wall_height
    scene_manager.create_scene_obstacles(scene_types=scene_types, show_visual_walls=show_visual_walls_list)

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

    reward_calculator = RewardCalculator(
        num_envs=num_envs,
        device=device,
        ra=float(cfg.env.rewards.get("ra", 0.5)),
        rl=float(cfg.env.rewards.get("rl", -0.5)),
        rh=float(cfg.env.rewards.get("rh", -0.5)),
        phi_thresh=float(cfg.env.rewards.get("phi_thresh", 0.2)),
        ras=float(cfg.env.rewards.get("ras", -0.02)),
        rc=float(cfg.env.rewards.get("rc", -1.0)),
        rcrit=float(cfg.env.rewards.get("rcrit", -1.0)),
        rcol=float(cfg.env.rewards.get("rcol", -100.0)),
        d_col=float(cfg.env.rewards.get("d_col", 0.12)),
        d_crit=float(cfg.env.rewards.get("d_crit", 0.35)),
        max_v=max_v,
        max_w=max_w,
    )

    vec_env = IsaacWaffleVecEnv(
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
        timeout_seconds=timeout_seconds,
        physics_dt=physics_dt,
        rng=rng,
        device=device,
    )

    return vec_env


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # SB3 logger (TensorBoard)
    log_dir = os.path.join(PROJECT_ROOT, "runs", "sb3_stage1")
    os.makedirs(log_dir, exist_ok=True)
    sb3_logger = sb3_logger_configure(log_dir, ["tensorboard", "stdout"])

    # 构建 env
    env = build_world_and_env(device=device)

    # 注意：这里使用 SB3 默认 MlpPolicy；如果需要换成当前 CNN+LSTM，可后续自定义 Policy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        n_steps=8 * env.num_envs,  # 对齐 rollouts=8, num_envs=4 => 32
        batch_size=32,
        n_epochs=4,
        learning_rate=5e-4,
        clip_range=0.2,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.2,
    )

    total_timesteps = 10000
    try:
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_SB3")
    finally:
        env.close()
        print("Training finished, env closed.")


if __name__ == "__main__":
    main()
