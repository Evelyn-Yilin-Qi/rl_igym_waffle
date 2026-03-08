"""
WaffleDrive 张量化环境
完美对齐论文 4 大训练场景 (Empty, Cylinder, Box, Door)
终极物理引擎解法：10吨动态刚体 + 地下错位掩埋 (杜绝死锁与爆炸)
"""

import torch
import math
import numpy as np
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
# 重新引入 Dynamic 物体和 RigidPrimView，保留 VisualCylinder
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, VisualCylinder
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx import get_physx_scene_query_interface

class WaffleDriveTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        self._num_envs = self._task_cfg["env"]["numEnvs"] 
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self.stage_level = self._task_cfg["env"]["randomization"]["stage_level"]
        
        self.max_v = self._task_cfg["env"]["robot_limits"]["max_v"]
        self.max_w = self._task_cfg["env"]["robot_limits"]["max_w"]
        self.lidar_max_range = self._task_cfg["env"]["lidar"]["max_range"]
        self.lidar_noise_std = self._task_cfg["env"]["lidar"]["noise_std"]
        
        self.w_v_track = self._task_cfg["env"]["rewards"]["v_tracking_weight"]
        self.w_heading = self._task_cfg["env"]["rewards"]["heading_penalty_weight"]
        self.w_centripetal = self._task_cfg["env"]["rewards"]["centripetal_accel_weight"]
        self.w_smooth = self._task_cfg["env"]["rewards"]["action_smooth_weight"]
        self.w_collision = self._task_cfg["env"]["rewards"]["collision_penalty"]
        self.w_critical = self._task_cfg["env"]["rewards"]["critical_penalty"]
        self.d_col = self._task_cfg["env"]["safety"]["d_col"]
        self.d_crit = self._task_cfg["env"]["safety"]["d_crit"]
        
        self.WHEEL_RADIUS = 0.033
        self.WHEEL_BASE = 0.287
        
        self._num_actions = 2
        self._num_observations = 44 
        
        super().__init__(name, env, offset)

        self.action_history = torch.zeros((self._num_envs, 4), device=self._device)
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device)
        self.physx_query_interface = get_physx_scene_query_interface()

    def set_up_scene(self, scene) -> None:
        robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path=self.default_zero_env_path + "/WafflePi")
        
        # Target 使用纯视觉件，随便移动不影响物理 BVH
        VisualCylinder(prim_path=self.default_zero_env_path + "/Target", name="target",
                        position=np.array([3.0, 0.0, 0.01]), 
                        radius=0.15, height=0.02, color=np.array([0.0, 1.0, 0.0]))

        # ==========================================================
        # 核心解法：使用 Dynamic 物体避免死锁，赋予 10000.0 的质量让其坚如磐石！
        # ==========================================================
        self.left_wall_scale = torch.tensor([0.2, 2.0, 1.0], device=self._device)
        DynamicCuboid(prim_path=self.default_zero_env_path + "/Wall_Left", name="wall_left",
                      position=np.array([1.5, 1.0, -10.0]), scale=self.left_wall_scale.cpu().numpy(), 
                      color=np.array([0.5, 0.5, 0.5]), mass=10000.0)
        
        self.right_wall_scale = torch.tensor([0.2, 2.0, 1.0], device=self._device)
        DynamicCuboid(prim_path=self.default_zero_env_path + "/Wall_Right", name="wall_right",
                       position=np.array([1.5, -1.0, -10.0]), scale=self.right_wall_scale.cpu().numpy(), 
                       color=np.array([0.5, 0.5, 0.5]), mass=10000.0)
               
        self.obstacle_scale_base = torch.tensor([0.5, 0.5, 0.5], device=self._device)
        DynamicCuboid(prim_path=self.default_zero_env_path + "/Obstacle_Box", name="obstacle_box",
                      position=np.array([2.0, 0.0, -10.0]), scale=self.obstacle_scale_base.cpu().numpy(), 
                      color=np.array([0.8, 0.2, 0.2]), mass=10000.0)

        cyl_positions = [[1.5, 0.5, -10.0], [1.5, -0.5, -10.0], [2.5, 0.5, -10.0], [2.5, -0.5, -10.0]]
        for i, pos in enumerate(cyl_positions):
            DynamicCylinder(prim_path=self.default_zero_env_path + f"/Cyl_{i}", name=f"cyl_{i}",
                            position=np.array(pos), radius=0.15, height=0.5, 
                            color=np.array([0.2, 0.5, 0.8]), mass=10000.0)

        super().set_up_scene(scene)

        self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/WafflePi", name="waffle_view")
        self.targets = XFormPrimView(prim_paths_expr="/World/envs/.*/Target", name="target_view")
        
        # 障碍物全部改回 RigidPrimView，以便在 Tensor API 层高效操控
        self.left_walls = RigidPrimView(prim_paths_expr="/World/envs/.*/Wall_Left", name="left_wall_view")
        self.right_walls = RigidPrimView(prim_paths_expr="/World/envs/.*/Wall_Right", name="right_wall_view")
        self.obs_boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/Obstacle_Box", name="obstacle_box_view")
        
        self.cyls = []
        for i in range(4):
            cyl_view = RigidPrimView(prim_paths_expr=f"/World/envs/.*/Cyl_{i}", name=f"cyl_view_{i}")
            self.cyls.append(cyl_view)
            scene.add(cyl_view)

        scene.add(self.robots)
        scene.add(self.targets)
        scene.add(self.left_walls)
        scene.add(self.right_walls)
        scene.add(self.obs_boxes)

    def post_reset(self):
        self.target_positions = self._env_pos.clone()
        self.target_positions[:, 0] += 3.0
        self.target_positions[:, 2] = 0.01

    def get_observations(self) -> dict:
        self.robot_positions, self.robot_orientations = self.robots.get_world_poses()
        robot_velocities = self.robots.get_velocities()
        
        target_vec = self.target_positions[:, :2] - self.robot_positions[:, :2]
        self.target_dist = torch.norm(target_vec, dim=-1).unsqueeze(-1)
        
        quat_norm = torch.norm(self.robot_orientations, dim=-1, keepdim=True)
        self.robot_orientations = self.robot_orientations / (quat_norm + 1e-8)
        w, x, y, z = self.robot_orientations[:, 0], self.robot_orientations[:, 1], self.robot_orientations[:, 2], self.robot_orientations[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        target_angle_abs = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        self.heading_error = target_angle_abs - yaw
        self.heading_error = torch.atan2(torch.sin(self.heading_error), torch.cos(self.heading_error))
        
        user_input = torch.cat((self.target_dist, self.heading_error.unsqueeze(-1)), dim=-1)
        base_vel = robot_velocities[:, [0, 5]] 
        
        lidar_z_offset = torch.tensor([0.0, 0.0, 0.17], device=self._device)
        ray_starts = self.robot_positions + lidar_z_offset
        angles = torch.linspace(0, 2 * math.pi, 36, device=self._device)
        global_angles = angles.unsqueeze(0) + yaw.unsqueeze(1)
        ray_dirs = torch.stack([torch.cos(global_angles), torch.sin(global_angles), torch.zeros_like(global_angles)], dim=-1)
        
        origins_expanded = ray_starts.unsqueeze(1).repeat(1, 36, 1) 
        origins_flat = origins_expanded.view(-1, 3).cpu().numpy()   
        dirs_flat = ray_dirs.view(-1, 3).cpu().numpy()              
        distances_flat = np.ones(origins_flat.shape[0]) * self.lidar_max_range

        for i in range(origins_flat.shape[0]):
            hit_info = self.physx_query_interface.raycast_closest(origins_flat[i].tolist(), dirs_flat[i].tolist(), self.lidar_max_range)
            if hit_info["hit"]:
                distances_flat[i] = hit_info["distance"]
                
        self.lidar_raw_dist = torch.tensor(distances_flat, device=self._device, dtype=torch.float32).view(self._num_envs, 36)
        
        lidar_tensor = torch.clamp(self.lidar_raw_dist, 0.0, self.lidar_max_range) / self.lidar_max_range
        noise = torch.randn_like(lidar_tensor) * self.lidar_noise_std
        lidar_tensor = torch.clamp(lidar_tensor + noise, 0.0, 1.0)
        
        self.obs_buf = torch.cat((lidar_tensor, user_input, base_vel, self.action_history), dim=-1)
        return {self.name: {"obs": self.obs_buf}}
    
    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        self.action_history = torch.roll(self.action_history, shifts=-2, dims=1)
        self.action_history[:, -2:] = actions
        
        self.current_v = actions[:, 0] * self.max_v
        self.current_w = actions[:, 1] * self.max_w
        
        left_wheel_vel = (self.current_v - self.current_w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
        right_wheel_vel = (self.current_v + self.current_w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
        
        joint_velocities = torch.zeros((self._num_envs, self.robots.num_dof), dtype=torch.float32, device=self._device)
        joint_velocities[:, 0] = left_wheel_vel
        joint_velocities[:, 1] = right_wheel_vel
        
        self.robots.set_joint_velocities(joint_velocities)

    def _hide_asset(self, view, env_ids, offset_x=0.0):
        # 错位掩埋法：每个障碍物在地下都有自己专属的位置，绝对不会因为穿模重叠而炸飞
        underground_pos = self._env_pos[env_ids].clone()
        underground_pos[:, 0] += offset_x
        underground_pos[:, 2] = -10.0 
        dummy_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(len(env_ids), 1)
        view.set_world_poses(underground_pos, dummy_rot, indices=env_ids)
        view.set_velocities(torch.zeros((len(env_ids), 6), device=self._device), indices=env_ids)

    def reset_idx(self, env_ids):
        n_resets = len(env_ids)
        env_pos = self._env_pos[env_ids]
        
        if self.stage_level == 1:
            env_types = torch.zeros_like(env_ids) 
        else:
            env_types = env_ids % 4 

        robot_new_pos = env_pos.clone()
        robot_new_pos[:, 2] += 0.05 
        target_new_pos = env_pos.clone()
        target_new_pos[:, 2] += 0.01 
        
        default_rx = torch.rand(n_resets, device=self._device) * 1.0 - 0.5
        default_ry = torch.rand(n_resets, device=self._device) * 2.0 - 1.0
        default_tx = torch.rand(n_resets, device=self._device) * 1.0 + 2.5
        default_ty = torch.rand(n_resets, device=self._device) * 2.0 - 1.0

        # 重置前，把大家都在地下错开摆好
        self._hide_asset(self.left_walls, env_ids, 10.0)
        self._hide_asset(self.right_walls, env_ids, 12.0)
        self._hide_asset(self.obs_boxes, env_ids, 14.0)
        for i, cyl_view in enumerate(self.cyls): 
            self._hide_asset(cyl_view, env_ids, 16.0 + i * 2.0)

        for type_idx in range(4):
            mask = (env_types == type_idx)
            sub_ids = env_ids[mask]
            if len(sub_ids) == 0: continue
            
            robot_new_pos[mask, 0] += default_rx[mask]
            robot_new_pos[mask, 1] += default_ry[mask]
            target_new_pos[mask, 0] += default_tx[mask]
            target_new_pos[mask, 1] += default_ty[mask]

            if type_idx == 0:
                pass # (a) Empty Env: 无障碍

            elif type_idx == 1:
                cyl_local_pos = [[1.5, 0.5], [1.5, -0.5], [2.5, 0.5], [2.5, -0.5]]
                for i, cyl_view in enumerate(self.cyls):
                    pos = env_pos[mask].clone()
                    pos[:, 0] += cyl_local_pos[i][0]
                    pos[:, 1] += cyl_local_pos[i][1]
                    pos[:, 2] = 0.25
                    rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(len(sub_ids), 1)
                    cyl_view.set_world_poses(pos, rot, indices=sub_ids)
                    cyl_view.set_velocities(torch.zeros((len(sub_ids), 6), device=self._device), indices=sub_ids)

            elif type_idx == 2:
                pos = env_pos[mask].clone()
                pos[:, 0] += 1.5
                pos[:, 1] += 0.0
                pos[:, 2] = 0.25
                rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(len(sub_ids), 1)
                self.obs_boxes.set_world_poses(pos, rot, indices=sub_ids)
                self.obs_boxes.set_velocities(torch.zeros((len(sub_ids), 6), device=self._device), indices=sub_ids)

            elif type_idx == 3:
                min_w, max_w = 0.9, 1.75
                door_widths = torch.rand(len(sub_ids), device=self._device) * (max_w - min_w) + min_w
                
                l_pos = env_pos[mask].clone()
                l_pos[:, 0] += 1.5
                l_pos[:, 1] += (door_widths / 2.0 + 1.0)
                l_pos[:, 2] = 0.5
                rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(len(sub_ids), 1)
                self.left_walls.set_world_poses(l_pos, rot, indices=sub_ids)
                self.left_walls.set_velocities(torch.zeros((len(sub_ids), 6), device=self._device), indices=sub_ids)
                
                r_pos = env_pos[mask].clone()
                r_pos[:, 0] += 1.5
                r_pos[:, 1] -= (door_widths / 2.0 + 1.0)
                r_pos[:, 2] = 0.5
                self.right_walls.set_world_poses(r_pos, rot, indices=sub_ids)
                self.right_walls.set_velocities(torch.zeros((len(sub_ids), 6), device=self._device), indices=sub_ids)

                robot_y_local = robot_new_pos[mask, 1] - env_pos[mask, 1]
                sign_y = torch.sign(robot_y_local)
                target_new_pos[mask, 1] = env_pos[mask, 1] - sign_y * (torch.rand(len(sub_ids), device=self._device) * 1.0 + 0.2)

        start_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(n_resets, 1)
        self.robots.set_world_poses(robot_new_pos, start_rotations, indices=env_ids)
        self.robots.set_velocities(torch.zeros((n_resets, 6), device=self._device), indices=env_ids)
        self.robots.set_joint_velocities(torch.zeros((n_resets, self.robots.num_dof), device=self._device), indices=env_ids)
        
        # Target 既然是纯视觉体，就可以随便移动
        self.targets.set_world_poses(target_new_pos, start_rotations, indices=env_ids)
        
        self.target_positions[env_ids] = target_new_pos
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.action_history[env_ids] = 0.0

    def calculate_metrics(self) -> None:
        centripetal_penalty = torch.abs(self.current_v * self.current_w) * self.w_centripetal
        heading_penalty = torch.abs(self.heading_error) * self.w_heading 
        
        last_actions = self.action_history[:, 0:2]
        current_normalized_actions = torch.stack([self.current_v / self.max_v, self.current_w / self.max_w], dim=-1)
        action_diff = torch.norm(current_normalized_actions - last_actions, dim=-1)
        smoothing_penalty = action_diff * self.w_smooth
        
        min_lidar_dist = self.lidar_raw_dist.min(dim=1).values
        is_collision = min_lidar_dist < self.d_col
        collision_penalty_reward = torch.where(is_collision, torch.tensor(self.w_collision, device=self._device), torch.tensor(0.0, device=self._device))
        
        is_critical = (min_lidar_dist >= self.d_col) & (min_lidar_dist < self.d_crit)
        critical_penalty_reward = torch.where(is_critical, torch.tensor(self.w_critical, device=self._device), torch.tensor(0.0, device=self._device))
        
        self.rew_buf[:] = (self.current_v * self.w_v_track + centripetal_penalty + heading_penalty + smoothing_penalty + 
                           collision_penalty_reward + critical_penalty_reward)

    def is_done(self) -> None:
        timeout = self.progress_buf >= self._max_episode_length
        reached_target = self.target_dist.squeeze(-1) < 0.25
        min_lidar_dist = self.lidar_raw_dist.min(dim=1).values
        collision = min_lidar_dist < self.d_col
        
        out_of_bounds = self.robot_positions[:, 2] < -0.1
        x, y = self.robot_orientations[:, 1], self.robot_orientations[:, 2]
        up_z = 1.0 - 2.0 * (x * x + y * y) 
        flipped = up_z < 0.90 
        
        self.reset_buf[:] = torch.where(timeout | reached_target | collision | out_of_bounds | flipped, 1, 0)