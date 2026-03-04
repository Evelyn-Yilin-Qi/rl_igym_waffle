# scripts/tb3_two_env_goal_reset_loop.py
# 运行：
#   isaac_python scripts/tb3_two_env_goal_reset_loop.py

import time
import math
import torch


def main():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "test": False,
        "task": {
            "name": "TB3_TwoEnv_GoalReset",
            "physics_engine": "physx",
            "env": {"numEnvs": 2, "envSpacing": 4.0, "maxEpisodeLength": 1000000},
            "sim": {
                "dt": 1.0 / 60.0,
                "substeps": 4,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
                    "contact_offset": 0.02,
                    "rest_offset": 0.0,
                    "bounce_threshold_velocity": 0.2,
                },
            },
        },
        "sim_device": "gpu",
        "device_id": 0,
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": False,
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    })
    OmegaConf.set_struct(cfg, False)

    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage

    sim_config = SimConfig(cfg)

    class TB3TwoEnvGoalReset(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config

            self._num_envs = int(self._task_cfg["env"]["numEnvs"])
            self._env_spacing = float(self._task_cfg["env"]["envSpacing"])

            self._num_observations = 1
            self._num_actions = 2

            self.robot_usd_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

            self.WHEEL_RADIUS = 0.033
            self.WHEEL_BASE = 0.287

            self.goal_offset_x = 1.0
            self.goal_reach_thresh = 0.10

            self.target_v = 0.08
            self.max_w = 1.8
            self.heading_k = 2.0

            self.warmup_steps = int(1.0 / self._task_cfg["sim"]["dt"])
            self.ramp_steps = int(2.0 / self._task_cfg["sim"]["dt"])

            self.drive_kp = 0.0
            self.drive_kd = 80.0
            self.max_effort = 8.0

            self.left_wheel_idx = None
            self.right_wheel_idx = None

            super().__init__(name, env, offset)

            self.local_step = torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)
            self.loop_count = torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)

            # wxyz
            self.identity_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            # debug
            self._dbg_once = False

        def set_up_scene(self, scene):
            add_reference_to_stage(self.robot_usd_path, self.default_zero_env_path + "/TB3")
            super().set_up_scene(scene)

            # 关键：绑定到真正的根（你截图里的 base_footprint）
            # 如果跑不起来/没 DOF，再换成 base_link：
            # "/World/envs/.*/TB3/a__namespace_base_link"
            self.robots = ArticulationView(
                prim_paths_expr="/World/envs/.*/TB3/a__namespace_base_footprint",
                name="tb3_view",
                reset_xform_properties=False
            )
            scene.add(self.robots)

        @staticmethod
        def _yaw_from_quat_wxyz(q_wxyz):
            w = q_wxyz[..., 0]
            x = q_wxyz[..., 1]
            y = q_wxyz[..., 2]
            z = q_wxyz[..., 3]
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(t3, t4)

        def _build_spawn_and_goal(self):
            env_pos = self._env_pos.clone()
            spawn = env_pos.clone()
            spawn[:, 2] = 0.10
            goal = env_pos.clone()
            goal[:, 0] = goal[:, 0] + self.goal_offset_x
            goal[:, 2] = 0.0
            return spawn, goal

        def post_reset(self):
            dof_names = self.robots.dof_names
            print("\n========== DOF NAMES ==========")
            for i, n in enumerate(dof_names):
                print(f"{i:02d}: {n}")
            print("================================\n")

            for i, n in enumerate(dof_names):
                s = n.lower()
                if self.left_wheel_idx is None and "wheel_left_joint" in s:
                    self.left_wheel_idx = i
                if self.right_wheel_idx is None and "wheel_right_joint" in s:
                    self.right_wheel_idx = i

            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                raise RuntimeError("没找到 wheel_left_joint / wheel_right_joint。请按 DOF NAMES 改匹配。")

            # gains/effort (optional)
            try:
                kps, kds = self.robots.get_gains()
                kps[:, self.left_wheel_idx] = self.drive_kp
                kps[:, self.right_wheel_idx] = self.drive_kp
                kds[:, self.left_wheel_idx] = self.drive_kd
                kds[:, self.right_wheel_idx] = self.drive_kd
                self.robots.set_gains(kps=kps, kds=kds)
            except Exception as e:
                print(f"[WARN] set_gains failed: {e}")

            try:
                max_eff = self.robots.get_max_efforts()
                if max_eff is not None:
                    max_eff[:, self.left_wheel_idx] = self.max_effort
                    max_eff[:, self.right_wheel_idx] = self.max_effort
                    self.robots.set_max_efforts(max_eff)
            except Exception as e:
                print(f"[WARN] set_max_efforts failed: {e}")

            env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(env_ids)

        def reset_idx(self, env_ids):
            n = len(env_ids)
            spawn, _ = self._build_spawn_and_goal()

            pos = spawn[env_ids].clone()
            rot = self.identity_wxyz.repeat(n, 1)

            self.robots.set_world_poses(pos, rot, indices=env_ids)
            self.robots.set_velocities(torch.zeros((n, 6), device=self._device), indices=env_ids)
            self.robots.set_joint_velocities(
                torch.zeros((n, self.robots.num_dof), device=self._device),
                indices=env_ids
            )

            self.local_step[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0

        def pre_physics_step(self, actions):
            # 一次性 debug：确认绑定的 prim、以及 set_world_poses 是否能写进去
            if not self._dbg_once:
                import sys
                print("[DBG] robots.count =", self.robots.count, flush=True)
                try:
                    print("[DBG] prim_paths =", self.robots.prim_paths, flush=True)
                except Exception as e:
                    print("[DBG] prim_paths not available:", e, flush=True)
                print("[DBG] dof_names_len =", len(self.robots.dof_names), flush=True)

                pos0, rot0 = self.robots.get_world_poses()
                pos1 = pos0.clone()
                pos1[:, 1] += 0.5
                self.robots.set_world_poses(pos1, rot0)
                pos_after, _ = self.robots.get_world_poses()
                dy = (pos_after[:, 1] - pos0[:, 1]).detach().cpu().numpy().tolist()
                print("[DBG] teleport delta_y =", dy, flush=True)
                sys.stdout.flush()
                self._dbg_once = True

            # reset（放最前）
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
                self.loop_count[reset_env_ids] += 1

            self.local_step += 1

            pos, rot = self.robots.get_world_poses()
            _, goal = self._build_spawn_and_goal()

            d = goal[:, 0:2] - pos[:, 0:2]
            dist = torch.sqrt(torch.sum(d * d, dim=-1))
            reached = dist < self.goal_reach_thresh
            self.reset_buf[:] = torch.where(reached, 1, 0)

            yaw = self._yaw_from_quat_wxyz(rot)
            goal_heading = torch.atan2(d[:, 1], d[:, 0])
            err = goal_heading - yaw
            err = torch.atan2(torch.sin(err), torch.cos(err))

            after_warmup = torch.clamp(self.local_step - self.warmup_steps, min=0)
            alpha = torch.clamp(after_warmup.to(torch.float32) / float(max(1, self.ramp_steps)), 0.0, 1.0)
            v = alpha * self.target_v

            w = torch.clamp(self.heading_k * err, -self.max_w, self.max_w)

            left_w = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            right_w = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS

            joint_velocities = torch.zeros((self._num_envs, self.robots.num_dof), dtype=torch.float32, device=self._device)
            joint_velocities[:, self.left_wheel_idx] = left_w
            joint_velocities[:, self.right_wheel_idx] = right_w
            self.robots.set_joint_velocities(joint_velocities)

        def get_observations(self):
            self.obs_buf = torch.zeros((self._num_envs, 1), device=self._device)
            return {self.name: {"obs": self.obs_buf}}

        def calculate_metrics(self):
            self.rew_buf[:] = 0.0

        def is_done(self):
            pass

    task = TB3TwoEnvGoalReset(name="TB3_TwoEnv_GoalReset", sim_config=sim_config, env=vec_env)
    vec_env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

    dt = float(cfg.task.sim.dt)
    print(f"[INFO] dt={dt}, num_envs={cfg.task.env.numEnvs}")
    print("[INFO] Running... Ctrl+C to stop.\n")

    step = 0
    log_every = int(0.5 / dt)

    try:
        while True:
            actions = torch.zeros((task._num_envs, 2), device=task._device)
            vec_env.step(actions)
            step += 1

            if step % log_every == 0:
                pos, _ = task.robots.get_world_poses()
                loops = task.loop_count.detach().cpu().numpy().tolist()
                p0 = pos[0].detach().cpu().numpy()
                p1 = pos[1].detach().cpu().numpy()
                print(
                    f"t={step*dt:6.2f}s  "
                    f"env0_pos=({p0[0]:+.2f},{p0[1]:+.2f},{p0[2]:+.2f})  "
                    f"env1_pos=({p1[0]:+.2f},{p1[1]:+.2f},{p1[2]:+.2f})  "
                    f"loops={loops}"
                )

            time.sleep(0.0005)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()