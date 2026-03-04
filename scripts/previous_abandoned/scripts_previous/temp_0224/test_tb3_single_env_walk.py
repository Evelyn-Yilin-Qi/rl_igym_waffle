"""
TB3 单环境稳定前进示例（OmniIsaacGymEnvs / Isaac Gym 风格）
- 只加载 1 个环境
- TB3 自动匀速前进（差速两轮速度目标）
- 稳定优先物理参数
- 退出用方案C：os._exit(0) 直接硬退出，绕过 Kit 清理阶段的 segfault

运行：
/isaac-sim/python.sh test_tb3_single_env_walk_exitC.py
"""

import math
import torch
import os

def main():
    print("[1] import OmegaConf")
    from omegaconf import OmegaConf

    num_envs = 1
    env_spacing = 4.0

    cfg = OmegaConf.create({
        "task": {
            "name": "TB3SingleEnvTask",
            "physics_engine": "physx",
            "env": {
                "numEnvs": num_envs,
                "envSpacing": env_spacing,
                "maxEpisodeLength": 100000,
            },
            "sim": {
                "dt": 0.005,
                "substeps": 4,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "use_gpu": True,
                    "num_threads": 4,
                    "solver_type": 1,
                    # 如果你的版本报 unknown key，就删掉对应项
                    "position_iterations": 16,
                    "velocity_iterations": 4,
                    "contact_offset": 0.02,
                    "rest_offset": 0.0,
                    "bounce_threshold_velocity": 0.2,
                }
            }
        },
        "sim_device": "gpu",
        "device_id": 0,
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": False,
        "test": False,
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    })
    OmegaConf.set_struct(cfg, False)

    print("[2] import VecEnvRLGames")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

    print("[3] create VecEnvRLGames")
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)

    print("[4] import modules")
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omniisaacgymenvs.tasks.base.rl_task import RLTask

    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats

    from omni.isaac.core.objects.ground_plane import GroundPlane
    from omni.isaac.core.materials.physics_material import PhysicsMaterial

    sim_config = SimConfig(cfg)

    class TB3SingleEnvTask(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config

            self._num_envs = self._task_cfg["env"]["numEnvs"]
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2

            self.robot_usd_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

            self.WHEEL_RADIUS = 0.033
            self.WHEEL_BASE = 0.287

            self.left_wheel_idx = None
            self.right_wheel_idx = None

            self.target_v = 0.10
            self.current_v = 0.0
            self.v_ramp_per_step = 0.001
            self.sim_step_count = 0
            self.drive_warmup_steps = 40  # 0.2s

            super().__init__(name, env, offset)

        def set_up_scene(self, scene):
            # Ground + 显式摩擦
            ground_mat = PhysicsMaterial(
                prim_path="/World/PhysicsMaterials/ground_mat",
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            )
            ground = GroundPlane(
                prim_path="/World/groundPlane",
                name="groundPlane",
                z_position=0.0,
            )
            ground.apply_physics_material(ground_mat)
            scene.add(ground)

            # Robot
            add_reference_to_stage(
                usd_path=self.robot_usd_path,
                prim_path=self.default_zero_env_path + "/TB3"
            )
            super().set_up_scene(scene)

            self.robots = ArticulationView(
                prim_paths_expr="/World/envs/.*/TB3",
                name="tb3_view",
                reset_xform_properties=False,
            )
            scene.add(self.robots)

        def post_reset(self):
            dof_names = self.robots.dof_names
            num_dof = self.robots.num_dof
            print(f"[post_reset] num_dof={num_dof}, dof_names={dof_names}")

            for i, n in enumerate(dof_names):
                nl = n.lower()
                if "wheel_left_joint" in nl:
                    self.left_wheel_idx = i
                if "wheel_right_joint" in nl:
                    self.right_wheel_idx = i

            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                if num_dof >= 2:
                    self.left_wheel_idx, self.right_wheel_idx = 0, 1
                    print("[post_reset] wheel name not matched, fallback idx: 0,1")
                else:
                    raise RuntimeError("num_dof < 2, cannot drive wheels")

            print(f"[post_reset] left_idx={self.left_wheel_idx}, right_idx={self.right_wheel_idx}")

            # drive：kp=0，kd=20，max_effort=0.1
            self._configure_wheel_drive_runtime(kp=0.0, kd=20.0, max_effort=0.1)

            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int64)
            self.reset_idx(env_ids)

        def _configure_wheel_drive_runtime(self, kp: float, kd: float, max_effort: float):
            if not (hasattr(self.robots, "get_gains") and hasattr(self.robots, "set_gains")):
                print("[drive] missing get_gains/set_gains, skip")
                return

            kps, kds = self.robots.get_gains()
            kps[:, self.left_wheel_idx] = kp
            kps[:, self.right_wheel_idx] = kp
            kds[:, self.left_wheel_idx] = kd
            kds[:, self.right_wheel_idx] = kd
            self.robots.set_gains(kps=kps, kds=kds)

            if hasattr(self.robots, "get_max_efforts") and hasattr(self.robots, "set_max_efforts"):
                me = self.robots.get_max_efforts()
                me[:, self.left_wheel_idx] = max_effort
                me[:, self.right_wheel_idx] = max_effort
                self.robots.set_max_efforts(me)

            kps2, kds2 = self.robots.get_gains()
            me2 = self.robots.get_max_efforts() if hasattr(self.robots, "get_max_efforts") else None
            print("[drive] runtime(env0): "
                  f"L(kp={kps2[0,self.left_wheel_idx].item():.3f}, kd={kds2[0,self.left_wheel_idx].item():.3f}), "
                  f"R(kp={kps2[0,self.right_wheel_idx].item():.3f}, kd={kds2[0,self.right_wheel_idx].item():.3f})")
            if me2 is not None:
                print(f"[drive] max_effort(env0): L={me2[0,self.left_wheel_idx].item():.3f}, R={me2[0,self.right_wheel_idx].item():.3f}")

        def reset_idx(self, env_ids):
            env_ids = env_ids.to(device=self._device)
            env_pos = self._env_pos[env_ids]

            pos = env_pos.clone()
            pos[:, 2] = 0.03

            euler = torch.zeros((len(env_ids), 3), device=self._device)
            quat = euler_angles_to_quats(euler)

            indices = env_ids.to(torch.int32)
            self.robots.set_world_poses(pos, quat, indices=indices)
            self.robots.set_velocities(torch.zeros((len(env_ids), 6), device=self._device), indices=indices)
            self.robots.set_joint_velocities(torch.zeros((len(env_ids), self.robots.num_dof), device=self._device), indices=env_ids)

            self.current_v = 0.0
            self.sim_step_count = 0
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0

        def pre_physics_step(self, actions):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)

            self.sim_step_count += 1

            if self.sim_step_count <= self.drive_warmup_steps:
                v = 0.0
            else:
                self.current_v = min(self.target_v, self.current_v + self.v_ramp_per_step)
                v = self.current_v

            w = 0.0
            left_w = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            right_w = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS

            targets = torch.zeros((self._num_envs, self.robots.num_dof), device=self._device, dtype=torch.float32)
            targets[:, self.left_wheel_idx] = left_w
            targets[:, self.right_wheel_idx] = right_w

            self.robots.set_joint_velocity_targets(targets)

        def get_observations(self):
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self.name: {"obs": self.obs_buf}}

        def calculate_metrics(self):
            self.rew_buf[:] = 0.0

        def is_done(self):
            self.reset_buf[:] = 0

    print("[5] create task")
    task = TB3SingleEnvTask(name="TB3SingleEnvTask", sim_config=sim_config, env=vec_env)

    print("[6] vec_env.set_task(init_sim=True)")
    vec_env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=True
    )

    print("[7] run loop (Ctrl+C to quit)")
    try:
        step = 0
        while True:
            actions = torch.zeros((num_envs, 2), device=task._device)
            obs, rewards, dones, info = vec_env.step(actions)
            step += 1

            if step % 60 == 0:
                pos, _ = task.robots.get_world_poses()
                vel6 = task.robots.get_velocities()
                lv = vel6[0, 0:3].detach().cpu().numpy()
                print(f"[step {step}] pos=({pos[0,0].item():.3f},{pos[0,1].item():.3f},{pos[0,2].item():.3f}) "
                      f"lin_vel=({lv[0]:.3f},{lv[1]:.3f},{lv[2]:.3f}) "
                      f"cmd_v={task.current_v:.3f}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt -> exit")

    finally:
        # 方案C：硬退出，跳过 Kit 清理阶段（避免 destroy NoneType + segfault）
        try:
            vec_env.close()
        except Exception:
            pass
        os._exit(0)

if __name__ == "__main__":
    main()