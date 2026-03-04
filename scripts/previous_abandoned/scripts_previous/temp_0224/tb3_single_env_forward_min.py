# scripts/tb3_A_single_env_min.py
import math
import time
import torch

def main():
    from omegaconf import OmegaConf

    # 单 env + 保守物理参数（先稳）
    cfg = OmegaConf.create({
        "test": False,
        "task": {
            "name": "TB3_A_Min",
            "physics_engine": "physx",
            "env": {
                "numEnvs": 1,
                "envSpacing": 4.0,
                "maxEpisodeLength": 2000,
            },
            "sim": {
                "dt": 1.0 / 60.0,
                "substeps": 4,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
                    # "position_iterations": 16,
                    # "velocity_iterations": 4,
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
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    })
    OmegaConf.set_struct(cfg, False)

    # 关键：先用 VecEnvRLGames 初始化 Isaac Sim（你之前能跑就是靠这个顺序）:contentReference[oaicite:1]{index=1}
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage

    sim_config = SimConfig(cfg)

    class TB3MinTask(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config

            self._num_envs = 2
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2

            # 你只需要保证这个 USD 路径存在
            self.robot_usd_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

            # TB3 运动学参数
            self.WHEEL_RADIUS = 0.033
            self.WHEEL_BASE = 0.287

            # 直行控制：缓启动
            self.target_v = 0.10          # m/s
            self.current_v = 0.0
            self.v_ramp_per_step = 0.001  # 每 step 增加多少 m/s
            self.warmup_steps = 90        # 前 0.5s（dt=1/60）不驱动
            self.step_count = 0

            self.left_wheel_idx = None
            self.right_wheel_idx = None

            super().__init__(name, env, offset)

        def set_up_scene(self, scene):
            add_reference_to_stage(
                usd_path=self.robot_usd_path,
                prim_path=self.default_zero_env_path + "/TB3"
            )

            super().set_up_scene(scene)

            self.robots = ArticulationView(
                prim_paths_expr="/World/envs/.*/TB3",
                name="tb3_view",
                reset_xform_properties=False
            )
            scene.add(self.robots)

        def get_observations(self):
            self.obs_buf = torch.zeros((1, 1), device=self._device)
            return {self.name: {"obs": self.obs_buf}}

        def post_reset(self):
            dof_names = self.robots.dof_names
            print("\n========== DOF NAMES ==========")
            for i, n in enumerate(dof_names):
                print(f"{i:02d}: {n}")
            print("================================\n")

            # 找 wheel_left_joint / wheel_right_joint
            for i, n in enumerate(dof_names):
                s = n.lower()
                if self.left_wheel_idx is None and "wheel_left_joint" in s:
                    self.left_wheel_idx = i
                if self.right_wheel_idx is None and "wheel_right_joint" in s:
                    self.right_wheel_idx = i

            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                raise RuntimeError("没找到 wheel_left_joint / wheel_right_joint。请根据打印的 dof_names 修改匹配规则。")

            print(f"[OK] wheel idx: left={self.left_wheel_idx}, right={self.right_wheel_idx}")

            # 设置驱动增益/最大力矩（只改运行时参数，不做 USD 层操作）
            # 速度驱动：kp=0, kd 给足，max_effort 给足但别离谱
            try:
                kps, kds = self.robots.get_gains()
                kps[:, self.left_wheel_idx] = 0.0
                kps[:, self.right_wheel_idx] = 0.0
                kds[:, self.left_wheel_idx] = 80.0
                kds[:, self.right_wheel_idx] = 80.0
                self.robots.set_gains(kps=kps, kds=kds)
            except Exception as e:
                print(f"[WARN] set_gains failed: {e}")

            try:
                max_eff = self.robots.get_max_efforts()
                if max_eff is not None:
                    max_eff[:, self.left_wheel_idx] = 8.0
                    max_eff[:, self.right_wheel_idx] = 8.0
                    self.robots.set_max_efforts(max_eff)
            except Exception as e:
                print(f"[WARN] set_max_efforts failed: {e}")

            # 初始位姿：z 稍高一点，避免穿模冲量
            env_ids = torch.arange(1, dtype=torch.int64, device=self._device)
            self.reset_idx(env_ids)

        def reset_idx(self, env_ids):
            env_ids = env_ids.to(device=self._device)
            indices = env_ids.to(dtype=torch.int32)

            env_pos = self._env_pos[env_ids]
            pos = env_pos.clone()
            pos[:, 2] = 0.08

            # 朝向固定（yaw=0）
            rot = torch.zeros((len(env_ids), 4), device=self._device)
            rot[:, 3] = 1.0

            self.robots.set_world_poses(pos, rot, indices=indices)
            self.robots.set_velocities(torch.zeros((len(env_ids), 6), device=self._device), indices=indices)
            self.robots.set_joint_velocities(torch.zeros((len(env_ids), self.robots.num_dof), device=self._device), indices=env_ids)

            self.current_v = 0.0
            self.step_count = 0
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0

        def pre_physics_step(self, actions):
            self.step_count += 1

            if self.step_count <= self.warmup_steps:
                v = 0.0
            else:
                self.current_v = min(self.target_v, self.current_v + self.v_ramp_per_step)
                v = self.current_v

            w = 0.0
            left_w = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            right_w = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS

            targets = torch.zeros((1, self.robots.num_dof), dtype=torch.float32, device=self._device)
            targets[0, self.left_wheel_idx] = left_w
            targets[0, self.right_wheel_idx] = right_w

            # 关键：只用 velocity_targets（不允许 set_joint_velocities 作为控制）
            self.robots.set_joint_velocity_targets(targets)

        def calculate_metrics(self):
            self.rew_buf[:] = 0.0

        def is_done(self):
            self.reset_buf[:] = 0

    task = TB3MinTask(name="TB3_A_Min", sim_config=sim_config, env=vec_env)

    vec_env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=True
    )

    dt = cfg.task.sim.dt
    total_steps = int(10.0 / dt)

    print(f"[INFO] running 10s: dt={dt}, steps={total_steps}")
    step = 0
    try:
        while step < total_steps:
            actions = torch.zeros((1, 2), device=task._device)
            obs, rewards, dones, info = vec_env.step(actions)
            step += 1

            if step % int(0.1 / dt) == 0:
                pos, rot = task.robots.get_world_poses()
                z = pos[0, 2].item()
                print(f"t={step*dt:5.2f}s  pos=({pos[0,0].item():+.3f},{pos[0,1].item():+.3f},{z:+.3f})")

            time.sleep(0.001)

        print("[DONE] 10s finished.")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()