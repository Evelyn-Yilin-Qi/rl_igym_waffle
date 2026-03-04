"""
独立测试脚本：单个环境，单个TB3机器人，让机器人向前移动
目标：验证TB3机器人能否在Isaac Sim中正确移动
"""
import numpy as np
import torch

def main():
    print("📦 导入基础模块...")
    from omegaconf import OmegaConf
    
    # 机器人USD路径
    robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
    
    # 创建配置
    print("⚙️ 创建配置...")
    cfg = OmegaConf.create({
        "task": {
            "name": "SingleTB3",
            "physics_engine": "physx",
            "env": {
                "numEnvs": 1,  # 只有一个环境
                "envSpacing": 4.0,
                "maxEpisodeLength": 1000,
            },
            "sim": {
                "dt": 0.025,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
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
    
    # 关键：先导入VecEnvRLGames，实例化它（这会初始化Isaac Sim）
    print("🚀 导入VecEnvRLGames...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    
    print("🔥 创建向量化环境管理器（这会初始化 Isaac Sim）...")
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)
    
    # 现在Isaac Sim已初始化，可以安全导入其他模块
    print("📦 导入OmniIsaacGym模块（Isaac Sim已初始化）...")
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    
    # 创建SimConfig
    print("🔧 创建SimConfig...")
    sim_config = SimConfig(cfg)
    
    # 创建简单的任务类
    class SingleTB3Task(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config
            self._num_envs = self._task_cfg["env"]["numEnvs"]
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2
            
            # TB3物理参数（从URDF文件）
            self.WHEEL_RADIUS = 0.033  # 轮子半径（米）
            self.WHEEL_BASE = 0.288    # 轮距（米）
            
            super().__init__(name, env, offset)
        
        def set_up_scene(self, scene):
            # 加载机器人
            add_reference_to_stage(usd_path=robot_path, prim_path=self.default_zero_env_path + "/WafflePi")
            super().set_up_scene(scene)
            self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/WafflePi", name="tb3_view")
            scene.add(self.robots)
        
        def get_observations(self):
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self.name: {"obs": self.obs_buf}}
        
        def pre_physics_step(self, actions):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
            
            # 关键：在每个物理步之前重新应用驱动配置
            # 确保速度控制模式持续生效
            if not hasattr(self, '_drive_configured') or getattr(self, '_step_count', 0) % 50 == 0:
                self._configure_joint_drives()
                self._drive_configured = True
            
            # 设置向前移动的速度
            v = 0.1  # 线速度 0.1 m/s
            w = 0.0  # 角速度 0（直走）
            
            # 计算左右轮速度（差分驱动模型）
            left_wheel_vel = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            right_wheel_vel = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            
            # 设置关节速度
            joint_velocities = torch.zeros((self._num_envs, self.robots.num_dof), dtype=torch.float32, device=self._device)
            joint_velocities[:, 0] = left_wheel_vel  # 假设关节0是左轮
            joint_velocities[:, 1] = right_wheel_vel  # 假设关节1是右轮
            
            self.robots.set_joint_velocities(joint_velocities)
            
            # 更新步数计数
            self._step_count = getattr(self, '_step_count', 0) + 1
        
        def _configure_joint_drives(self):
            """配置关节驱动为速度控制模式"""
            try:
                from omni.isaac.core.utils.prims import get_prim_at_path
                from pxr import UsdPhysics
                
                # 获取机器人路径
                robot_paths = self.robots.prim_paths if hasattr(self.robots, 'prim_paths') else []
                if not robot_paths:
                    return
                
                # 为所有环境配置驱动
                for robot_path in robot_paths:
                    prim = get_prim_at_path(robot_path)
                    if not prim:
                        continue
                    
                    # 递归查找轮子关节
                    def find_joint(prim, target_name):
                        if target_name in prim.GetName():
                            return prim
                        for child in prim.GetChildren():
                            result = find_joint(child, target_name)
                            if result:
                                return result
                        return None
                    
                    # 配置左右轮关节
                    for joint_name in ["wheel_left_joint", "wheel_right_joint"]:
                        joint_prim = find_joint(prim, joint_name)
                        if joint_prim and joint_prim.IsValid():
                            drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                            if drive_api:
                                # 设置为速度控制模式
                                try:
                                    drive_api.GetStiffnessAttr().Set(0.0)
                                    drive_api.GetDampingAttr().Set(10.0)
                                    drive_api.GetMaxForceAttr().Set(1000.0)
                                except:
                                    # 如果Set失败，尝试Create
                                    try:
                                        drive_api.CreateStiffnessAttr(0.0)
                                        drive_api.CreateDampingAttr(10.0)
                                        drive_api.CreateMaxForceAttr(1000.0)
                                    except:
                                        pass
                            else:
                                # 如果没有驱动配置，创建一个
                                try:
                                    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                                    drive_api.CreateStiffnessAttr(0.0)
                                    drive_api.CreateDampingAttr(10.0)
                                    drive_api.CreateMaxForceAttr(1000.0)
                                except:
                                    pass
            except Exception as e:
                print(f"⚠️ 配置关节驱动时出错: {e}")
        
        def reset_idx(self, env_ids):
            num_resets = len(env_ids)
            env_pos = self._env_pos[env_ids]
            robot_positions = env_pos.clone()
            robot_positions[:, 2] = 0.2  # Z=0.2米，让它落下来
            robot_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(num_resets, 1)
            indices = env_ids.to(dtype=torch.int32)
            self.robots.set_world_poses(robot_positions, robot_rotations, indices=indices)
            self.robots.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
            self.robots.set_joint_velocities(torch.zeros((num_resets, self.robots.num_dof), device=self._device), indices=env_ids)
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
        
        def post_reset(self):
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)
            
            # 在post_reset后配置驱动
            print("🔧 配置关节驱动为速度控制模式...")
            self._configure_joint_drives()
            print("✅ 驱动配置完成")
        
        def calculate_metrics(self):
            self.rew_buf[:] = 0.0
        
        def is_done(self):
            self.reset_buf[:] = 0
    
    # 创建任务
    print("📦 创建测试任务...")
    task = SingleTB3Task(name="SingleTB3", sim_config=sim_config, env=vec_env)
    
    # 设置任务并初始化仿真
    print("🔧 初始化仿真环境...")
    vec_env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)
    
    print("✅ 已创建 1 个环境")
    print("🔄 开始物理模拟，让机器人落到地面...")
    
    # 运行物理模拟，让机器人落到地面
    for step in range(200):
        actions = torch.zeros((1, 2), device=task._device)
        task.pre_physics_step(actions)
        task.progress_buf += 1
        task.calculate_metrics()
        task.is_done()
        vec_env.render()
        
        if step % 20 == 0:
            positions, _ = task.robots.get_world_poses()
            print(f"Step {step}: 机器人Z位置 = {positions[0, 2].item():.4f} 米")
    
    # 获取初始位置
    initial_positions, _ = task.robots.get_world_poses()
    print("\n🚀 开始让机器人向前走（v=0.1 m/s）...")
    print(f"初始位置 X={initial_positions[0, 0].item():.4f}, Y={initial_positions[0, 1].item():.4f}, Z={initial_positions[0, 2].item():.4f}")
    
    # 主循环：让机器人持续向前走
    try:
        import time
        step = 200
        
        while True:
            actions = torch.zeros((1, 2), device=task._device)
            task.pre_physics_step(actions)
            task.progress_buf += 1
            task.calculate_metrics()
            task.is_done()
            vec_env.render()
            
            step += 1
            if step % 50 == 0:
                positions, _ = task.robots.get_world_poses()
                x_pos = positions[0, 0].item()
                y_pos = positions[0, 1].item()
                z_pos = positions[0, 2].item()
                x_initial = initial_positions[0, 0].item()
                y_initial = initial_positions[0, 1].item()
                
                x_displacement = x_pos - x_initial
                y_displacement = y_pos - y_initial
                distance = np.sqrt(x_displacement**2 + y_displacement**2)
                
                # 检查关节速度
                joint_vels = task.robots.get_joint_velocities()
                
                print(f"Step {step}: 位置 X={x_pos:.4f}, Y={y_pos:.4f}, Z={z_pos:.4f}")
                print(f"         移动距离: X方向={x_displacement:.4f}m, 总距离={distance:.4f}m")
                print(f"         关节速度={joint_vels[0, :2].cpu().numpy()}")
                
                if distance > 0.01:
                    print("✅ 机器人正在移动！")
                else:
                    print("⚠️ 机器人仍未移动")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
        final_positions, _ = task.robots.get_world_poses()
        x_final = final_positions[0, 0].item()
        y_final = final_positions[0, 1].item()
        x_initial = initial_positions[0, 0].item()
        y_initial = initial_positions[0, 1].item()
        distance = np.sqrt((x_final - x_initial)**2 + (y_final - y_initial)**2)
        print(f"\n✅ 最终移动距离: {distance:.4f} 米")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
