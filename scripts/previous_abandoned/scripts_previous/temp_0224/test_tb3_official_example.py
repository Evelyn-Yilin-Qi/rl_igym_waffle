"""
完全独立的TB3加载示例
基于OmniIsaacGymEnvs官方文档和示例模式
不参考任何工程代码，完全从零开始
"""
import numpy as np
import torch
import math

def main():
    # 步骤1: 导入基础模块（必须在main函数内导入omni模块）
    print("📦 导入基础模块...")
    from omegaconf import OmegaConf
    
    # 步骤2: 创建配置（参考官方文档的最小配置）
    print("⚙️ 创建配置...")
    num_envs = 4
    env_spacing = 4.0
    
    cfg = OmegaConf.create({
        "task": {
            "name": "TB3Task",
            "physics_engine": "physx",
            "env": {
                "numEnvs": num_envs,
                "envSpacing": env_spacing,
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
    
    # 步骤3: 导入VecEnvRLGames（这会初始化Isaac Sim）
    print("🚀 导入VecEnvRLGames...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    
    print("🔥 创建向量化环境管理器...")
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)
    
    # 步骤4: 现在可以安全导入其他omni模块
    print("📦 导入OmniIsaacGym模块...")
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats
    
    # 步骤5: 创建SimConfig
    print("🔧 创建SimConfig...")
    sim_config = SimConfig(cfg)
    
    # 步骤6: 定义任务类（完全基于官方模式）
    class TB3Task(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config
            
            self._num_envs = self._task_cfg["env"]["numEnvs"]
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2
            
            # TB3 USD路径
            self.robot_usd_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
            
            # TB3物理参数（从URDF和igym_waffle_env.py获取）
            self.WHEEL_RADIUS = 0.033  # 轮子半径（米）
            self.WHEEL_BASE = 0.287   # 轮距（米）
            
            # 轮子关节索引（将在post_reset中初始化）
            self.left_wheel_idx = None
            self.right_wheel_idx = None
            
            super().__init__(name, env, offset)
        
        def set_up_scene(self, scene):
            # 加载机器人USD资产（参考官方cartpole.py的模式）
            add_reference_to_stage(
                usd_path=self.robot_usd_path,
                prim_path=self.default_zero_env_path + "/TB3"
            )
            
            # 调用父类方法创建环境（这会克隆所有环境）
            super().set_up_scene(scene)
            
            # 关键修复：在USD层面为每个环境设置随机初始朝向
            # 这样旋转会在物理引擎初始化时就被读取，而不是在运行时被重置
            print("🔄 在USD层面为每个环境设置随机初始朝向...")
            from pxr import UsdGeom, Gf
            from omni.isaac.core.utils.stage import get_current_stage
            
            stage = get_current_stage()
            
            # 生成随机旋转角度
            import random
            random_angles = [random.uniform(0, 2 * math.pi) for _ in range(self._num_envs)]
            
            # 为每个环境的TB3设置不同的随机朝向
            for i in range(self._num_envs):
                env_path = f"/World/envs/env_{i}/TB3"
                prim = stage.GetPrimAtPath(env_path)
                
                if prim and prim.IsValid():
                    # 获取Xformable接口
                    xformable = UsdGeom.Xformable(prim)
                    
                    # 清除现有的旋转操作（如果有）
                    xformable.ClearXformOpOrder()
                    
                    # 添加Z轴旋转操作（对应UI中的Orient Z值）
                    angle_rad = random_angles[i]
                    angle_deg = math.degrees(angle_rad)
                    
                    # 使用AddRotateZOp添加Z轴旋转
                    rotate_z_op = xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat)
                    rotate_z_op.Set(angle_deg)  # 设置角度（度）
                    
                    # 确保Xform操作顺序正确
                    xformable.SetXformOpOrder([rotate_z_op])
                    
                    print(f"   环境 {i}: 设置USD Xform RotateZ = {angle_deg:.1f}°")
            
            # 创建ArticulationView（参考官方cartpole.py第71行）
            self.robots = ArticulationView(
                prim_paths_expr="/World/envs/.*/TB3",
                name="tb3_view",
                reset_xform_properties=False
            )
            scene.add(self.robots)
        
        def get_observations(self):
            # 简单的观察空间
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self.name: {"obs": self.obs_buf}}
        
        def pre_physics_step(self, actions):
            # 处理重置
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
            
            # 不应用任何动作，让机器人自然落下
        
        def reset_idx(self, env_ids):
            # 参考官方ingenuity.py第201-224行的模式
            num_resets = len(env_ids)
            env_ids = env_ids.to(device=self._device)
            
            # 获取环境位置
            env_pos = self._env_pos[env_ids]
            robot_positions = env_pos.clone()
            robot_positions[:, 2] = 0.2  # Z=0.2米高度
            
            # 关键：使用在post_reset中保存的随机旋转
            # 参考官方ingenuity.py第219行的模式
            robot_rotations = self.initial_rotations[env_ids].clone()
            
            # 设置位置和旋转（参考官方ingenuity.py第219行）
            indices = env_ids.to(dtype=torch.int32)
            self.robots.set_world_poses(robot_positions, robot_rotations, indices=indices)
            
            # 重置速度
            self.robots.set_velocities(
                torch.zeros((num_resets, 6), device=self._device),
                indices=indices
            )
            self.robots.set_joint_velocities(
                torch.zeros((num_resets, self.robots.num_dof), device=self._device),
                indices=env_ids
            )
            
            # 重置缓冲区
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
        
        def post_reset(self):
            # 参考官方ingenuity.py第177-184行的模式
            # 先获取当前的world poses（此时机器人已经加载，但可能还是默认旋转）
            self.root_pos, self.root_rot = self.robots.get_world_poses()
            
            # 生成随机旋转（绕Z轴，水平面旋转）
            print("🔄 为每个环境生成随机初始朝向...")
            random_angles = torch.rand(self._num_envs, device=self._device) * 2 * math.pi
            
            # 转换为四元数（参考官方ingenuity.py使用euler_angles_to_quats）
            euler_angles = torch.zeros((self._num_envs, 3), device=self._device)
            euler_angles[:, 2] = random_angles  # yaw (绕Z轴)
            self.initial_rotations = euler_angles_to_quats(euler_angles)
            
            # 确保在正确的设备上
            if self.initial_rotations.device != self._device:
                self.initial_rotations = self.initial_rotations.to(device=self._device)
            
            # 打印初始朝向
            print("📐 各环境初始朝向（角度）：")
            for i in range(self._num_envs):
                angle_deg = random_angles[i].item() * 180.0 / math.pi
                print(f"   环境 {i}: {angle_deg:.1f}°")
            
            # 调用reset_idx来应用初始状态（参考官方cartpole.py第143-144行）
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)
        
        def calculate_metrics(self):
            self.rew_buf[:] = 0.0
        
        def is_done(self):
            self.reset_buf[:] = 0
    
    # 步骤7: 创建任务实例
    print("📦 创建任务...")
    task = TB3Task(name="TB3Task", sim_config=sim_config, env=vec_env)
    
    # 步骤8: 设置任务并初始化仿真（参考官方文档）
    print("🔧 初始化仿真环境...")
    vec_env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=True
    )
    
    print(f"✅ 已创建 {num_envs} 个环境")
    print("🔄 开始物理模拟...")
    
    # 步骤9: 运行主循环
    # 关键：根据官方文档，PhysX需要在设置状态后步进才能更新
    # 所以在每个物理步进后，重新应用随机旋转
    for step in range(200):
        actions = torch.zeros((num_envs, 2), device=task._device)
        task.pre_physics_step(actions)
        task.progress_buf += 1
        task.calculate_metrics()
        task.is_done()
        vec_env.render()
        
        # 关键修复：在每个物理步进后，重新应用随机旋转
        # 这是因为PhysX的限制：设置根状态后需要步进才能更新链接状态
        # 参考官方文档关于PhysX限制的说明
        if hasattr(task, 'initial_rotations'):
            positions, _ = task.robots.get_world_poses()
            all_indices = torch.arange(num_envs, dtype=torch.int32, device=task._device)
            task.robots.set_world_poses(positions, task.initial_rotations, indices=all_indices)
        
        # 每20步打印一次状态
        if step % 20 == 0:
            positions, rotations = task.robots.get_world_poses()
            print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
            
            # 验证旋转是否正确保持
            for i in range(min(2, num_envs)):
                actual_q = rotations[i].cpu().numpy()
                expected_q = task.initial_rotations[i].cpu().numpy()
                actual_angle = 2 * math.atan2(actual_q[3], actual_q[0]) * 180.0 / math.pi
                expected_angle = 2 * math.atan2(expected_q[3], expected_q[0]) * 180.0 / math.pi
                print(f"   环境 {i}: 期望角度={expected_angle:.1f}°, 实际角度={actual_angle:.1f}°")
    
    print("\n✅ 测试完成！")
    print("💡 提示：按Ctrl+C退出...")
    
    # 保持运行
    try:
        import time
        step = 200
        while True:
            actions = torch.zeros((num_envs, 2), device=task._device)
            task.pre_physics_step(actions)
            task.progress_buf += 1
            task.calculate_metrics()
            task.is_done()
            vec_env.render()
            
            # 关键修复：在无限循环中也要保持旋转
            if hasattr(task, 'initial_rotations'):
                positions, _ = task.robots.get_world_poses()
                all_indices = torch.arange(num_envs, dtype=torch.int32, device=task._device)
                task.robots.set_world_poses(positions, task.initial_rotations, indices=all_indices)
            
            step += 1
            if step % 200 == 0:
                positions, rotations = task.robots.get_world_poses()
                print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
