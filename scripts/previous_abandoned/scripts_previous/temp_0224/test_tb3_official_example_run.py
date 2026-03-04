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
                "dt": 0.01,  # 关键：降低时间步长，提高稳定性
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
            
            # 关键：让所有小车向前走
            # 使用差速驱动模型：v = 0.1 m/s（向前），w = 0（不转弯）
            if self.left_wheel_idx is not None and self.right_wheel_idx is not None:
                v = 0.1  # 线速度：0.1 m/s 向前
                w = 0.0  # 角速度：0 rad/s（不转弯）
                
                # 差速驱动运动学模型
                # left_wheel_vel = (v - w * wheel_base/2) / wheel_radius
                # right_wheel_vel = (v + w * wheel_base/2) / wheel_radius
                left_wheel_vel = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
                right_wheel_vel = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
                
                # 创建关节速度目标张量
                # 关键：使用velocity_targets而不是set_joint_velocities
                # set_joint_velocities是"直接覆盖状态"，set_joint_velocity_targets是"给驱动器设置目标"
                joint_velocity_targets = torch.zeros((self._num_envs, self.robots.num_dof), dtype=torch.float32, device=self._device)
                joint_velocity_targets[:, self.left_wheel_idx] = left_wheel_vel
                joint_velocity_targets[:, self.right_wheel_idx] = right_wheel_vel
                
                # 关键修复：使用set_joint_velocity_targets()而不是set_joint_velocities()
                # 这会给驱动器设置速度目标，而不是强行覆盖状态
                try:
                    self.robots.set_joint_velocity_targets(joint_velocity_targets)
                except AttributeError:
                    # 如果API不存在，回退到set_joint_velocities（但这不是理想方式）
                    print("   ⚠️ set_joint_velocity_targets不存在，使用set_joint_velocities")
                    self.robots.set_joint_velocities(joint_velocity_targets)
        
        def reset_idx(self, env_ids):
            # 参考官方ingenuity.py第201-224行的模式
            num_resets = len(env_ids)
            env_ids = env_ids.to(device=self._device)
            
            # 获取环境位置
            env_pos = self._env_pos[env_ids]
            robot_positions = env_pos.clone()
            robot_positions[:, 2] = 0.05  # 关键：降低起始高度，0.2米对TB3来说太高了
            
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
            # 关键：获取关节信息，找到左右轮关节的索引
            # ArticulationView需要在post_reset时才能完全初始化
            print("🔍 获取关节信息...")
            try:
                dof_names = self.robots.dof_names
                num_dof = self.robots.num_dof
                print(f"   num_dof = {num_dof}")
                print(f"   dof_names = {dof_names}")
                
                # 查找左右轮关节的索引
                # 关键：使用更精确的匹配，只匹配明确的轮子关节名称
                # 根据截图和URDF，关节名称可能是：
                # - a_namespace_wheel_left_joint / a_namespace_wheel_right_joint
                # - wheel_left_joint / wheel_right_joint
                for i, name in enumerate(dof_names):
                    name_lower = name.lower()
                    # 精确匹配：必须包含wheel_left_joint或wheel_right_joint
                    if 'wheel_left_joint' in name_lower:
                        if self.left_wheel_idx is None:  # 只取第一个匹配的
                            self.left_wheel_idx = i
                            print(f"   ✅ 找到左轮关节: 索引={i}, 名称={name}")
                    elif 'wheel_right_joint' in name_lower:
                        if self.right_wheel_idx is None:  # 只取第一个匹配的
                            self.right_wheel_idx = i
                            print(f"   ✅ 找到右轮关节: 索引={i}, 名称={name}")
                
                # 如果没找到，尝试默认索引（通常左轮是0，右轮是1）
                if self.left_wheel_idx is None or self.right_wheel_idx is None:
                    print(f"   ⚠️ 未找到轮子关节名称，使用默认索引: left=0, right=1")
                    if num_dof >= 2:
                        self.left_wheel_idx = 0
                        self.right_wheel_idx = 1
                    else:
                        print(f"   ❌ 错误：关节数量不足 ({num_dof})，无法设置轮子速度")
            except Exception as e:
                print(f"   ⚠️ 获取关节信息时出错: {e}")
                # 使用默认索引
                if self.robots.num_dof >= 2:
                    self.left_wheel_idx = 0
                    self.right_wheel_idx = 1
                    print(f"   ✅ 使用默认索引: left=0, right=1")
            
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
            
            # 关键：在post_reset中配置轮子关节的驱动参数
            # 确保是velocity drive模式（stiffness=0, damping合理, max_force合理）
            print("🔧 配置轮子关节驱动参数...")
            self._configure_wheel_drives()
            
            # 调用reset_idx来应用初始状态（参考官方cartpole.py第143-144行）
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)
        
        def _configure_wheel_drives(self):
            """配置轮子关节的驱动参数，确保是velocity drive模式"""
            from pxr import UsdPhysics
            from omni.isaac.core.utils.stage import get_current_stage
            
            stage = get_current_stage()
            
            # 根据dof_names找到的关节索引，获取对应的关节prim路径
            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                print("   ⚠️ 轮子关节索引未找到，跳过驱动配置")
                return
            
            # 获取关节名称
            dof_names = self.robots.dof_names
            left_joint_name = dof_names[self.left_wheel_idx]
            right_joint_name = dof_names[self.right_wheel_idx]
            
            print(f"   🔍 查找关节prim: left={left_joint_name}, right={right_joint_name}")
            
            configured_count = 0
            for env_id in range(self._num_envs):
                env_path = f"/World/envs/env_{env_id}/TB3"
                
                # 尝试查找轮子关节prim（根据实际USD结构）
                # 可能的路径格式（根据之前的截图和URDF结构）
                possible_base_paths = [
                    f"{env_path}/a__namespace_base_link",
                    f"{env_path}/base_link",
                    f"{env_path}",
                ]
                
                for base_path in possible_base_paths:
                    # 尝试左轮关节
                    left_joint_path = f"{base_path}/{left_joint_name}"
                    prim = stage.GetPrimAtPath(left_joint_path)
                    if prim and prim.IsValid():
                        try:
                            # 获取或创建DriveAPI
                            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                            if not drive_api:
                                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                            
                            # 设置velocity drive参数
                            # stiffness=0: velocity drive模式
                            # damping=10: 合理的阻尼
                            # max_force=30: 保守的扭矩限制（TB3质量轻，不需要太大）
                            drive_api.GetStiffnessAttr().Set(0.0)
                            drive_api.GetDampingAttr().Set(10.0)
                            drive_api.GetMaxForceAttr().Set(30.0)
                            configured_count += 1
                            if env_id == 0:  # 只打印第一个环境的详细信息
                                print(f"   ✅ 环境 {env_id}: 已配置左轮驱动参数 (stiffness=0, damping=10, max_force=30)")
                        except Exception as e:
                            if env_id == 0:
                                print(f"   ⚠️ 配置左轮驱动时出错: {e}")
                    
                    # 尝试右轮关节
                    right_joint_path = f"{base_path}/{right_joint_name}"
                    prim = stage.GetPrimAtPath(right_joint_path)
                    if prim and prim.IsValid():
                        try:
                            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                            if not drive_api:
                                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                            
                            drive_api.GetStiffnessAttr().Set(0.0)
                            drive_api.GetDampingAttr().Set(10.0)
                            drive_api.GetMaxForceAttr().Set(30.0)
                            configured_count += 1
                            if env_id == 0:
                                print(f"   ✅ 环境 {env_id}: 已配置右轮驱动参数 (stiffness=0, damping=10, max_force=30)")
                        except Exception as e:
                            if env_id == 0:
                                print(f"   ⚠️ 配置右轮驱动时出错: {e}")
            
            if configured_count > 0:
                print(f"   ✅ 总共配置了 {configured_count} 个关节驱动（{self._num_envs}个环境 × 2个轮子）")
            else:
                print(f"   ⚠️ 未能配置任何关节驱动，可能路径不匹配")
        
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
    # 关键修复：使用vec_env.step(actions)来推进物理仿真
    # 这确保了PhysX真正执行物理步进，而不是只渲染
    print("💡 使用vec_env.step()来推进物理仿真...")
    
    try:
        import time
        step = 0
        while True:
            # 关键：使用vec_env.step(actions)来推进物理仿真
            # 这会内部调用：pre_physics_step → sim step → post_physics_step/obs/reward/done等
            actions = torch.zeros((num_envs, 2), device=task._device)
            obs, rewards, dones, info = vec_env.step(actions)
            
            step += 1
            
            # 每20步打印一次状态
            if step % 20 == 0:
                positions, rotations = task.robots.get_world_poses()
                print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
                
                # 打印机器人XY位置，看是否在移动
                print(f"   机器人XY位置 (env 0) = [{positions[0, 0].item():.3f}, {positions[0, 1].item():.3f}]")
                
                # 如果有轮子关节索引，打印关节速度
                if task.left_wheel_idx is not None and task.right_wheel_idx is not None:
                    try:
                        joint_velocities = task.robots.get_joint_velocities()
                        print(f"   关节速度 (env 0): left={joint_velocities[0, task.left_wheel_idx].item():.4f}, right={joint_velocities[0, task.right_wheel_idx].item():.4f}")
                    except:
                        pass
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
