"""
最简单的测试脚本：4个环境，机器人能显示并落到地面，然后向前走
参考 train_stage1.py 的加载方式，使用 VecEnvRLGames 来管理应用生命周期
基于 test_ground_set_template_success_copy_1.py，只添加速度控制
"""
# ⛔️ 绝对不能在文件顶部 import 任何自定义的 envs 或 omni 模块！
import numpy as np
import torch

def main():
    # ⚠️ 关键：在main函数内部导入omni相关模块
    # 参考 train_stage1.py 和 skrl_utils.py 的模式
    # 正确的顺序：先导入VecEnvRLGames类，实例化它（初始化Isaac Sim），然后才能导入SimConfig
    print("📦 导入基础模块...")
    from omegaconf import OmegaConf
    
    # 机器人USD路径 - 使用新生成的带速度控制配置的USD文件
    robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi_velocity.usd"
    
    # 创建4个环境，每个环境间距4米
    num_envs = 4
    env_spacing = 4.0
    
    # 创建最小配置（参考 train_stage1.py 和 skrl_utils.py 的方式）
    print("⚙️ 创建配置...")
    cfg = OmegaConf.create({
        "task": {
            "name": "SimpleWaffle",
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
        "test": False,  # SimConfig 需要这个键
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    })
    
    # 确保配置结构正确（参考 skrl_utils.py）
    # 必须在创建 SimConfig 之前设置，并且确保所有必需的键都存在
    OmegaConf.set_struct(cfg, False)
    
    # 确保所有必需的顶层键都存在（参考 skrl_utils.py 的 default_oige_args）
    # 这些键是 SimConfig 需要的
    if "test" not in cfg:
        cfg["test"] = False
    if "seed" not in cfg:
        cfg["seed"] = 42
    if "enable_livestream" not in cfg:
        cfg["enable_livestream"] = False
    if "enable_cameras" not in cfg:
        cfg["enable_cameras"] = False
    
    # 关键：先导入VecEnvRLGames类（不导入SimConfig）
    print("🚀 导入VecEnvRLGames...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    
    # 实例化VecEnvRLGames（这会初始化 Isaac Sim）
    print("🔥 创建向量化环境管理器（这会初始化 Isaac Sim）...")
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)
    
    # 现在 Isaac Sim 已经初始化，可以安全导入 SimConfig 和其他模块
    print("📦 导入OmniIsaacGym模块（Isaac Sim已初始化）...")
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    
    # 在创建 SimConfig 之前，再次确保所有必需的键都存在
    # 参考 skrl_utils.py 的做法
    OmegaConf.set_struct(cfg, False)
    
    # 确保所有必需的顶层键都存在（SimConfig 需要这些键）
    required_top_level_keys = {
        "device_id": 0,
        "sim_device": "gpu",
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": False,
        "test": False,
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    }
    for k, v in required_top_level_keys.items():
        if k not in cfg:
            cfg[k] = v
    
    # 创建SimConfig
    print("🔧 创建SimConfig...")
    sim_config = SimConfig(cfg)
    
    # 创建一个简单的任务类来加载机器人
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    
    class SimpleWaffleTask(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config
            self._num_envs = self._task_cfg["env"]["numEnvs"]
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2
            
            # TB3 物理参数（参考 URDF 文件）
            # 从 turtlebot3_waffle_pi.urdf 中：
            # - 左轮位置：y=0.144，右轮位置：y=-0.144
            # - 轮距 = 0.144 - (-0.144) = 0.288 米
            # - 轮子半径 = 0.033 米（在 collision 中定义）
            self.WHEEL_RADIUS = 0.033  # 轮子半径（米）
            self.WHEEL_BASE = 0.288    # 轮距（米）- 修正为 URDF 中的实际值
            
            super().__init__(name, env, offset)
        
        def set_up_scene(self, scene):
            # 在每个环境中加载机器人
            # 关键：使用与实际工作代码完全相同的路径和方式
            add_reference_to_stage(usd_path=robot_path, prim_path=self.default_zero_env_path + "/WafflePi")
            
            # ⚠️ 关键修复：参考 OIGE 官方示例（cartpole.py 第78行，ingenuity.py 第98行）
            # 必须调用 apply_articulation_settings 来应用关节驱动配置
            # 这确保 USD 文件中的驱动配置（stiffness=0.0, damping=10.0）被正确应用
            from omni.isaac.core.utils.prims import get_prim_at_path
            robot_prim = get_prim_at_path(self.default_zero_env_path + "/WafflePi")
            if robot_prim:
                # 创建一个最小配置（参考 cartpole.py 的实现方式）
                # 注意：如果配置文件中没有 "WafflePi" 的配置，使用默认值
                try:
                    actor_config = self._sim_config.parse_actor_config("WafflePi")
                except:
                    # 如果配置不存在，创建一个最小配置
                    actor_config = {
                        "override_usd_defaults": False,
                        "enable_self_collisions": False,
                        "enable_gyroscopic_forces": True,
                        "solver_position_iteration_count": 4,
                        "solver_velocity_iteration_count": 0,
                        "sleep_threshold": 0.005,
                        "stabilization_threshold": 0.001,
                    }
                self._sim_config.apply_articulation_settings("WafflePi", robot_prim, actor_config)
            
            super().set_up_scene(scene)
            # 关键：使用与实际工作代码完全相同的表达式
            self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/WafflePi", name="waffle_view")
            scene.add(self.robots)
            
            # 注意：ArticulationView 在 set_up_scene 时可能还未初始化
            # 关节信息将在 post_reset 中获取
            print(f"🔍 机器人关节信息（将在 post_reset 中获取）...")
            # 默认使用索引 0 和 1（将在 post_reset 中更新）
            self.left_wheel_idx = 0
            self.right_wheel_idx = 1
        
        def get_observations(self):
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self.name: {"obs": self.obs_buf}}
        
        def pre_physics_step(self, actions):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
            
            # 添加速度控制：让所有小车向前走
            # 参考 igym_waffle_env.py 的实现
            # 设置线速度 v = 0.1 m/s，角速度 w = 0（直走）
            v = 0.1  # 向前线速度 0.1 m/s
            w = 0.0  # 角速度 0（直走）
            
            # 计算左右轮速度（差分驱动模型）
            # left_wheel_vel = (v - w * wheel_base/2) / wheel_radius
            # right_wheel_vel = (v + w * wheel_base/2) / wheel_radius
            left_wheel_vel = (v - w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            right_wheel_vel = (v + w * (self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
            
            # 设置关节速度（使用检测到的关节索引）
            joint_velocities = torch.zeros((self._num_envs, self.robots.num_dof), dtype=torch.float32, device=self._device)
            # 使用保存的关节索引（如果已检测到），否则使用默认值
            left_idx = getattr(self, 'left_wheel_idx', 0)
            right_idx = getattr(self, 'right_wheel_idx', 1)
            joint_velocities[:, left_idx] = left_wheel_vel
            joint_velocities[:, right_idx] = right_wheel_vel
            
            # 调试：打印设置的速度值（仅前几次）
            if not hasattr(self, '_vel_set_count'):
                self._vel_set_count = 0
            if self._vel_set_count < 3:
                print(f"🔍 设置关节速度 (count {self._vel_set_count}): left={left_wheel_vel:.4f}, right={right_wheel_vel:.4f}")
                print(f"   joint_velocities[0, :] = {joint_velocities[0, :].cpu().numpy()}")
                self._vel_set_count += 1
            
            # 关键尝试：在每个物理步之前重新应用驱动配置
            # 因为驱动配置可能在物理步进后被重置
            step_count = getattr(self, '_step_count', 0)
            self._step_count = step_count + 1
            if not hasattr(self, '_drive_config_applied') or step_count % 100 == 0:
                try:
                    from omni.isaac.core.utils.prims import get_prim_at_path
                    from pxr import UsdPhysics
                    
                    # 为第一个环境重新应用驱动配置（作为示例）
                    robot_paths = self.robots.prim_paths if hasattr(self.robots, 'prim_paths') else []
                    if robot_paths:
                        first_robot_path = robot_paths[0]
                        prim = get_prim_at_path(first_robot_path)
                        if prim:
                            # 查找轮子关节并重新应用驱动配置
                            def find_joint_recursive(prim, target_name):
                                if target_name in prim.GetName():
                                    return prim
                                for child in prim.GetChildren():
                                    result = find_joint_recursive(child, target_name)
                                    if result:
                                        return result
                                return None
                            
                            for joint_name in ["wheel_left_joint", "wheel_right_joint"]:
                                joint_prim = find_joint_recursive(prim, joint_name)
                                if joint_prim and joint_prim.IsValid():
                                    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                                    if drive_api:
                                        # 重新设置驱动参数为速度控制模式
                                        try:
                                            drive_api.GetStiffnessAttr().Set(0.0)
                                            drive_api.GetDampingAttr().Set(10.0)
                                            drive_api.GetMaxForceAttr().Set(1000.0)
                                        except:
                                            drive_api.CreateStiffnessAttr(0.0)
                                            drive_api.CreateDampingAttr(10.0)
                                            drive_api.CreateMaxForceAttr(1000.0)
                    
                    self._drive_config_applied = True
                except Exception as e:
                    # 静默失败，不影响主流程
                    pass
            
            # 设置关节速度
            self.robots.set_joint_velocities(joint_velocities)
            
            # 调试：立即检查是否被设置（仅前几次）
            # if self._vel_set_count <= 3:
            #     try:
            #         check_vels = self.robots.get_joint_velocities()
            #         print(f"   设置后立即读取: {check_vels[0, :2].cpu().numpy()}")
            #     except:
            #         pass
        
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
            
            # 调试：在post_reset后检查关节信息
            if not hasattr(self, '_post_reset_debug_printed'):
                print(f"🔍 post_reset 后关节信息：")
                print(f"   num_dof = {self.robots.num_dof}")
                # 尝试获取关节位置和速度
                try:
                    joint_positions = self.robots.get_joint_positions()
                    joint_velocities = self.robots.get_joint_velocities()
                    print(f"   关节位置 shape = {joint_positions.shape}")
                    print(f"   关节速度 shape = {joint_velocities.shape}")
                    print(f"   关节位置 (env 0) = {joint_positions[0, :].cpu().numpy()}")
                    print(f"   关节速度 (env 0) = {joint_velocities[0, :].cpu().numpy()}")
                except Exception as e:
                    print(f"   ⚠️ 无法获取关节信息: {e}")
                
                # 关键尝试：检查并配置关节驱动模式
                # 尝试通过 USD API 直接访问关节并配置驱动
                # 重要：需要为所有环境配置驱动，而不仅仅是第一个
                try:
                    from omni.isaac.core.utils.prims import get_prim_at_path
                    from pxr import UsdPhysics
                    
                    # 获取所有环境的机器人路径
                    robot_paths = self.robots.prim_paths if hasattr(self.robots, 'prim_paths') else []
                    if not robot_paths and hasattr(self.robots, '_prim_paths'):
                        robot_paths = self.robots._prim_paths
                    
                    # 至少处理第一个环境（用于调试输出）
                    if robot_paths:
                        first_robot_path = robot_paths[0]
                        print(f"   🔍 尝试检查关节驱动配置...")
                        print(f"   机器人路径: {first_robot_path}")
                        
                        # 尝试获取关节 prim
                        try:
                            prim = get_prim_at_path(first_robot_path)
                            if prim:
                                print(f"   ✅ 成功获取机器人 prim")
                                # 尝试查找关节
                                from pxr import Usd
                                stage = prim.GetStage()
                                # 查找轮子关节（关键：查找 joint，不是 link）
                                # 从 URDF 看，关节名称是 wheel_left_joint 和 wheel_right_joint
                                wheel_joint_names = ["wheel_left_joint", "wheel_right_joint"]
                                
                                for joint_name in wheel_joint_names:
                                    # 尝试多种可能的路径格式
                                    possible_paths = [
                                        f"{first_robot_path}/{joint_name}",
                                        f"{first_robot_path}/base_link/{joint_name}",
                                        f"{first_robot_path}/*/{joint_name}",
                                    ]
                                    
                                    found_joint = False
                                    for joint_path_str in possible_paths:
                                        try:
                                            # 使用路径表达式查找
                                            from pxr import Usd
                                            stage = prim.GetStage()
                                            
                                            # 尝试直接路径
                                            joint_prim = stage.GetPrimAtPath(joint_path_str)
                                            if not joint_prim or not joint_prim.IsValid():
                                                # 尝试递归查找
                                                def find_joint_recursive(prim, target_name):
                                                    if prim.GetName() == target_name or target_name in prim.GetName():
                                                        return prim
                                                    for child in prim.GetChildren():
                                                        result = find_joint_recursive(child, target_name)
                                                        if result:
                                                            return result
                                                    return None
                                                
                                                joint_prim = find_joint_recursive(prim, joint_name)
                                            
                                            if joint_prim and joint_prim.IsValid():
                                                print(f"   ✅ 找到关节: {joint_prim.GetPath()}")
                                                found_joint = True
                                                
                                                # 检查是否有驱动
                                                drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                                                if drive_api:
                                                    print(f"      ✅ 找到驱动配置")
                                                    # 打印驱动参数
                                                    try:
                                                        max_force = drive_api.GetMaxForceAttr().Get()
                                                        damping = drive_api.GetDampingAttr().Get()
                                                        stiffness = drive_api.GetStiffnessAttr().Get()
                                                        print(f"         当前参数: max_force={max_force}, damping={damping}, stiffness={stiffness}")
                                                        
                                                        # 关键修复：如果 stiffness 太大，说明是位置控制模式
                                                        # 需要修改为速度控制模式（stiffness=0）
                                                        if stiffness > 1000.0:
                                                            print(f"         ⚠️ 检测到位置控制模式（stiffness过大），修改为速度控制模式...")
                                                            try:
                                                                # 修改驱动参数为速度控制模式
                                                                drive_api.GetStiffnessAttr().Set(0.0)  # 速度控制：stiffness=0
                                                                drive_api.GetDampingAttr().Set(10.0)   # 合理的阻尼值
                                                                drive_api.GetMaxForceAttr().Set(1000.0)  # 合理的最大力
                                                                # 注意：MaxVelocity 可能不存在，跳过
                                                                print(f"         ✅ 已修改为速度控制模式: stiffness=0.0, damping=10.0, max_force=1000.0")
                                                            except Exception as e:
                                                                print(f"         ⚠️ 修改驱动参数时出错: {e}")
                                                                # 尝试使用 Create 方法
                                                                try:
                                                                    drive_api.CreateStiffnessAttr(0.0)
                                                                    drive_api.CreateDampingAttr(10.0)
                                                                    drive_api.CreateMaxForceAttr(1000.0)
                                                                    print(f"         ✅ 使用 Create 方法成功修改驱动参数")
                                                                except Exception as e2:
                                                                    print(f"         ❌ Create 方法也失败: {e2}")
                                                            
                                                            # 关键：为所有环境修改驱动配置
                                                            if len(robot_paths) > 1:
                                                                print(f"         🔄 为所有 {len(robot_paths)} 个环境修改驱动配置...")
                                                                for env_idx, robot_path in enumerate(robot_paths[1:], 1):
                                                                    try:
                                                                        env_prim = get_prim_at_path(robot_path)
                                                                        if env_prim:
                                                                            # 递归查找同名关节
                                                                            def find_joint_recursive(prim, target_name):
                                                                                if target_name in prim.GetName():
                                                                                    return prim
                                                                                for child in prim.GetChildren():
                                                                                    result = find_joint_recursive(child, target_name)
                                                                                    if result:
                                                                                        return result
                                                                                return None
                                                                            
                                                                            env_joint_prim = find_joint_recursive(env_prim, joint_name)
                                                                            if env_joint_prim and env_joint_prim.IsValid():
                                                                                env_drive_api = UsdPhysics.DriveAPI.Get(env_joint_prim, "angular")
                                                                                if env_drive_api:
                                                                                    try:
                                                                                        env_drive_api.GetStiffnessAttr().Set(0.0)
                                                                                        env_drive_api.GetDampingAttr().Set(10.0)
                                                                                        env_drive_api.GetMaxForceAttr().Set(1000.0)
                                                                                    except:
                                                                                        # 如果 Set 失败，尝试 Create
                                                                                        try:
                                                                                            env_drive_api.CreateStiffnessAttr(0.0)
                                                                                            env_drive_api.CreateDampingAttr(10.0)
                                                                                            env_drive_api.CreateMaxForceAttr(1000.0)
                                                                                        except:
                                                                                            pass
                                                                    except Exception as e:
                                                                        print(f"         ⚠️ 环境 {env_idx} 修改失败: {e}")
                                                                print(f"         ✅ 所有环境的驱动配置已修改")
                                                        else:
                                                            print(f"         ✅ 驱动配置已正确（速度控制模式）")
                                                    except Exception as e:
                                                        print(f"         ⚠️ 读取/修改驱动参数时出错: {e}")
                                                else:
                                                    print(f"      ⚠️ 未找到驱动配置，尝试创建...")
                                                    # 尝试创建驱动配置
                                                    try:
                                                        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                                                        # 设置驱动参数（速度控制模式）
                                                        drive_api.CreateMaxForceAttr(1000.0)  # 最大力
                                                        drive_api.CreateDampingAttr(10.0)    # 阻尼
                                                        drive_api.CreateStiffnessAttr(0.0)   # 刚度（速度控制设为0）
                                                        # 注意：MaxVelocity 可能不存在，跳过
                                                        print(f"      ✅ 已创建速度驱动配置")
                                                    except Exception as e:
                                                        print(f"      ❌ 创建驱动配置失败: {e}")
                                                break
                                        except Exception as e:
                                            continue
                                    
                                    if not found_joint:
                                        print(f"   ⚠️ 未找到关节: {joint_name}")
                        except Exception as e:
                            print(f"   ⚠️ 检查关节驱动时出错: {e}")
                except Exception as e:
                    print(f"   ⚠️ 无法检查关节驱动配置: {e}")
                
                self._post_reset_debug_printed = True
        
        def calculate_metrics(self):
            self.rew_buf[:] = 0.0
        
        def is_done(self):
            self.reset_buf[:] = 0
    
    # 创建任务
    print("📦 创建测试任务...")
    task = SimpleWaffleTask(name="SimpleWaffle", sim_config=sim_config, env=vec_env)
    
    # 设置任务并初始化仿真（参考 train_stage1.py）
    print("🔧 初始化仿真环境...")
    vec_env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)
    
    print(f"✅ 已创建 {num_envs} 个环境")
    print("🔄 开始物理模拟，让机器人落到地面...")
    
    # 运行物理模拟，让机器人落到地面
    for step in range(200):
        # 创建零动作
        actions = torch.zeros((num_envs, 2), device=task._device)
        
        # 执行一步（VecEnvRLGames内部会处理应用更新）
        task.pre_physics_step(actions)
        
        # 更新进度缓冲区
        task.progress_buf += 1
        
        # 计算奖励和完成状态
        task.calculate_metrics()
        task.is_done()
        
        # 渲染（VecEnvRLGames内部处理）
        vec_env.render()
        
        if step % 20 == 0:
            # 获取机器人位置
            positions, _ = task.robots.get_world_poses()
            print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
    
    # 获取初始位置（用于对比移动距离）
    initial_positions, _ = task.robots.get_world_poses()
    print("\n🚀 开始让机器人向前走（v=0.1 m/s）...")
    print(f"初始位置 X: {initial_positions[:, 0].cpu().numpy()}")
    
    # 获取最终位置
    final_positions, _ = task.robots.get_world_poses()
    print("\n✅ 测试完成！最终机器人Z位置：")
    for i in range(num_envs):
        print(f"   环境 {i}: Z = {final_positions[i, 2]:.4f} 米")
    
    print("\n💡 提示：VecEnvRLGames会自动保持应用运行")
    print("按Ctrl+C退出...")
    
    # 保持运行 - VecEnvRLGames内部会处理应用生命周期
    # 我们只需要定期调用相关方法
    try:
        import time
        step = 200
        
        # 持续设置的速度值
        v = 0.1  # 向前线速度 0.1 m/s
        w = 0.0  # 角速度 0（直走）
        left_wheel_vel = (v - w * (task.WHEEL_BASE / 2.0)) / task.WHEEL_RADIUS
        right_wheel_vel = (v + w * (task.WHEEL_BASE / 2.0)) / task.WHEEL_RADIUS
        
        while True:
            actions = torch.zeros((num_envs, 2), device=task._device)
            
            # 在pre_physics_step中设置速度
            task.pre_physics_step(actions)
            
            # 更新进度缓冲区
            task.progress_buf += 1
            
            # 计算奖励和完成状态
            task.calculate_metrics()
            task.is_done()
            
            # 渲染（触发物理步进）
            vec_env.render()
            
            # 关键尝试：在render()之后再次设置关节速度
            # 因为物理步进可能会重置速度，我们需要在每个物理步后重新设置
            try:
                joint_velocities = torch.zeros((num_envs, task.robots.num_dof), dtype=torch.float32, device=task._device)
                joint_velocities[:, 0] = left_wheel_vel
                joint_velocities[:, 1] = right_wheel_vel
                task.robots.set_joint_velocities(joint_velocities)
            except Exception as e:
                # 如果失败，继续运行
                pass
            
            step += 1
            if step % 200 == 0:
                positions, _ = task.robots.get_world_poses()
                x_positions = positions[:, 0].cpu().numpy()
                y_positions = positions[:, 1].cpu().numpy()
                x_initial = initial_positions[:, 0].cpu().numpy()
                y_initial = initial_positions[:, 1].cpu().numpy()
                x_displacement = x_positions - x_initial
                y_displacement = y_positions - y_initial
                distance = np.sqrt(x_displacement**2 + y_displacement**2)
                
                # 检查关节速度
                try:
                    joint_vels = task.robots.get_joint_velocities()
                    print(f"Step {step}: 位置 X={x_positions}, Y={y_positions}, Z={positions[:, 2].cpu().numpy()}")
                    print(f"         移动距离: X方向={x_displacement}, 总距离={distance}")
                    print(f"         关节速度(env 0)={joint_vels[0, :2].cpu().numpy()}")
                except:
                    print(f"Step {step}: 位置 X={x_positions}, Y={y_positions}, Z={positions[:, 2].cpu().numpy()}")
                    print(f"         移动距离: X方向={x_displacement}, 总距离={distance}")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
        # 获取最终位置
        final_positions, _ = task.robots.get_world_poses()
        print("\n✅ 最终位置和移动距离：")
        for i in range(num_envs):
            x_final = final_positions[i, 0].item()
            y_final = final_positions[i, 1].item()
            z_final = final_positions[i, 2].item()
            x_initial = initial_positions[i, 0].item()
            y_initial = initial_positions[i, 1].item()
            distance = ((x_final - x_initial)**2 + (y_final - y_initial)**2)**0.5
            print(f"   环境 {i}: X={x_final:.3f}, Y={y_final:.3f}, Z={z_final:.3f}, 移动距离={distance:.3f}米")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
