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
                "dt": 0.005,  # 关键：进一步降低时间步长，提高接触稳定性
                "substeps": 4,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
                    # 稳定性基线参数（轻小差速车常用）
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

            # 平滑前进控制参数（减少瞬时冲击导致的前倾）
            self.target_v = 0.10
            self.current_v = 0.0
            self.v_ramp_per_step = 0.001
            self.sim_step_count = 0
            self.drive_warmup_steps = 40  # dt=0.005 下约 0.2 秒

            # 诊断实验开关：低摩擦实验（用于判断是否为抓地导致的翘头）
            self.enable_low_friction_test = True
            self.low_friction_static = 0.1
            self.low_friction_dynamic = 0.1
            self._low_friction_applied = False
            
            super().__init__(name, env, offset)
        
        def set_up_scene(self, scene):
            # 加载机器人USD资产（参考官方cartpole.py的模式）
            add_reference_to_stage(
                usd_path=self.robot_usd_path,
                prim_path=self.default_zero_env_path + "/TB3"
            )
            
            # 调用父类方法创建环境（这会克隆所有环境）
            super().set_up_scene(scene)

            # 关键修复：在 clone 之后，对每个 env 的 TB3 prim 分别应用 articulation 设置
            # 避免只对 env_0 原型 prim 生效
            self._apply_articulation_settings_all_envs()
            
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
            
            self.sim_step_count += 1

            # 关键：让所有小车向前走
            # 使用差速驱动模型：v = 0.1 m/s（向前），w = 0（不转弯）
            if self.left_wheel_idx is not None and self.right_wheel_idx is not None:
                # 前 0.2 秒不驱动，先让车体与地面接触稳定
                if self.sim_step_count <= self.drive_warmup_steps:
                    v = 0.0
                else:
                    # 使用缓启动，避免瞬时速度目标导致接触冲击和前倾
                    self.current_v = min(self.target_v, self.current_v + self.v_ramp_per_step)
                    v = self.current_v
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
            robot_positions[:, 2] = 0.03  # 进一步降低起始高度，减少落地冲击
            
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

            # 每次reset后重新从低速爬升，避免重置后瞬时抬头/点头
            self.current_v = 0.0
            self.sim_step_count = 0
            
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
            self._apply_low_friction_materials_once()
            
            # 调用reset_idx来应用初始状态（参考官方cartpole.py第143-144行）
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)

        def _apply_articulation_settings_all_envs(self):
            """在 clone 之后对每个环境 prim 应用 articulation 设置"""
            from omni.isaac.core.utils.prims import get_prim_at_path

            try:
                actor_config = self._sim_config.parse_actor_config("TB3")
            except Exception:
                actor_config = {
                    "override_usd_defaults": False,
                    "enable_self_collisions": False,
                    "enable_gyroscopic_forces": True,
                    "solver_position_iteration_count": 16,
                    "solver_velocity_iteration_count": 4,
                    "sleep_threshold": 0.005,
                    "stabilization_threshold": 0.001,
                    "contact_offset": 0.02,
                    "rest_offset": 0.0,
                }

            actor_config["solver_position_iteration_count"] = 16
            actor_config["solver_velocity_iteration_count"] = 4
            actor_config["contact_offset"] = 0.02
            actor_config["rest_offset"] = 0.0
            actor_config["sleep_threshold"] = 0.005
            actor_config["stabilization_threshold"] = 0.001

            applied = 0
            for env_id in range(self._num_envs):
                prim_path = f"/World/envs/env_{env_id}/TB3"
                robot_prim = get_prim_at_path(prim_path)
                if robot_prim:
                    self._sim_config.apply_articulation_settings("TB3", robot_prim, actor_config)
                    applied += 1

            print(f"🔧 articulation settings 已应用到 {applied}/{self._num_envs} 个 env")

        def _apply_low_friction_materials_once(self):
            """低摩擦实验：统一设置 wheel + ground 材质为低摩擦"""
            if self._low_friction_applied or not self.enable_low_friction_test:
                return

            from pxr import Sdf, UsdShade, UsdPhysics
            from omni.isaac.core.utils.stage import get_current_stage

            stage = get_current_stage()
            mat_path = Sdf.Path("/World/PhysicsMaterials/LowFrictionDebug")

            # 创建物理材质并设置低摩擦
            mat = UsdShade.Material.Define(stage, mat_path)
            mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            mat_api.CreateStaticFrictionAttr().Set(float(self.low_friction_static))
            mat_api.CreateDynamicFrictionAttr().Set(float(self.low_friction_dynamic))
            mat_api.CreateRestitutionAttr().Set(0.0)

            wheel_bind_count = 0
            ground_bind_count = 0

            # 绑定 wheel 碰撞体
            for env_id in range(self._num_envs):
                env_tb3_path = f"/World/envs/env_{env_id}/TB3"
                tb3_prim = stage.GetPrimAtPath(env_tb3_path)
                if not tb3_prim or not tb3_prim.IsValid():
                    continue

                stack = [tb3_prim]
                while stack:
                    p = stack.pop()
                    p_path_lower = str(p.GetPath()).lower()
                    is_wheel = ("wheel_left_link" in p_path_lower) or ("wheel_right_link" in p_path_lower)
                    if is_wheel and p.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI(p).Bind(
                            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
                        )
                        wheel_bind_count += 1
                    for c in p.GetChildren():
                        stack.append(c)

            # 绑定 ground 碰撞体（按路径名称包含 ground 进行匹配）
            stack = [stage.GetPseudoRoot()]
            while stack:
                p = stack.pop()
                path_lower = str(p.GetPath()).lower()
                if ("ground" in path_lower) and p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI(p).Bind(
                        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
                    )
                    ground_bind_count += 1
                for c in p.GetChildren():
                    stack.append(c)

            self._low_friction_applied = True
            print(
                f"🧪 低摩擦实验已启用: static={self.low_friction_static}, dynamic={self.low_friction_dynamic}, "
                f"wheel_bind={wheel_bind_count}, ground_bind={ground_bind_count}"
            )
        
        def _configure_wheel_drives(self):
            """配置轮子关节的驱动参数，优先使用 ArticulationView 真实生效路径"""
            # 优先路径：直接改 articulation 的 DOF 参数（最可靠）
            if self._configure_wheel_drives_from_articulation():
                self._print_wheel_drive_runtime("Articulation 配置后")
                return

            # 回退路径：USD DriveAPI（可能存在 prim 路径不匹配风险）
            print("   ⚠️ Articulation DOF 参数接口不可用，回退到 USD DriveAPI 路径")
            self._configure_wheel_drives_from_usd()
            self._print_wheel_drive_runtime("USD DriveAPI 配置后")

        def _configure_wheel_drives_from_articulation(self):
            """直接通过 ArticulationView 读写 DOF 参数，确保运行时生效"""
            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                print("   ⚠️ 轮子关节索引未找到，无法配置 Articulation DOF 参数")
                return False

            # 目标参数（与当前试验基线保持一致）
            target_kp = 0.0
            target_kd = 15.0
            target_max_effort = 0.1

            try:
                # 1) 读取现有参数
                if not hasattr(self.robots, "get_gains") or not hasattr(self.robots, "set_gains"):
                    print("   ⚠️ ArticulationView 缺少 get_gains/set_gains 接口")
                    return False

                kps, kds = self.robots.get_gains()
                if kps is None or kds is None:
                    print("   ⚠️ get_gains 返回空值")
                    return False

                # 2) 修改左右轮 kp / kd
                kps[:, self.left_wheel_idx] = target_kp
                kps[:, self.right_wheel_idx] = target_kp
                kds[:, self.left_wheel_idx] = target_kd
                kds[:, self.right_wheel_idx] = target_kd
                self.robots.set_gains(kps=kps, kds=kds)

                # 3) 若支持 max efforts，设置左右轮 max_effort
                if hasattr(self.robots, "get_max_efforts") and hasattr(self.robots, "set_max_efforts"):
                    max_efforts = self.robots.get_max_efforts()
                    if max_efforts is not None:
                        max_efforts[:, self.left_wheel_idx] = target_max_effort
                        max_efforts[:, self.right_wheel_idx] = target_max_effort
                        self.robots.set_max_efforts(max_efforts)
                        print("   ✅ 通过 ArticulationView 设置了左右轮 max_effort=0.1")
                    else:
                        print("   ⚠️ get_max_efforts 返回空值，max_effort 未通过 ArticulationView 设置")
                else:
                    print("   ⚠️ ArticulationView 不支持 get/set_max_efforts 接口")

                print("   ✅ 通过 ArticulationView 设置了左右轮 kp/kd")
                return True

            except Exception as e:
                print(f"   ⚠️ Articulation DOF 参数配置失败: {e}")
                return False

        def _configure_wheel_drives_from_usd(self):
            """回退：通过 USD DriveAPI 配置轮子关节参数"""
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
                            # damping=15: 提高速度环阻尼，降低抽动
                            # max_force=0.1: 按 TB3 质量量级限制可用轮上扭矩
                            drive_api.GetStiffnessAttr().Set(0.0)
                            drive_api.GetDampingAttr().Set(15.0)
                            drive_api.GetMaxForceAttr().Set(0.1)
                            configured_count += 1
                            if env_id == 0:  # 只打印第一个环境的详细信息
                                print(f"   ✅ 环境 {env_id}: 已配置左轮驱动参数 (stiffness=0, damping=15, max_force=0.1)")
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
                            drive_api.GetDampingAttr().Set(15.0)
                            drive_api.GetMaxForceAttr().Set(0.1)
                            configured_count += 1
                            if env_id == 0:
                                print(f"   ✅ 环境 {env_id}: 已配置右轮驱动参数 (stiffness=0, damping=15, max_force=0.1)")
                        except Exception as e:
                            if env_id == 0:
                                print(f"   ⚠️ 配置右轮驱动时出错: {e}")
            
            if configured_count > 0:
                print(f"   ✅ 总共配置了 {configured_count} 个关节驱动（{self._num_envs}个环境 × 2个轮子）")
            else:
                print(f"   ⚠️ 未能配置任何关节驱动，可能路径不匹配")

        def _print_wheel_drive_runtime(self, tag):
            """打印运行时真实生效的轮子 DOF 参数（用于判断配置是否落地）"""
            if self.left_wheel_idx is None or self.right_wheel_idx is None:
                return
            try:
                print(f"📊 {tag} - 轮子 DOF 运行时参数 (env0):")
                print(f"   左轮 idx={self.left_wheel_idx}, 右轮 idx={self.right_wheel_idx}")

                if hasattr(self.robots, "get_dof_types"):
                    try:
                        dof_types = self.robots.get_dof_types()
                        print(f"   dof_types = {dof_types}")
                    except Exception as e:
                        print(f"   ⚠️ get_dof_types 读取失败: {e}")

                if hasattr(self.robots, "get_gains"):
                    kps, kds = self.robots.get_gains()
                    print(
                        f"   gains: "
                        f"L(kp={kps[0, self.left_wheel_idx].item():.6f}, kd={kds[0, self.left_wheel_idx].item():.6f}), "
                        f"R(kp={kps[0, self.right_wheel_idx].item():.6f}, kd={kds[0, self.right_wheel_idx].item():.6f})"
                    )

                if hasattr(self.robots, "get_max_efforts"):
                    max_efforts = self.robots.get_max_efforts()
                    if max_efforts is not None:
                        print(
                            f"   max_effort: "
                            f"L={max_efforts[0, self.left_wheel_idx].item():.6f}, "
                            f"R={max_efforts[0, self.right_wheel_idx].item():.6f}"
                        )
                    else:
                        print("   ⚠️ get_max_efforts 返回空值")
            except Exception as e:
                print(f"   ⚠️ 运行时轮子 DOF 参数打印失败: {e}")
        
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
                        left_omega = joint_velocities[0, task.left_wheel_idx].item()
                        right_omega = joint_velocities[0, task.right_wheel_idx].item()
                        omega_avg = 0.5 * (left_omega + right_omega)
                        v_expected = task.WHEEL_RADIUS * omega_avg

                        root_vel = task.robots.get_velocities()
                        vx = root_vel[0, 0].item()
                        vy = root_vel[0, 1].item()
                        v_planar = math.sqrt(vx * vx + vy * vy)
                        ratio = v_planar / (abs(v_expected) + 1e-6)

                        print(
                            f"   关节速度 (env 0): left={left_omega:.4f}, right={right_omega:.4f}, "
                            f"omega_avg={omega_avg:.4f}"
                        )
                        print(
                            f"   速度一致性 (env 0): v_xy=({vx:.4f}, {vy:.4f}), |v_xy|={v_planar:.4f}, "
                            f"r*omega={v_expected:.4f}, ratio={ratio:.3f}"
                        )
                    except:
                        pass
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
