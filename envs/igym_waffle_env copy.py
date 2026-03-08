"""
WaffleDrive RL Task Environment
实现论文中的 RL 环境，包含观察空间、奖励函数和重置逻辑
"""
import torch
import numpy as np
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from .rewards import RewardCalculator
from .observations import assemble_observations, get_base_velocity_from_tensor
from .user_intent import compute_user_intent_torch

# 导入场景和机器人模块
from sim.scenes import SceneManager, SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR
from sim.robot import tb3_setup, tb3_config
from sim.scenes.scene_base import quat_wxyz_from_yaw


class WaffleDriveTask(RLTask):
    """
    WaffleDrive RL 任务环境
    """
    def __init__(self, name, sim_config, env, offset=None):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        # 环境参数
        self._num_envs = int(self._task_cfg["env"]["numEnvs"])
        self._env_spacing = float(self._task_cfg["env"]["envSpacing"])
        
        # 观察和动作空间
        self._num_observations = 44  # LiDAR[36] + User Input[2] + Base Velocity[2] + Action History[4]
                                      # Base Velocity 是测量的当前线速度和角速度 (v, w)
        self._num_actions = 2  # [v, w]
        
        # 机器人 USD 路径（统一使用 sim 模块中的配置）
        self.robot_usd_path = tb3_config.TB3_USD
        
        # 从配置读取参数
        self.max_v = float(self._task_cfg["env"]["robot_limits"]["max_v"])
        self.max_w = float(self._task_cfg["env"]["robot_limits"]["max_w"])
        self.lidar_max_range = float(self._task_cfg["env"]["lidar"]["max_range"])
        self.lidar_num_rays = int(self._task_cfg["env"]["lidar"]["num_rays"])
        self.stage_level = int(self._task_cfg["env"]["randomization"]["stage_level"])
        self.reset_dist = float(self._task_cfg["env"]["resetDist"])
        
        # 超时时间（秒），与原始代码保持一致
        self.TIMEOUT_SECONDS = 60.0
        
        # 差速驱动参数（TB3 Waffle Pi）
        self.wheel_radius = tb3_config.WHEEL_RADIUS
        self.wheel_base = tb3_config.WHEEL_BASE
        
        # 奖励函数参数
        reward_cfg = self._task_cfg["env"]["rewards"]
        self.ra = float(reward_cfg.get("ra", 0.5))
        self.rl = float(reward_cfg.get("rl", -0.5))
        self.rh = float(reward_cfg.get("rh", -0.5))
        self.phi_thresh = float(reward_cfg.get("phi_thresh", 0.2))
        self.ras = float(reward_cfg.get("ras", -0.02))
        self.rc = float(reward_cfg.get("rc", -1.0))
        self.rcrit = float(reward_cfg.get("rcrit", -1.0))
        self.rcol = float(reward_cfg.get("rcol", -100.0))
        self.d_col = float(reward_cfg.get("d_col", 0.12))
        self.d_crit = float(reward_cfg.get("d_crit", 0.35))
        self.enable_centripetal = bool(reward_cfg.get("enable_centripetal", False))
        self.w_centripetal = float(reward_cfg.get("w_centripetal", 0.0))
        
        # 调用父类初始化
        super().__init__(name, env, offset)
        
        # 动作历史缓冲区（存储历史时刻的测量速度，不是命令动作）
        self.action_history = torch.zeros(
            (self._num_envs, 4), dtype=torch.float32, device=self._device
        )  # [v_t-1, w_t-1, v_t-2, w_t-2] - 历史测量速度
        
        # 上一时刻的测量速度（用于更新 action_history）
        self.prev_measured_v = torch.zeros(
            (self._num_envs,), dtype=torch.float32, device=self._device
        )
        self.prev_measured_w = torch.zeros(
            (self._num_envs,), dtype=torch.float32, device=self._device
        )
        
        # 当前动作（用于奖励计算）
        self.current_actions = torch.zeros(
            (self._num_envs, 2), dtype=torch.float32, device=self._device
        )
        
        # 初始化奖励计算器（max_v 和 max_w 在 envs 中统一管理，直接传入）
        self.reward_calculator = RewardCalculator(
            num_envs=self._num_envs,
            device=self._device,
            ra=self.ra, rl=self.rl,
            rh=self.rh, phi_thresh=self.phi_thresh,
            ras=self.ras, rc=self.rc, rcrit=self.rcrit, rcol=self.rcol,
            d_col=self.d_col, d_crit=self.d_crit,
            enable_centripetal=self.enable_centripetal,
            w_centripetal=self.w_centripetal,
            max_v=self.max_v, max_w=self.max_w
        )
        
        # 存储机器人位置、朝向等（用于观察和奖励计算）
        self.robot_positions = None
        self.robot_orientations = None
        self.target_positions = None
        self.env_origins = None
        
        # 存储每个环境的 episode 开始时间（用于超时检测）
        self.episode_start_time = torch.zeros(
            (self._num_envs,), dtype=torch.float32, device=self._device
        )
        
        # 当前仿真时间（在 calculate_metrics 中更新）
        self.current_time = 0.0
        
        # 存储 user_intent（用于优化，避免重复计算）
        self.user_intent_env = None
        
        # 场景配置
        scene_cfg = self._task_cfg["env"].get("scene", {})
        self.env_size = float(scene_cfg.get("env_size", 6.0))
        self.show_visual_walls = bool(scene_cfg.get("show_visual_walls", False))
        
        # 根据 stage_level 确定场景类型
        if self.stage_level == 1:
            # Stage 1: 所有环境都是 empty
            self.scene_types = [SCENE_EMPTY] * self._num_envs
        elif self.stage_level == 2:
            # Stage 2: 循环分配四个场景
            scene_list = [SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR]
            self.scene_types = [scene_list[i % 4] for i in range(self._num_envs)]
        else:
            # 默认：所有环境都是 empty
            self.scene_types = [SCENE_EMPTY] * self._num_envs
        
        # 初始化场景管理器（在 set_up_scene 中设置 stage 和 env_origins）
        self.scene_manager = SceneManager(
            num_envs=self._num_envs,
            env_size=self.env_size
        )
        
        # 随机数生成器（用于场景重置）
        self.rng = np.random.default_rng(42)
    
    def set_up_scene(self, scene):
        """设置场景"""
        # 加载机器人
        add_reference_to_stage(self.robot_usd_path, self.default_zero_env_path + "/TB3")
        super().set_up_scene(scene)
        
        # 创建机器人视图
        self.robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/TB3/a__namespace_base_footprint",
            name="tb3_view",
            reset_xform_properties=False
        )
        scene.add(self.robots)
        
        # 设置场景管理器的 stage 和 env_origins
        stage = omni.usd.get_context().get_stage()
        self.scene_manager.set_stage(stage)
        self.scene_manager.set_env_origins(self._env_pos.cpu().numpy())
        
        # 创建场景障碍物
        show_visual_walls_list = [self.show_visual_walls] * self._num_envs
        self.scene_manager.create_scene_obstacles(
            scene_types=self.scene_types,
            show_visual_walls=show_visual_walls_list
        )
        
        # 配置机器人物理属性（质量和质心）
        tb3_setup.apply_massapi_all_tb3(
            base_mass=tb3_config.BASE_MASS,
            com_x=tb3_config.COM_X,
            com_z=tb3_config.COM_Z
        )
    
    def post_reset(self):
        """重置后处理"""
        # 配置轮子关节参数并获取 DOF 索引
        self.left_wheel_idx, self.right_wheel_idx = tb3_setup.configure_wheel_joints(
            self.robots,
            wheel_kp=tb3_config.WHEEL_KP,
            wheel_kd=tb3_config.WHEEL_KD,
            wheel_max_effort=tb3_config.WHEEL_MAX_EFFORT
        )
        
        if self.left_wheel_idx is None or self.right_wheel_idx is None:
            # 如果找不到，尝试手动查找
            try:
                self.left_wheel_idx = self.robots.get_dof_index("wheel_left_joint")
                self.right_wheel_idx = self.robots.get_dof_index("wheel_right_joint")
            except:
                try:
                    self.left_wheel_idx = self.robots.get_dof_index("a__namespace_wheel_left_joint")
                    self.right_wheel_idx = self.robots.get_dof_index("a__namespace_wheel_right_joint")
                except:
                    # 如果还是找不到，使用索引（假设前两个 DOF 是左右轮）
                    self.left_wheel_idx = 0
                    self.right_wheel_idx = 1
        
        env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(env_ids)
    
    def reset_idx(self, env_ids):
        """重置指定环境"""
        n = len(env_ids)
        env_ids_np = env_ids.cpu().numpy()
        
        # 重置场景障碍物（随机化）
        self.scene_manager.reset_scene_obstacles(env_ids_np, rng=self.rng)
        
        # 根据场景类型获取机器人初始位置和朝向
        spawn_positions_list = []
        spawn_rotations_list = []
        goal_positions_list = []
        
        for env_id in env_ids_np:
            # 获取机器人初始配置
            spawn_pos, spawn_yaw = self.scene_manager.get_robot_spawn_config(env_id, rng=self.rng)
            spawn_quat = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))[0]  # (4,)
            
            spawn_positions_list.append(spawn_pos)
            spawn_rotations_list.append(spawn_quat)
            
            # 获取目标位置
            goal_pos = self.scene_manager.get_goal_config(env_id, rng=self.rng)
            goal_positions_list.append(goal_pos)
        
        # 转换为 torch tensor
        spawn_positions = torch.tensor(np.stack(spawn_positions_list), dtype=torch.float32, device=self._device)
        spawn_rotations = torch.tensor(np.stack(spawn_rotations_list), dtype=torch.float32, device=self._device)
        goal_positions = torch.tensor(np.stack(goal_positions_list), dtype=torch.float32, device=self._device)
        
        # 设置机器人位置和朝向
        self.robots.set_world_poses(spawn_positions, spawn_rotations, indices=env_ids)
        self.robots.set_velocities(
            torch.zeros((n, 6), device=self._device), indices=env_ids
        )
        self.robots.set_joint_velocities(
            torch.zeros((n, self.robots.num_dof), device=self._device), indices=env_ids
        )
        
        # 存储目标位置（用于观察和奖励计算）
        if self.target_positions is None:
            self.target_positions = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.target_positions[env_ids] = goal_positions
        
        # 重置缓冲区
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # 重置 episode 开始时间
        self.episode_start_time[env_ids] = self.current_time
        
        # 重置动作历史
        self.action_history[env_ids] = 0.0
        self.current_actions[env_ids] = 0.0
        # 重置上一时刻的测量速度
        self.prev_measured_v[env_ids] = 0.0
        self.prev_measured_w[env_ids] = 0.0
        
        # 重置 user_intent（下次 get_observations 会重新计算）
        # 注意：这里不重置整个 user_intent_env，因为可能只有部分环境重置
        # 在 get_observations 中会重新计算所有环境的 user_intent
    
    def pre_physics_step(self, actions):
        """物理步进前处理"""
        # 处理重置
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # 保存当前动作（用于奖励计算和后续应用）
        # actions: (N, 2) [v, w]，归一化的 [-1, 1]
        self.current_actions = actions.clone()
        
        # 反归一化动作到物理值（用于应用到机器人）
        actions_denorm = actions.clone()
        actions_denorm[:, 0] = actions_denorm[:, 0] * self.max_v  # v 反归一化到 [-max_v, max_v]
        actions_denorm[:, 1] = actions_denorm[:, 1] * self.max_w   # w 反归一化到 [-max_w, max_w]
        
        # 注意：action_history 的更新在 get_observations 中进行
        # 因为需要获取测量速度，而不是命令动作
        
        # 将动作转换为差速驱动的左右轮角速度并应用到机器人
        v_cmd = actions_denorm[:, 0]  # 线速度命令 (m/s)
        w_cmd = actions_denorm[:, 1]  # 角速度命令 (rad/s)
        
        # 差速驱动运动学（wheel_base = 0.288 m，左右轮中心距离）：
        # v = (v_left + v_right) / 2  (线速度 = 左右轮速度的平均值)
        # w = (v_right - v_left) / wheel_base  (角速度 = 速度差 / 轮距)
        # 反推得到左右轮速度：
        # v_left = v - w * wheel_base / 2
        # v_right = v + w * wheel_base / 2
        v_left = v_cmd - w_cmd * (self.wheel_base / 2.0)
        v_right = v_cmd + w_cmd * (self.wheel_base / 2.0)
        
        # 转换为轮子角速度 (rad/s)
        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius
        
        # 应用轮子角速度到机器人
        # 设置关节速度（DOF 索引在 post_reset 中已获取）
        joint_velocities = torch.zeros((self._num_envs, self.robots.num_dof), device=self._device)
        joint_velocities[:, self.left_wheel_idx] = w_left
        joint_velocities[:, self.right_wheel_idx] = w_right
        
        self.robots.set_joint_velocities(joint_velocities)
    
    def get_observations(self):
        """获取观察"""
        # 获取机器人状态
        self.robot_positions, self.robot_orientations = self.robots.get_world_poses()
        self.env_origins = self._env_pos.clone()
        
        # 目标位置已在 reset_idx 中设置，这里不需要重新计算
        # 如果 target_positions 还未初始化，使用默认值
        if self.target_positions is None:
            # 使用配置中的默认目标位置
            target_offset = torch.tensor(
                self._task_cfg["env"]["target"]["pos"], device=self._device
            )
            self.target_positions = self.env_origins + target_offset
        
        # 计算 LiDAR 范围（简化版本，实际应从传感器读取）
        lidar_ranges = self._compute_lidar_ranges(
            self.robot_positions, self.robot_orientations
        )
        
        # 获取基础速度（max_v 和 max_w 在 envs 中统一管理）
        robot_velocities = self.robots.get_velocities()
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities,
            max_v=self.max_v,
            max_w=self.max_w
        )
        
        # 获取当前时刻的测量速度（物理值，未归一化）
        # 用于更新 action_history（存储历史测量速度）
        current_measured_v = robot_velocities[:, 0]  # 线速度 v
        current_measured_w = robot_velocities[:, 5]  # 角速度 w
        
        # 更新动作历史（存储历史时刻的测量速度，不是命令动作）
        # Action History 存储的是 t-1 和 t-2 时刻的测量速度
        self.action_history[:, 2:] = self.action_history[:, :2].clone()  # 移动历史：t-2 = 旧的 t-1
        self.action_history[:, 0] = self.prev_measured_v.clone()  # v_t-1：上一时刻的测量线速度
        self.action_history[:, 1] = self.prev_measured_w.clone()  # w_t-1：上一时刻的测量角速度
        
        # 更新上一时刻的测量速度（用于下次观察）
        self.prev_measured_v = current_measured_v.clone()
        self.prev_measured_w = current_measured_w.clone()
        
        # 计算用户意图向量（计算一次，用于观察和奖励）
        _, user_intent_env, _ = compute_user_intent_torch(
            self.robot_positions,
            self.robot_orientations,
            self.target_positions,
            self.env_origins,
            normalize=True
        )
        self.user_intent_env = user_intent_env  # 存储用于 calculate_metrics
        
        # 组装观察向量（传入已计算的 user_intent，避免重复计算）
        obs_np = assemble_observations(
            robot_positions=self.robot_positions.cpu().numpy(),
            robot_orientations=self.robot_orientations.cpu().numpy(),
            goal_positions=self.target_positions.cpu().numpy(),
            env_origins=self.env_origins.cpu().numpy(),
            lidar_ranges=lidar_ranges.cpu().numpy(),
            base_vel=base_vel_np,
            action_history=self.action_history.cpu().numpy(),
            max_v=self.max_v,
            max_w=self.max_w,
            lidar_max_range=self.lidar_max_range,
            user_input=user_intent_env.cpu().numpy()  # 传入已计算的 user_intent
        )
        
        self.obs_buf = torch.from_numpy(obs_np).to(self._device)
        
        return {self.name: {"obs": self.obs_buf}}
    
    def calculate_metrics(self):
        """计算奖励和指标"""
        # 计算 LiDAR 范围
        lidar_ranges = self._compute_lidar_ranges(
            self.robot_positions, self.robot_orientations
        )
        
        # 获取机器人速度
        robot_velocities = self.robots.get_velocities()
        
        # 计算奖励（传入已计算的 user_intent，避免重复计算）
        # 如果 get_observations 中已计算，则使用存储的值；否则重新计算
        if self.user_intent_env is None:
            _, user_intent_env, _ = compute_user_intent_torch(
                self.robot_positions,
                self.robot_orientations,
                self.target_positions,
                self.env_origins,
                normalize=True
            )
        else:
            user_intent_env = self.user_intent_env
        
        self.rew_buf[:] = self.reward_calculator.compute_rewards(
            lidar_ranges=lidar_ranges,
            robot_positions=self.robot_positions,
            robot_velocities=robot_velocities,
            robot_orientations=self.robot_orientations,
            goal_positions=self.target_positions,
            env_origins=self.env_origins,
            actions=self.current_actions,
            action_history=self.action_history,
            user_intent_env=user_intent_env  # 传入已计算的 user_intent
        )
        
        # 更新进度缓冲区和仿真时间
        self.progress_buf += 1
        self.current_time += self._task_cfg["sim"]["dt"]
        
        # 检查重置条件
        # 1. 超时（使用时间而不是步数，与原始代码保持一致）
        reset_timeout = (self.current_time - self.episode_start_time) >= self.TIMEOUT_SECONDS
        
        # 2. 到达目标（距离 < resetDist）
        robot_xy = self.robot_positions[:, :2]
        goal_xy = self.target_positions[:, :2]
        dist_to_goal = torch.linalg.norm(robot_xy - goal_xy, dim=1)
        reset_goal_reached = dist_to_goal < self.reset_dist
        
        # 3. 发生碰撞
        # 3a. 障碍物碰撞（从障碍物奖励判断，r_obstacles <= rcol 表示碰撞）
        r_obstacles = self.reward_calculator.reward_components.get("obstacles", torch.zeros(self._num_envs, device=self._device))
        reset_collision_obstacle = r_obstacles <= (self.reward_calculator.rcol + 1e-5)
        
        # 3b. 边界碰撞（使用场景管理器检测）
        robot_pos_local = self.robot_positions[:, :2].cpu().numpy() - self.env_origins[:, :2].cpu().numpy()
        reset_collision_boundary = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        for i in range(self._num_envs):
            if self.scene_manager.check_boundary_collision(i, robot_pos_local[i]):
                reset_collision_boundary[i] = True
            # 也检查障碍物碰撞（作为补充，因为 LiDAR 可能检测不到）
            if self.scene_manager.check_collision_with_obstacles(i, robot_pos_local[i]):
                reset_collision_boundary[i] = True
        
        reset_collision = reset_collision_obstacle | reset_collision_boundary
        
        # 设置重置标志（任一条件满足即重置）
        self.reset_buf[:] = torch.where(
            reset_timeout | reset_goal_reached | reset_collision,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
    
    def _compute_lidar_ranges(self, robot_positions, robot_orientations):
        """
        计算 LiDAR 范围（简化版本）
        实际应该使用 Isaac Sim 的 LiDAR 传感器
        """
        # 这里返回一个占位符，实际应该从传感器读取
        # 暂时返回最大范围（表示无障碍）
        lidar_ranges = torch.ones(
            (self._num_envs, self.lidar_num_rays),
            dtype=torch.float32,
            device=self._device
        ) * self.lidar_max_range
        
        return lidar_ranges
