"""
流程环境初始化模块
封装 World、Robots、SceneManager 等环境的初始化逻辑
"""
import math
import numpy as np
from omegaconf import OmegaConf
from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    MAX_V, MAX_W, apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import SceneManager


class EnvironmentSetup:
    """
    训练环境初始化类
    
    负责：
    1. 初始化 World 和 Physics
    2. 计算环境布局（接近正方形）
    3. 加载和初始化机器人
    4. 创建差速控制器
    5. 创建场景管理器和障碍物
    6. 创建目标标记
    """
    
    def __init__(
        self,
        env_cfg,
        num_envs,
        scene_types,
        simulation_app,
        rng,
        show_visual_walls=None,
        reset_scene_obstacles=True
    ):
        """
        初始化环境设置
        
        Args:
            env_cfg: 环境配置（OmegaConf对象）
            num_envs: 环境数量
            scene_types: 场景类型列表，长度为 num_envs
            simulation_app: Isaac Sim 应用实例
            rng: 随机数生成器
            show_visual_walls: 是否显示可视化墙壁
                - None: 让 SceneManager 自动决定（EMPTY场景不显示，其他显示）
                - bool: 所有环境统一设置
                - list: 每个环境单独设置
            reset_scene_obstacles: 是否在初始化时重置场景障碍物（Stage2需要）
        """
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.scene_types = scene_types
        self.simulation_app = simulation_app
        self.rng = rng
        self.show_visual_walls = show_visual_walls
        self.reset_scene_obstacles = reset_scene_obstacles
        
        # 从配置读取参数；场景尺寸参数缺失时回退到 SceneManager 默认值
        self.env_size = float(OmegaConf.select(env_cfg, "env.scene.env_size", default=6.0))
        self.env_gap = 2.0
        self.env_spacing = self.env_size + self.env_gap
        self.wall_thickness = float(OmegaConf.select(env_cfg, "env.scene.wall_thickness", default=0.08))
        self.wall_height = float(OmegaConf.select(env_cfg, "env.scene.wall_height", default=1.2))
        self.physics_dt = float(env_cfg.sim.dt)
        self.render_dt = 1.0 / 30.0
        self.max_v = float(MAX_V)
        self.max_w = float(MAX_W)
        
        # 初始化结果（在 setup_all 中填充）
        self.world = None
        self.robots = None
        self.scene_manager = None
        self.diff_ctrl = None
        self.left_idx = None
        self.right_idx = None
        self.env_origins = None
        self.goal_pos = None
        self.markers = None
        self.stage = None
    
    def _compute_env_layout(self):
        """
        计算接近正方形的环境布局
        
        Returns:
            env_origins: (num_envs, 3) 环境原点位置
            num_cols: 列数
            num_rows: 行数
        """
        # 找到最接近 sqrt(num_envs) 的因数作为列数
        target_cols = int(math.sqrt(self.num_envs))
        num_cols = target_cols
        # 从目标值向下找，找到能整除 num_envs 的最大因数
        for cols in range(target_cols, 0, -1):
            if self.num_envs % cols == 0:
                num_cols = cols
                break
        num_rows = self.num_envs // num_cols
        
        print(f"\n=== 环境布局 ===")
        print(f"环境数量: {self.num_envs}")
        print(f"布局: {num_cols}列 × {num_rows}行")
        
        env_origins = np.zeros((self.num_envs, 3), dtype=np.float32)
        for i in range(self.num_envs):
            ix = i % num_cols      # X方向索引（列）
            iy = i // num_cols     # Y方向索引（行）
            env_origins[i, 0] = ix * self.env_spacing
            env_origins[i, 1] = iy * self.env_spacing
        
        return env_origins, num_cols, num_rows
    
    def setup_world(self):
        """初始化 World 和 Physics"""
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.render_dt
        )
        PhysicsContext().substeps = 8
        self.world.scene.add_default_ground_plane()
        self.stage = self.world.scene.stage
        return self.world, self.stage
    
    def setup_robots(self):
        """
        加载和初始化机器人
        
        Returns:
            robots: ArticulationView 实例
            left_idx: 左轮关节索引
            right_idx: 右轮关节索引
        """
        # 计算环境布局
        self.env_origins, _, _ = self._compute_env_layout()
        
        # 加载机器人
        for i in range(self.num_envs):
            tb3_root = f"/World/envs/env_{i}/TB3"
            add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)
        
        # 等待场景稳定
        for _ in range(180):
            self.world.step(render=True)

        # 在 Articulation 加入 World 并产生接触「之前」写入 MassAPI，避免初始化后再改质量/惯量
        # 导致 PhysX 重解接触、车尾俯仰来回振荡几次。
        mass_base_cnt, _ = apply_massapi_all_tb3()
        
        # 创建机器人视图
        self.robots = ArticulationView(
            prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
            name="tb3_view",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.robots)
        self.world.reset()
        self.robots.initialize()
        # USD 若尚未展开，早期 MassAPI 可能未命中；此处补写一次（与早期参数相同，无额外弹跳风险）
        if mass_base_cnt == 0:
            apply_massapi_all_tb3()
        
        # 首轮接触后静置，耗散接触求解带来的小幅弹跳
        for _ in range(32):
            self.world.step(render=True)
        
        # 配置轮子关节
        self.left_idx, self.right_idx = configure_wheel_joints(self.robots)
        if self.left_idx is None or self.right_idx is None:
            print("[ERROR] Could not find wheel joints")
            self.simulation_app.close()
            raise RuntimeError("Could not find wheel joints")

        self.robots.set_joint_velocity_targets(
            np.zeros((self.num_envs, self.robots.num_dof), dtype=np.float32)
        )
        for _ in range(8):
            self.world.step(render=True)
        
        # 创建差速控制器
        self.diff_ctrl = DifferentialController(
            name="tb3_diff_ctrl",
            wheel_radius=WHEEL_RADIUS,
            wheel_base=WHEEL_BASE,
            max_linear_speed=self.max_v,
            max_angular_speed=self.max_w,
        )
        
        return self.robots, self.left_idx, self.right_idx, self.diff_ctrl
    
    def setup_scenes(self):
        """
        创建场景管理器和障碍物
        
        Returns:
            scene_manager: SceneManager 实例
        """
        # 处理 show_visual_walls 参数
        if self.show_visual_walls is None:
            # None: 让 SceneManager 自动决定
            show_visual_walls_list = None
        elif isinstance(self.show_visual_walls, bool):
            # bool: 所有环境统一设置
            show_visual_walls_list = [self.show_visual_walls] * self.num_envs
        elif isinstance(self.show_visual_walls, list):
            # list: 每个环境单独设置
            show_visual_walls_list = self.show_visual_walls
        else:
            raise ValueError(f"Invalid show_visual_walls type: {type(self.show_visual_walls)}")
        
        # 创建场景管理器
        self.scene_manager = SceneManager(
            num_envs=self.num_envs,
            env_size=self.env_size,
            env_origins=self.env_origins,
            stage=self.stage
        )
        self.scene_manager.wall_thickness = self.wall_thickness
        self.scene_manager.wall_height = self.wall_height
        
        # 创建场景障碍物
        self.scene_manager.create_scene_obstacles(
            scene_types=self.scene_types,
            show_visual_walls=show_visual_walls_list
        )
        
        # Stage2 需要重置障碍物位置（从地下移上来）
        if self.reset_scene_obstacles:
            self.scene_manager.reset_scene_obstacles(
                env_ids=np.arange(self.num_envs),
                rng=self.rng
            )
        
        return self.scene_manager
    
    def setup_goals(self):
        """
        创建目标位置和标记
        
        Returns:
            goal_pos: (num_envs, 3) 目标位置数组
            markers: 目标标记对象列表
        """
        self.goal_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.markers = []
        
        for i in range(self.num_envs):
            self.goal_pos[i] = self.scene_manager.get_goal_config(i, rng=self.rng)
            m = self.world.scene.add(
                VisualSphere(
                    prim_path=f"/World/envs/env_{i}/GoalMarker",
                    name=f"goal_marker_{i}",
                    position=self.goal_pos[i].tolist(),
                    radius=0.05,
                    color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                )
            )
            self.markers.append(m)
        
        return self.goal_pos, self.markers
    
    def setup_all(self):
        """
        一次性完成所有环境初始化
        
        Returns:
            dict: 包含所有初始化结果
                - world: World 实例
                - robots: ArticulationView 实例
                - scene_manager: SceneManager 实例
                - diff_ctrl: DifferentialController 实例
                - left_idx: 左轮关节索引
                - right_idx: 右轮关节索引
                - env_origins: (num_envs, 3) 环境原点
                - goal_pos: (num_envs, 3) 目标位置
                - markers: 目标标记列表
                - stage: Stage 对象
        """
        # 1. 初始化 World
        self.setup_world()
        
        # 2. 初始化机器人
        self.setup_robots()
        
        # 3. 初始化场景
        self.setup_scenes()
        
        # 4. 初始化目标
        self.setup_goals()
        
        return {
            'world': self.world,
            'robots': self.robots,
            'scene_manager': self.scene_manager,
            'diff_ctrl': self.diff_ctrl,
            'left_idx': self.left_idx,
            'right_idx': self.right_idx,
            'env_origins': self.env_origins,
            'goal_pos': self.goal_pos,
            'markers': self.markers,
            'stage': self.stage
        }
