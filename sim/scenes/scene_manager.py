"""
Scene Manager
场景管理器：创建、重置和碰撞检测
"""
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

from .scene_base import (
    SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR,
    sample_goal_offsets, quat_wxyz_from_yaw
)
from ..robot.tb3_config import ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y


class SceneManager:
    """场景管理器"""
    
    def __init__(self, num_envs, env_size=6.0, env_origins=None, stage=None):
        """
        初始化场景管理器
        
        Args:
            num_envs: 环境数量
            env_size: 环境大小 (m)
            env_origins: (num_envs, 3) 环境原点位置
            stage: USD stage 对象
        """
        self.num_envs = num_envs
        self.env_size = env_size
        self.env_half = env_size * 0.5
        self.env_origins = env_origins
        self.stage = stage
        
        # 场景 handles（每个环境的障碍物引用）
        self.scene_handles = [None] * num_envs
        self.scene_types = [SCENE_EMPTY] * num_envs
        
        # 场景配置
        self.wall_thickness = 0.08
        self.wall_height = 1.2
        
    def set_stage(self, stage):
        """设置 USD stage"""
        self.stage = stage
    
    def set_env_origins(self, env_origins):
        """设置环境原点"""
        self.env_origins = env_origins
    
    def create_scene_obstacles(self, scene_types, show_visual_walls=None):
        """
        为所有环境创建障碍物
        
        Args:
            scene_types: (num_envs,) 每个环境的场景类型列表
            show_visual_walls: (num_envs,) 是否显示边界墙，None 则根据场景类型自动决定
        """
        if self.stage is None:
            raise ValueError("Stage not set. Call set_stage() first.")
        if self.env_origins is None:
            raise ValueError("Env origins not set. Call set_env_origins() first.")
        
        self.scene_types = scene_types
        
        if show_visual_walls is None:
            # 默认：empty 场景不显示墙，其他场景显示
            show_visual_walls = [st != SCENE_EMPTY for st in scene_types]
        
        for i in range(self.num_envs):
            handles = self._create_env_obstacles(
                env_id=i,
                scene_type=scene_types[i],
                show_visual_walls=show_visual_walls[i]
            )
            self.scene_handles[i] = handles
    
    def _create_env_obstacles(self, env_id, scene_type, show_visual_walls=False):
        """
        为单个环境创建障碍物
        
        Returns:
            handles: 障碍物 handles 字典
        """
        root = f"/World/envs/env_{env_id}"
        origin = self.env_origins[env_id]
        half = self.env_half
        
        handles = {}
        
        # 创建边界墙
        walls = []
        wall_defs = [
            (f"{root}/BoundLeft", (self.wall_thickness, self.env_size + 2 * self.wall_thickness, self.wall_height),
             (origin[0] - half - self.wall_thickness * 0.5, origin[1], self.wall_height * 0.5)),
            (f"{root}/BoundRight", (self.wall_thickness, self.env_size + 2 * self.wall_thickness, self.wall_height),
             (origin[0] + half + self.wall_thickness * 0.5, origin[1], self.wall_height * 0.5)),
            (f"{root}/BoundFront", (self.env_size + 2 * self.wall_thickness, self.wall_thickness, self.wall_height),
             (origin[0], origin[1] - half - self.wall_thickness * 0.5, self.wall_height * 0.5)),
            (f"{root}/BoundBack", (self.env_size + 2 * self.wall_thickness, self.wall_thickness, self.wall_height),
             (origin[0], origin[1] + half + self.wall_thickness * 0.5, self.wall_height * 0.5)),
        ]
        for p, size, center in wall_defs:
            prim, t_op, s_op = self._create_cube(
                self.stage, p, size_xyz=size, center_xyz=center,
                visible=show_visual_walls, color=(0.6, 0.6, 0.6), collision=True
            )
            walls.append({"prim": prim, "t_op": t_op, "s_op": s_op})
        handles["walls"] = walls
        
        # 创建 Box 障碍物（初始隐藏）
        box_prim, box_t, box_s = self._create_cube(
            self.stage, f"{root}/ScenarioBox",
            size_xyz=(1.0, 1.0, 0.6), center_xyz=(origin[0], origin[1], -10.0),
            visible=True, color=(0.85, 0.85, 0.85), collision=True
        )
        handles["box"] = {"prim": box_prim, "t_op": box_t, "s_op": box_s}
        
        # 创建 Cylinder 障碍物（初始隐藏）
        cyls = []
        for k in range(4):
            c_prim, c_t, c_geom = self._create_cylinder(
                self.stage, f"{root}/ScenarioCyl_{k}",
                radius=0.18, height=0.7, center_xyz=(origin[0], origin[1], -10.0),
                visible=True, color=(0.7, 0.7, 0.7), collision=True
            )
            cyls.append({"prim": c_prim, "t_op": c_t, "geom": c_geom})
        handles["cyls"] = cyls
        
        # 创建 Door 障碍物（初始隐藏）
        door_l_prim, door_l_t, door_l_s = self._create_cube(
            self.stage, f"{root}/DoorSegLeft",
            size_xyz=(0.12, 2.0, 1.0), center_xyz=(origin[0], origin[1], -10.0),
            visible=True, color=(0.7, 0.7, 0.7), collision=True
        )
        door_r_prim, door_r_t, door_r_s = self._create_cube(
            self.stage, f"{root}/DoorSegRight",
            size_xyz=(0.12, 2.0, 1.0), center_xyz=(origin[0], origin[1], -10.0),
            visible=True, color=(0.7, 0.7, 0.7), collision=True
        )
        handles["door_left"] = {"prim": door_l_prim, "t_op": door_l_t, "s_op": door_l_s}
        handles["door_right"] = {"prim": door_r_prim, "t_op": door_r_t, "s_op": door_r_s}
        
        # 设置可见性
        box_visible = True if scene_type == SCENE_BOX else show_visual_walls
        self._set_visibility(box_prim, box_visible)
        for c in cyls:
            self._set_visibility(c["prim"], show_visual_walls)
        self._set_visibility(door_l_prim, show_visual_walls)
        self._set_visibility(door_r_prim, show_visual_walls)
        
        return handles
    
    def reset_scene_obstacles(self, env_ids, rng=None):
        """
        重置场景障碍物（随机化位置/大小）
        
        Args:
            env_ids: 要重置的环境 ID 列表
            rng: numpy RandomGenerator，如果为 None 则使用默认
        """
        if rng is None:
            rng = np.random.default_rng()
        
        for env_id in env_ids:
            scene_type = self.scene_types[env_id]
            handles = self.scene_handles[env_id]
            env_origin = self.env_origins[env_id]
            
            # 先隐藏所有障碍物
            self._hide_prim_below_ground(handles["box"]["t_op"])
            for c in handles["cyls"]:
                self._hide_prim_below_ground(c["t_op"])
            self._hide_prim_below_ground(handles["door_left"]["t_op"])
            self._hide_prim_below_ground(handles["door_right"]["t_op"])
            
            # 根据场景类型重置障碍物
            if scene_type == SCENE_BOX:
                l = rng.uniform(1.0, 3.0)
                b = rng.uniform(1.0, 3.0)
                h = 0.6
                handles["box"]["s_op"].Set(Gf.Vec3d(float(l), float(b), float(h)))
                handles["box"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1]), h * 0.5))
                # 确保BOX可见
                self._set_visibility(handles["box"]["prim"], True)
            
            elif scene_type == SCENE_CYLINDER:
                h = 0.7
                fixed_positions = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
                for i, c in enumerate(handles["cyls"]):
                    rad = rng.uniform(0.1, 0.3)
                    ox, oy = fixed_positions[i]
                    c["geom"].CreateRadiusAttr(float(rad))
                    c["geom"].CreateHeightAttr(float(h))
                    c["t_op"].Set(Gf.Vec3d(float(env_origin[0] + ox), float(env_origin[1] + oy), h * 0.5))
                    # 确保圆柱体可见
                    self._set_visibility(c["prim"], True)
            
            elif scene_type == SCENE_DOOR:
                door_w = rng.uniform(0.9, 1.75)
                wall_t = 0.12
                wall_h = 1.0
                max_offset = 2.0 - door_w * 0.5
                door_offset_y = rng.uniform(-max_offset, max_offset)
                door_top = door_offset_y + door_w * 0.5
                door_bottom = door_offset_y - door_w * 0.5
                upper_seg_len = 3.0 - door_top
                upper_center_local_y = (3.0 + door_top) * 0.5
                lower_seg_len = door_bottom - (-3.0)
                lower_center_local_y = (door_bottom + (-3.0)) * 0.5
                
                min_seg_len = 0.1
                if upper_seg_len < min_seg_len:
                    upper_seg_len = min_seg_len
                    upper_center_local_y = 2.5
                if lower_seg_len < min_seg_len:
                    lower_seg_len = min_seg_len
                    lower_center_local_y = -2.5
                
                handles["door_left"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(upper_seg_len), float(wall_h)))
                handles["door_right"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(lower_seg_len), float(wall_h)))
                handles["door_left"]["t_op"].Set(
                    Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + upper_center_local_y), wall_h * 0.5)
                )
                handles["door_right"]["t_op"].Set(
                    Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + lower_center_local_y), wall_h * 0.5)
                )
                # 确保门的两段墙可见
                self._set_visibility(handles["door_left"]["prim"], True)
                self._set_visibility(handles["door_right"]["prim"], True)
            
            # EMPTY场景：确保所有障碍物都隐藏（已经在上面隐藏了）
    
    def get_robot_spawn_config(self, env_id, rng=None):
        """
        获取机器人初始位置和朝向配置
        
        Args:
            env_id: 环境 ID
            rng: numpy RandomGenerator
        
        Returns:
            spawn_pos: (3,) 世界坐标系位置
            spawn_yaw: 标量，yaw 角（弧度）
        """
        if rng is None:
            rng = np.random.default_rng()
        
        scene_type = self.scene_types[env_id]
        env_origin = self.env_origins[env_id]
        
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            local_x = 0.0
            local_y = 0.0
            spawn_yaw = rng.uniform(-np.pi, np.pi)
        else:  # SCENE_BOX or SCENE_DOOR
            local_x = 2.5
            local_y = 2.5
            dx = -2.5 - local_x
            dy = -2.5 - local_y
            spawn_yaw = np.arctan2(dy, dx)
        
        spawn_pos = np.array([env_origin[0] + local_x, env_origin[1] + local_y, 0.035], dtype=np.float32)
        return spawn_pos, spawn_yaw
    
    def get_goal_config(self, env_id, rng=None):
        """
        获取目标位置配置
        
        Args:
            env_id: 环境 ID
            rng: numpy RandomGenerator
        
        Returns:
            goal_pos: (3,) 世界坐标系位置
        """
        if rng is None:
            rng = np.random.default_rng()
        
        scene_type = self.scene_types[env_id]
        env_origin = self.env_origins[env_id]
        
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            goal_offset = sample_goal_offsets(1, r_min=1.0, r_max=1.5)[0]
            goal_pos = np.array([
                env_origin[0] + goal_offset[0],
                env_origin[1] + goal_offset[1],
                0.03
            ], dtype=np.float32)
        else:  # SCENE_BOX or SCENE_DOOR
            goal_pos = np.array([
                env_origin[0] - 2.5,
                env_origin[1] - 2.5,
                0.03
            ], dtype=np.float32)
        
        return goal_pos
    
    def check_collision_with_obstacles(self, env_id, robot_pos_local):
        """
        检测机器人与障碍物碰撞
        
        Args:
            env_id: 环境 ID
            robot_pos_local: (2,) 机器人局部位置 [x, y]
        
        Returns:
            bool: 是否碰撞
        """
        scene_type = self.scene_types[env_id]
        handles = self.scene_handles[env_id]
        env_origin = self.env_origins[env_id]
        
        x, y = robot_pos_local[0], robot_pos_local[1]
        
        if scene_type == SCENE_BOX:
            box_t = handles["box"]["t_op"]
            box_s = handles["box"]["s_op"]
            box_pos = box_t.Get()
            box_size = box_s.Get()
            
            box_center_world = np.array([box_pos[0], box_pos[1]])
            robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
            
            box_half_x = box_size[0] * 0.5
            box_half_y = box_size[1] * 0.5
            
            box_min_x = box_center_world[0] - box_half_x
            box_max_x = box_center_world[0] + box_half_x
            box_min_y = box_center_world[1] - box_half_y
            box_max_y = box_center_world[1] + box_half_y
            
            robot_min_x = robot_pos_world[0] - ROBOT_HALF_LENGTH_X
            robot_max_x = robot_pos_world[0] + ROBOT_HALF_LENGTH_X
            robot_min_y = robot_pos_world[1] - ROBOT_HALF_WIDTH_Y
            robot_max_y = robot_pos_world[1] + ROBOT_HALF_WIDTH_Y
            
            if (robot_min_x < box_max_x and robot_max_x > box_min_x and
                robot_min_y < box_max_y and robot_max_y > box_min_y):
                return True
        
        elif scene_type == SCENE_CYLINDER:
            for cyl in handles["cyls"]:
                cyl_t = cyl["t_op"]
                cyl_geom = cyl["geom"]
                cyl_pos = cyl_t.Get()
                cyl_radius = cyl_geom.GetRadiusAttr().Get()
                
                cyl_center_world = np.array([cyl_pos[0], cyl_pos[1]])
                robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
                
                dist_to_cyl = np.linalg.norm(robot_pos_world - cyl_center_world)
                if dist_to_cyl < (cyl_radius + max(ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y)):
                    return True
        
        elif scene_type == SCENE_DOOR:
            door_l_t = handles["door_left"]["t_op"]
            door_r_t = handles["door_right"]["t_op"]
            door_l_s = handles["door_left"]["s_op"]
            door_r_s = handles["door_right"]["s_op"]
            
            door_l_pos = door_l_t.Get()
            door_r_pos = door_r_t.Get()
            door_l_size = door_l_s.Get()
            door_r_size = door_r_s.Get()
            
            robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
            
            # 检查左墙
            wall_l_center = np.array([door_l_pos[0], door_l_pos[1]])
            wall_l_half_len = door_l_size[1] * 0.5
            wall_l_thickness = door_l_size[0] * 0.5
            
            dist_to_wall_l_x = abs(robot_pos_world[0] - wall_l_center[0])
            dist_to_wall_l_y = abs(robot_pos_world[1] - wall_l_center[1])
            if dist_to_wall_l_x < (wall_l_thickness + ROBOT_HALF_LENGTH_X):
                if dist_to_wall_l_y < (wall_l_half_len + ROBOT_HALF_WIDTH_Y):
                    return True
            
            # 检查右墙
            wall_r_center = np.array([door_r_pos[0], door_r_pos[1]])
            wall_r_half_len = door_r_size[1] * 0.5
            wall_r_thickness = door_r_size[0] * 0.5
            
            dist_to_wall_r_x = abs(robot_pos_world[0] - wall_r_center[0])
            dist_to_wall_r_y = abs(robot_pos_world[1] - wall_r_center[1])
            if dist_to_wall_r_x < (wall_r_thickness + ROBOT_HALF_LENGTH_X):
                if dist_to_wall_r_y < (wall_r_half_len + ROBOT_HALF_WIDTH_Y):
                    return True
        
        return False
    
    def check_boundary_collision(self, env_id, robot_pos_local):
        """
        检测边界碰撞
        
        Args:
            env_id: 环境 ID
            robot_pos_local: (2,) 机器人局部位置 [x, y]
        
        Returns:
            bool: 是否碰撞
        """
        x, y = robot_pos_local[0], robot_pos_local[1]
        if abs(x) > (self.env_half - ROBOT_HALF_LENGTH_X) or abs(y) > (self.env_half - ROBOT_HALF_WIDTH_Y):
            return True
        return False
    
    # ========== 工具函数 ==========
    
    def _create_cube(self, stage, prim_path, size_xyz, center_xyz, visible=True, color=(0.6, 0.6, 0.6), collision=True):
        """创建立方体"""
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        
        prim = cube.GetPrim()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        s_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        
        t_op.Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])))
        s_op.Set(Gf.Vec3d(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))
        
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
        
        return prim, t_op, s_op
    
    def _create_cylinder(self, stage, prim_path, radius, height, center_xyz, visible=True, color=(0.5, 0.7, 0.9), collision=True):
        """创建圆柱体"""
        cyl = UsdGeom.Cylinder.Define(stage, prim_path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(float(height))
        
        prim = cyl.GetPrim()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])))
        
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
        
        return prim, t_op, cyl
    
    def _set_visibility(self, prim, visible):
        """设置可见性"""
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    
    def _hide_prim_below_ground(self, t_op, z=-10.0):
        """将 prim 隐藏到地下"""
        p = t_op.Get()
        t_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(z)))
