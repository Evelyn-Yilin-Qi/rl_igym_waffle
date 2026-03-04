"""
Scene Base Classes and Utility Functions
场景基类和工具函数
"""
import numpy as np


# 场景类型常量
SCENE_EMPTY = "empty"
SCENE_BOX = "box"
SCENE_CYLINDER = "cylinder"
SCENE_DOOR = "door"


def quat_wxyz_from_yaw(yaw):
    """
    从 yaw 角生成四元数 (w, x, y, z)
    
    Args:
        yaw: (N,) 或标量，yaw 角（弧度）
    
    Returns:
        q: (N, 4) 或 (4,)，四元数 [w, x, y, z]
    """
    if isinstance(yaw, (int, float)):
        yaw = np.array([yaw], dtype=np.float32)
    elif not isinstance(yaw, np.ndarray):
        yaw = np.array(yaw, dtype=np.float32)
    
    half = 0.5 * yaw
    q = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    q[:, 0] = np.cos(half)  # w
    q[:, 3] = np.sin(half)  # z
    return q


def wrap_to_pi(a):
    """
    将角度包装到 [-π, π] 范围
    
    Args:
        a: 角度（弧度）
    
    Returns:
        包装后的角度
    """
    return np.arctan2(np.sin(a), np.cos(a))


def yaw_from_quat_wxyz(q):
    """
    从四元数 (w, x, y, z) 提取 yaw 角
    
    Args:
        q: (N, 4) 或 (4,)，四元数 [w, x, y, z]
    
    Returns:
        yaw: (N,) 或标量，yaw 角（弧度）
    """
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float32)
    
    if q.ndim == 1:
        q = q.reshape(1, -1)
    
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    if yaw.shape[0] == 1:
        return yaw[0]
    return yaw


def sample_goal_offsets(n, r_min=1.0, r_max=1.5):
    """
    随机采样目标偏移（相对于环境原点）
    
    Args:
        n: 采样数量
        r_min: 最小半径 (m)
        r_max: 最大半径 (m)
    
    Returns:
        offsets: (n, 2) 目标偏移 [dx, dy]
    """
    ang = np.random.uniform(-np.pi, np.pi, size=(n,)).astype(np.float32)
    r = (r_min + (r_max - r_min) * np.random.rand(n)).astype(np.float32)
    dx = r * np.cos(ang)
    dy = r * np.sin(ang)
    return np.stack([dx, dy], axis=1)
