"""
Obstacle Penalty Component
障碍物惩罚组件
"""
import numpy as np
from .base import BaseRewardComponent


class ObstaclePenalty(BaseRewardComponent):
    """障碍物惩罚：根据最近障碍物距离计算惩罚"""
    def compute(self, min_dist, collision, **kwargs):
        """
        Args:
            min_dist: (num_envs,) 最近障碍物距离
            collision: (num_envs,) 是否碰撞（bool）
        Returns:
            reward: (num_envs,) 障碍物惩罚
            info: dict
        """
        if not self.enabled:
            return np.zeros(len(min_dist), dtype=np.float32), {}
        
        min_dist = np.asarray(min_dist, dtype=np.float32)
        collision = np.asarray(collision, dtype=bool)
        
        params = self.config.get('params', {})
        dcol = params.get('dcol', 0.2)
        dcrit = params.get('dcrit', 0.5)
        rc = params.get('rc', -1.0)
        rcrit = params.get('rcrit', -1.0)
        rcol = params.get('rcol', -100.0)  # 碰撞惩罚（论文表格：-100）
        
        non_collision_mask = ~collision
        reward = np.zeros(len(min_dist), dtype=np.float32)
        
        # 碰撞时直接应用 rcol 惩罚（论文表格：-100）
        reward[collision] = rcol * self.weight
        
        # 非碰撞环境：根据距离计算惩罚
        if np.any(non_collision_mask):
            # 筛选非碰撞环境的最近距离
            min_dist_non_collision = min_dist[non_collision_mask]
            # 计算dcol < d < dcrit的环境掩码
            danger_mask = (min_dist_non_collision > dcol) & (min_dist_non_collision < dcrit)
            # 计算障碍物惩罚
            obstacle_pen_non_collision = np.zeros_like(min_dist_non_collision)
            obstacle_pen_non_collision[danger_mask] = (rc + rcrit * (dcrit - min_dist_non_collision[danger_mask])) * self.weight
            # 赋值回总惩罚数组
            reward[non_collision_mask] = obstacle_pen_non_collision
        
        return reward, {"obstacle_pen": reward, "collision_pen": reward[collision] if np.any(collision) else np.array([])}
