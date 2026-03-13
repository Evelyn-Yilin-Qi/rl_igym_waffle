"""
Heading Penalty Component
航向惩罚组件
"""
import numpy as np
from .base import BaseRewardComponent


class HeadingPenalty(BaseRewardComponent):
    """航向惩罚：根据航向角误差计算惩罚"""
    def compute(self, yaw_err, collision, **kwargs):
        """
        Args:
            yaw_err: (num_envs,) 航向角误差（Φ）
            collision: (num_envs,) 是否碰撞（bool）
        Returns:
            reward: (num_envs,) 航向惩罚
            info: dict
        """
        if not self.enabled:
            return np.zeros(len(yaw_err), dtype=np.float32), {}
        
        yaw_err = np.asarray(yaw_err, dtype=np.float32)
        collision = np.asarray(collision, dtype=bool)
        
        # 仅对非碰撞环境计算
        non_collision_mask = ~collision
        reward = np.zeros(len(yaw_err), dtype=np.float32)
        
        if np.any(non_collision_mask):
            params = self.config.get('params', {})
            phi_thresh = params.get('phi_thresh', 0)
            rh = params.get('rh', -2.8)
            rl = params.get('rl', -1)
            
            # 筛选非碰撞环境的航向误差
            yaw_err_non_collision = yaw_err[non_collision_mask]
            abs_err = np.abs(yaw_err_non_collision)
            # 仅当|Φ| > Φthresh时施加惩罚
            heading_mask = abs_err > phi_thresh
            heading_pen_non_collision = np.zeros_like(yaw_err_non_collision)
            heading_pen_non_collision[heading_mask] = (
                (rh * (abs_err[heading_mask] ** 2) + rl * abs_err[heading_mask]) * self.weight
            )
            # 赋值回总惩罚数组
            reward[non_collision_mask] = heading_pen_non_collision
        
        return reward, {"heading_pen": reward}
