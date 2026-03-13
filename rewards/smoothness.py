"""
Smoothness Penalty Component
动作平滑惩罚组件
"""
import numpy as np
from .base import BaseRewardComponent


class SmoothnessPenalty(BaseRewardComponent):
    """动作平滑惩罚：一阶和二阶平滑惩罚"""
    def compute(self, current_act, prev_act, prev_prev_act, collision, **kwargs):
        """
        Args:
            current_act: (num_envs, 2) 当前归一化动作 [v, w]
            prev_act: (num_envs, 2) 上一归一化动作
            prev_prev_act: (num_envs, 2) 前两归一化动作
            collision: (num_envs,) 是否碰撞（bool）
        Returns:
            reward: (num_envs,) 平滑惩罚
            info: dict
        """
        if not self.enabled:
            return np.zeros(len(collision), dtype=np.float32), {}
        
        current_act = np.asarray(current_act, dtype=np.float32)
        prev_act = np.asarray(prev_act, dtype=np.float32)
        prev_prev_act = np.asarray(prev_prev_act, dtype=np.float32)
        collision = np.asarray(collision, dtype=bool)
        
        # 仅对非碰撞环境计算
        non_collision_mask = ~collision
        reward = np.zeros(len(collision), dtype=np.float32)
        
        if np.any(non_collision_mask):
            params = self.config.get('params', {})
            ras1 = params.get('ras1', -0.02)  # 一阶平滑惩罚系数
            ras2 = params.get('ras2', -0.02)  # 二阶平滑惩罚系数
            
            # 筛选非碰撞环境的动作
            current_act_non_collision = current_act[non_collision_mask]
            prev_act_non_collision = prev_act[non_collision_mask]
            prev_prev_act_non_collision = prev_prev_act[non_collision_mask]
            
            # 一阶平滑惩罚：|a_t - a_t-1|²（按动作维度求和）
            first_order = np.sum((current_act_non_collision - prev_act_non_collision) ** 2, axis=1)
            # 二阶平滑惩罚：|a_t - 2*a_t-1 + a_t-2|²
            second_order = np.sum((current_act_non_collision - 2*prev_act_non_collision + prev_prev_act_non_collision) ** 2, axis=1)
            # 总平滑惩罚
            smooth_pen_non_collision = (ras1 * first_order + ras2 * second_order) * self.weight
            # 赋值回总惩罚数组
            reward[non_collision_mask] = smooth_pen_non_collision
        
        return reward, {"smooth_pen": reward}
