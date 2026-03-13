"""
Static Penalty Component
静止惩罚组件
"""
import numpy as np
from .base import BaseRewardComponent


class StaticPenalty(BaseRewardComponent):
    """静止惩罚：当线速度和角速度都很小时施加惩罚"""
    def compute(self, v_cmd, w_cmd, **kwargs):
        """
        Args:
            v_cmd: (num_envs,) 线速度指令
            w_cmd: (num_envs,) 角速度指令
        Returns:
            reward: (num_envs,) 静止惩罚
            info: dict
        """
        if not self.enabled:
            return np.zeros(len(v_cmd), dtype=np.float32), {}
        
        v_cmd = np.asarray(v_cmd, dtype=np.float32)
        w_cmd = np.asarray(w_cmd, dtype=np.float32)
        
        v_thresh = self.config.get('params', {}).get('v_thresh', 0.1)
        w_thresh = self.config.get('params', {}).get('w_thresh', 0.2)
        penalty_value = self.config.get('params', {}).get('penalty_value', -1.0)
        
        static_mask = (np.abs(v_cmd) < v_thresh) & (np.abs(w_cmd) < w_thresh)
        reward = np.zeros(len(v_cmd), dtype=np.float32)
        reward[static_mask] = penalty_value * self.weight
        
        return reward, {"static_pen": reward}
