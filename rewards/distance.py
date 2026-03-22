"""
Distance Reward Component
距离奖励组件
"""
import numpy as np
from .base import BaseRewardComponent


class DistanceReward(BaseRewardComponent):
    """距离奖励：距离越近奖励越高（负距离项）"""
    def compute(self, dist, **kwargs):
        if not self.enabled:
            return np.zeros(len(dist), dtype=np.float32), {}

        dist = np.asarray(dist, dtype=np.float32)
        scale = self.config.get("params", {}).get("scale", -1.0)
        reward = scale * dist * self.weight
        return reward, {"distance_reward": reward}
