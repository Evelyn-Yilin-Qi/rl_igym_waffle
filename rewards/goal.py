"""
Goal Reward Component
到达目标/超时奖励组件
"""
import numpy as np
from .base import BaseRewardComponent


class GoalReward(BaseRewardComponent):
    """目标奖励：到达奖励 + 超时惩罚"""
    def compute(self, reached, timeout, dist, **kwargs):
        if not self.enabled:
            return np.zeros(len(reached), dtype=np.float32), {}

        reached = np.asarray(reached, dtype=bool)
        timeout = np.asarray(timeout, dtype=bool)
        dist = np.asarray(dist, dtype=np.float32)
        n = len(reached)
        reward = np.zeros(n, dtype=np.float32)

        params = self.config.get("params", {})
        reached_bonus = params.get("reached_bonus", 10.0)
        timeout_base = params.get("timeout_base", -8.0)
        timeout_dist_scale = params.get("timeout_dist_scale", -1.0)

        reward[reached] = reached_bonus * self.weight
        reward[timeout] = (timeout_base + timeout_dist_scale * dist[timeout]) * self.weight
        return reward, {"goal_reward": reward}
