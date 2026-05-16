"""
Reward Composer
奖励组合器：统一管理和组合所有奖励组件
"""
import numpy as np
from .obstacle import ObstaclePenalty
from .heading import HeadingPenalty
from .smoothness import SmoothnessPenalty
from .static import StaticPenalty
from .centrifugal import CentrifugalPenalty
from .distance import DistanceReward
from .goal import GoalReward


# 组件名称到类的映射
COMPONENT_CLASSES = {
    "obstacle": ObstaclePenalty,
    "heading": HeadingPenalty,
    "smoothness": SmoothnessPenalty,
    "static": StaticPenalty,
    "centrifugal": CentrifugalPenalty,
    "distance_reward": DistanceReward,
    "goal_reward": GoalReward,
}


class RewardComposer:
    """奖励组合器"""
    def __init__(self, config):
        """
        Args:
            config: 配置字典，包含 components 列表
                components: [
                    {
                        "name": "distance",
                        "enabled": true,
                        "weight": 1.0,
                        "params": {...}
                    },
                    ...
                ]
        """
        self.config = config
        self.components = []
        
        # 创建所有启用的组件
        for comp_config in config.get('components', []):
            comp_name = comp_config.get('name')
            if comp_name in COMPONENT_CLASSES:
                component = COMPONENT_CLASSES[comp_name](comp_config)
                self.components.append(component)
            else:
                print(f"Warning: Unknown reward component '{comp_name}', skipping.")
    
    def compute(self, **kwargs):
        """
        计算总奖励（组合所有启用的组件）
        Args:
            **kwargs: 输入参数（传递给各个组件）
        Returns:
            total_reward: (num_envs,) 总奖励
            reward_info: dict 所有组件的奖励信息（用于TensorBoard）
        """
        # 获取环境数量（从任意输入参数推断）
        num_envs = None
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                num_envs = len(value)
                break
        
        if num_envs is None:
            raise ValueError("Cannot determine number of environments from input arguments")
        
        # 初始化总奖励（碰撞时奖励为0）
        collision = kwargs.get('collision', np.zeros(num_envs, dtype=bool))
        total_reward = np.zeros(num_envs, dtype=np.float32)
        reward_info = {}
        
        # 非碰撞环境掩码
        non_collision_mask = ~np.asarray(collision, dtype=bool)
        
        # 计算各个组件的奖励
        # 注意：
        # - obstacle组件：碰撞时应用rcol惩罚，非碰撞时应用距离惩罚
        # - static, heading, smoothness：只对非碰撞环境计算
        for component in self.components:
            if component.enabled:
                comp_reward, comp_info = component.compute(**kwargs)
                # obstacle组件的碰撞惩罚需要应用到所有环境（包括碰撞环境）
                # 检查组件类型（通过类名判断）
                if isinstance(component, ObstaclePenalty):
                    # obstacle组件自己处理碰撞和非碰撞的奖励分配
                    total_reward += comp_reward
                else:
                    # 其他组件只累加到非碰撞环境
                    total_reward[non_collision_mask] += comp_reward[non_collision_mask]
                # 合并信息
                reward_info.update(comp_info)
        
        return total_reward, reward_info
