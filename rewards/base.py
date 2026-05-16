"""
Base Reward Component
奖励组件基类
"""
import numpy as np


class BaseRewardComponent:
    """奖励组件基类"""
    def __init__(self, config):
        """
        Args:
            config: 配置字典，包含组件的参数
        """
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)
    
    def compute(self, **kwargs):
        """
        计算奖励分量
        Args:
            **kwargs: 输入参数（根据具体组件而定）
        Returns:
            reward: (num_envs,) 奖励值
            info: dict 额外信息（用于TensorBoard）
        """
        raise NotImplementedError
