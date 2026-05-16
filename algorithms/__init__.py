"""
RL Algorithms Module
强化学习算法模块
"""
from .base import BaseRLAlgorithm
from .ppo import PPO

__all__ = [
    'BaseRLAlgorithm',
    'PPO',
]


def create_algorithm(algorithm_type, policy, value, config, device=None):
    """
    创建RL算法（工厂函数）
    Args:
        algorithm_type: 算法类型（如 "ppo"）
        policy: 策略网络
        value: 价值网络
        config: 算法配置字典
        device: 计算设备
    Returns:
        algorithm: BaseRLAlgorithm实例
    """
    if algorithm_type == "ppo":
        return PPO(policy=policy, value=value, config=config, device=device)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
