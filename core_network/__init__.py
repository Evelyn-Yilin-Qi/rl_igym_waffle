"""
Core Network Module
核心网络架构模块
"""
from .base import BasePolicy, BaseValue
from .simple_fc import SimpleFCPolicy, SimpleFCValue
from .simple_fc_sft import SimpleFCSFTPolicy, SimpleFCSFTValue
from .essay_base import EssayBasePolicy, EssayBaseValue

__all__ = [
    'BasePolicy', 'BaseValue',
    'SimpleFCPolicy', 'SimpleFCValue',
    'SimpleFCSFTPolicy', 'SimpleFCSFTValue',
    'EssayBasePolicy', 'EssayBaseValue',
]


def create_policy(network_type, **kwargs):
    """
    创建策略网络（工厂函数）
    Args:
        network_type: 网络类型（如 "simple_fc", "essay_base"）
        **kwargs: 网络参数
    Returns:
        policy: BasePolicy实例
    """
    if network_type == "simple_fc":
        return SimpleFCPolicy(**kwargs)
    elif network_type == "simple_fc_sft":
        return SimpleFCSFTPolicy(**kwargs)
    elif network_type == "essay_base":
        return EssayBasePolicy(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def create_value(network_type, **kwargs):
    """
    创建价值网络（工厂函数）
    Args:
        network_type: 网络类型（如 "simple_fc", "essay_base"）
        **kwargs: 网络参数
    Returns:
        value: BaseValue实例
    """
    if network_type == "simple_fc":
        return SimpleFCValue(**kwargs)
    elif network_type == "simple_fc_sft":
        return SimpleFCSFTValue(**kwargs)
    elif network_type == "essay_base":
        return EssayBaseValue(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
