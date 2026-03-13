"""
Base Network Architecture
核心网络架构基类
"""
import torch.nn as nn


class BasePolicy(nn.Module):
    """策略网络基类 - 统一接口：输入44维，输出2维"""
    def __init__(self, obs_dim=44, act_dim=2, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def forward(self, obs):
        """
        前向传播
        Args:
            obs: (batch, obs_dim) 观察向量
        Returns:
            mean: (batch, act_dim) 动作均值
            log_std: (batch, act_dim) 动作对数标准差
        """
        raise NotImplementedError


class BaseValue(nn.Module):
    """价值网络基类"""
    def __init__(self, obs_dim=44, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
    
    def forward(self, obs):
        """
        前向传播
        Args:
            obs: (batch, obs_dim) 观察向量
        Returns:
            value: (batch, 1) 状态价值
        """
        raise NotImplementedError
