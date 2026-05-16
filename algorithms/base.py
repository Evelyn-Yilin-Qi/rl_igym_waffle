"""
Base RL Algorithm
RL算法基类
"""
import torch


class BaseRLAlgorithm:
    """RL算法基类 - 统一接口"""
    def __init__(self, policy, value, config, device=None):
        """
        Args:
            policy: 策略网络（BasePolicy）
            value: 价值网络（BaseValue）
            config: 算法配置字典
            device: 计算设备（torch.device）
        """
        self.policy = policy
        self.value = value
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_action(self, obs):
        """
        选择动作
        Args:
            obs: 观察向量
        Returns:
            action: 动作
            log_prob: 动作的对数概率
        """
        raise NotImplementedError
    
    def store_transition(self, obs, act, rew, next_obs, done, log_prob):
        """存储经验"""
        raise NotImplementedError
    
    def update(self):
        """更新策略"""
        raise NotImplementedError
    
    def save(self, path):
        """保存模型"""
        raise NotImplementedError
    
    def load(self, path):
        """加载模型"""
        raise NotImplementedError
