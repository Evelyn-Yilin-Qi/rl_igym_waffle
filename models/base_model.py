"""
Base Model Interface
所有 WaffleDrive 网络架构的基类
"""
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class BaseWaffleModel(GaussianMixin, DeterministicMixin, Model):
    """
    WaffleDrive 模型基类
    所有自定义网络架构都应继承此类
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std=True, min_log_std=-20, max_log_std=2)
        DeterministicMixin.__init__(self, clip_actions)
    
    def get_specification(self):
        """返回模型规范（如 RNN 状态）"""
        raise NotImplementedError("Subclasses must implement get_specification()")
    
    def act(self, inputs, role=""):
        """根据角色选择行为"""
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        else:
            return GaussianMixin.act(self, inputs, role)
    
    def compute(self, inputs, role):
        """前向传播"""
        raise NotImplementedError("Subclasses must implement compute()")
