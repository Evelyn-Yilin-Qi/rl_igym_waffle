"""
PPO Agent Wrapper
封装 skrl PPO agent，简化使用
"""
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory


class PPOAgentBase:
    """
    PPO Agent 基础封装类
    简化 skrl PPO 的初始化和使用
    """
    def __init__(
        self,
        models: Dict[str, Any],
        memory: RandomMemory,
        observation_space: Any,
        action_space: Any,
        device: str,
        cfg: Optional[DictConfig] = None,
        **ppo_kwargs
    ):
        """
        初始化 PPO Agent
        
        Args:
            models: 模型字典 {"policy": model, "value": model}
            memory: 经验回放缓冲区
            observation_space: 观察空间
            action_space: 动作空间
            device: 设备
            cfg: 配置对象（可选）
            **ppo_kwargs: 额外的 PPO 配置参数
        """
        # 合并配置
        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        if cfg is not None:
            # 从配置对象提取 agent 配置（支持 cfg.train.agent 或 cfg.agent）
            if hasattr(cfg, "train") and hasattr(cfg.train, "agent"):
                agent_cfg = dict(cfg.train.agent)
                ppo_cfg.update(agent_cfg)
            elif hasattr(cfg, "agent"):
                agent_cfg = dict(cfg.agent)
                ppo_cfg.update(agent_cfg)
            elif isinstance(cfg, dict):
                ppo_cfg.update(cfg)
        
        # 更新额外参数
        ppo_cfg.update(ppo_kwargs)
        
        # 创建 PPO agent
        self.agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
    
    def act(self, states, timestep, timesteps):
        """选择动作"""
        return self.agent.act(states, timestep, timesteps)
    
    def record_transition(self, *args, **kwargs):
        """记录转移"""
        return self.agent.record_transition(*args, **kwargs)
    
    def update(self, timestep, timesteps):
        """更新策略"""
        return self.agent.update(timestep, timesteps)
    
    def save(self, path):
        """保存模型"""
        return self.agent.save(path)
    
    def load(self, path):
        """加载模型"""
        return self.agent.load(path)
