"""
PPO Algorithm Implementation
PPO算法实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from .base import BaseRLAlgorithm


class PPO(BaseRLAlgorithm):
    """PPO算法核心类"""
    def __init__(self, policy, value, config, device=None):
        """
        Args:
            policy: 策略网络（BasePolicy）
            value: 价值网络（BaseValue）
            config: 算法配置字典，包含：
                - lr: 学习率
                - gamma: 折扣因子
                - lamda: GAE系数
                - clip_eps: 裁剪系数
                - k_epochs: 每次更新迭代次数
                - batch_size: 批次大小
                - entropy_coef: 熵系数（可选，默认0.01）
            device: 计算设备
        """
        super().__init__(policy, value, config, device)
        
        # 将网络移到指定设备
        self.policy = self.policy.to(self.device)
        self.value = self.value.to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=config.get('lr', 3e-4))
        self.critic_optimizer = optim.Adam(self.value.parameters(), lr=config.get('lr', 3e-4))
        
        # 超参数
        self.gamma = config.get('gamma', 0.99)
        self.lamda = config.get('lamda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.k_epochs = config.get('k_epochs', 3)
        self.batch_size = config.get('batch_size', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 经验缓存
        self.buffer = {
            "obs": [], "acts": [], "rews": [], 
            "next_obs": [], "dones": [], "log_probs": []
        }

    def select_action(self, obs):
        """根据观察选择动作（带探索）"""
        # 将观察数据移到指定设备
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean, log_std = self.policy(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        # 动作裁剪（限制在合理范围）
        action = torch.clamp(action, -1.0, 1.0)
        
        # 返回时转回CPU和numpy（因为后续计算用numpy）
        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().numpy()

    def store_transition(self, obs, act, rew, next_obs, done, log_prob):
        """存储经验"""
        self.buffer["obs"].append(obs)
        self.buffer["acts"].append(act)
        self.buffer["rews"].append(rew)
        self.buffer["next_obs"].append(next_obs)
        self.buffer["dones"].append(done)
        self.buffer["log_probs"].append(log_prob)

    def compute_gae(self, rewards, dones, values, next_values):
        """计算GAE（广义优势估计）"""
        advantages = []
        advantage = 0.0
        rewards /= 100.0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = td_error + self.gamma * self.lamda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        returns = np.array(advantages) + np.array(values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8) 
        return advantages, returns

    def update(self):
        """
        更新PPO策略
        Returns:
            dict: 包含训练loss信息的字典
                - actor_loss: 策略损失
                - critic_loss: 价值损失
                - entropy: 策略熵
                - approx_kl: 近似KL散度
        """
        # 转换为tensor并移到指定设备
        obs = torch.FloatTensor(np.array(self.buffer["obs"])).to(self.device)
        acts = torch.FloatTensor(np.array(self.buffer["acts"])).to(self.device)
        rews = np.array(self.buffer["rews"])
        next_obs = torch.FloatTensor(np.array(self.buffer["next_obs"])).to(self.device)
        dones = np.array(self.buffer["dones"])
        old_log_probs = torch.FloatTensor(np.array(self.buffer["log_probs"])).to(self.device)
        
        # 计算价值和下一步价值
        values = self.value(obs).detach().cpu().numpy().flatten()
        next_values = self.value(next_obs).detach().cpu().numpy().flatten()
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(rews, dones, values, next_values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 数值稳定性：裁剪 returns 和 advantages，防止过大值
        # 检查是否有 NaN 或 Inf
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            print(f"[WARNING] Invalid returns detected, replacing with zeros")
            returns = torch.zeros_like(returns)
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print(f"[WARNING] Invalid advantages detected, replacing with zeros")
            advantages = torch.zeros_like(advantages)
        
        # 裁剪到合理范围（防止过大奖励导致训练不稳定）
        returns = torch.clamp(returns, min=-1000.0, max=1000.0)
        advantages = torch.clamp(advantages, min=-100.0, max=100.0)
        
        # 用于累计loss统计
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_batches = 0
        
        # 多次迭代更新
        for _ in range(self.k_epochs):
            # 打乱数据
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            # 分批更新
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # 计算当前策略的log_prob
                mean, log_std = self.policy(obs[batch_idx])
                std = log_std.exp()
                dist = Normal(mean, std)
                current_log_probs = dist.log_prob(acts[batch_idx]).sum(dim=1)
                
                # 计算比率和近似KL散度
                ratio = torch.exp(current_log_probs - old_log_probs[batch_idx])
                approx_kl = (old_log_probs[batch_idx] - current_log_probs).mean()
                
                # 裁剪的优势损失
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch_idx]
                entropy = dist.entropy().sum(dim=1)
                # 策略损失 = 原损失 - 熵系数*熵（减号是因为要最大化熵，最小化损失）
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # 更新策略网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                
                # 检查梯度是否包含 NaN 或 Inf
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                if torch.isnan(actor_grad_norm) or torch.isinf(actor_grad_norm):
                    print(f"[WARNING] Invalid actor gradient norm: {actor_grad_norm}, skipping update")
                    self.actor_optimizer.zero_grad()
                else:
                    self.actor_optimizer.step()
                
                # 更新价值网络
                critic_loss = nn.MSELoss()(self.value(obs[batch_idx]).flatten(), returns[batch_idx])
                
                # 检查 critic loss 是否过大
                if critic_loss.item() > 1000.0:
                    print(f"[WARNING] Critic loss too large: {critic_loss.item():.2f}, clipping to 1000.0")
                    critic_loss = torch.clamp(critic_loss, max=1000.0)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                
                # 检查梯度是否包含 NaN 或 Inf
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
                if torch.isnan(critic_grad_norm) or torch.isinf(critic_grad_norm):
                    print(f"[WARNING] Invalid critic gradient norm: {critic_grad_norm}, skipping update")
                    self.critic_optimizer.zero_grad()
                else:
                    self.critic_optimizer.step()
                
                # NaN检测：检查更新后的参数
                for name, param in self.policy.named_parameters():
                    if torch.isnan(param).any():
                        raise ValueError(f"NaN detected in policy parameter '{name}' after update!")
                for name, param in self.value.named_parameters():
                    if torch.isnan(param).any():
                        raise ValueError(f"NaN detected in value parameter '{name}' after update!")
                
                # 累计loss统计
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl.item()
                num_batches += 1
        
        # 清空缓存
        for key in self.buffer.keys():
            self.buffer[key].clear()
        
        # 返回平均loss信息
        return {
            "actor_loss": total_actor_loss / num_batches if num_batches > 0 else 0.0,
            "critic_loss": total_critic_loss / num_batches if num_batches > 0 else 0.0,
            "entropy": total_entropy / num_batches if num_batches > 0 else 0.0,
            "approx_kl": total_approx_kl / num_batches if num_batches > 0 else 0.0,
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
