"""
Simple FC Network Architecture
简单全连接网络架构（分支网络：LiDAR + 状态特征）
"""
import torch
import torch.nn as nn
from .base import BasePolicy, BaseValue


class SimpleFCPolicy(BasePolicy):
    """Simple FC策略网络（拆分输入分支：LiDAR+状态特征）"""
    def __init__(self, obs_dim=44, act_dim=2, lidar_dim=36, state_dim=8, hidden_dim=128, **kwargs):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        
        # LiDAR特征提取分支（36维输入）
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh()
        )
        
        # 状态特征分支（8维：UserIntent(2)+BaseVel(2)+ActionHistory(4)）
        # UserIntent 口径：user_intent_ego = [ux, uy]
        # - ux: 自车正前方为正（forward +X）
        # - uy: 自车左侧为正（left +Y）
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Tanh()
        )
        
        # 合并特征后输出动作分布
        self.fc_merge = nn.Linear(hidden_dim//2 + hidden_dim//2, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        # 动作标准差限制（避免方差过大/过小）
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        # 拆分输入：前36维是LiDAR，后8维是状态特征
        lidar_feat = x[:, :self.lidar_dim]    # (batch, 36)
        state_feat = x[:, self.lidar_dim:]    # (batch, 8)
        
        # 分支特征提取
        lidar_out = self.lidar_branch(lidar_feat)  # (batch, 64)
        state_out = self.state_branch(state_feat)  # (batch, 64)
        
        # 合并特征
        merge_feat = torch.cat([lidar_out, state_out], dim=1)  # (batch, 128)
        merge_feat = torch.tanh(self.fc_merge(merge_feat))     # (batch, 128)
        
        # 输出动作分布
        mean = self.mean_layer(merge_feat)
        log_std = self.log_std_layer(merge_feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class SimpleFCValue(BaseValue):
    """Simple FC价值网络（拆分输入分支：LiDAR+状态特征）"""
    def __init__(self, obs_dim=44, lidar_dim=36, state_dim=8, hidden_dim=128, **kwargs):
        super().__init__(obs_dim=obs_dim)
        
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        
        # LiDAR特征提取分支
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh()
        )
        
        # 状态特征分支
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Tanh()
        )
        
        # 合并特征后输出状态价值
        self.fc_merge = nn.Linear(hidden_dim//2 + hidden_dim//2, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 拆分输入
        lidar_feat = x[:, :self.lidar_dim]
        state_feat = x[:, self.lidar_dim:]
        
        # 分支特征提取
        lidar_out = self.lidar_branch(lidar_feat)
        state_out = self.state_branch(state_feat)
        
        # 合并特征
        merge_feat = torch.cat([lidar_out, state_out], dim=1)
        merge_feat = torch.tanh(self.fc_merge(merge_feat))
        
        # 输出状态价值
        value = self.value_layer(merge_feat)
        return value
