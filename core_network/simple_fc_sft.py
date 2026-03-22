"""
Simple FC SFT Network Architecture
SFT 版本简单全连接网络（LiDAR + 状态分支，适配44维观察）
"""
import torch
import torch.nn as nn
from .base import BasePolicy, BaseValue


class SimpleFCSFTPolicy(BasePolicy):
    """SFT策略网络（ReLU分支结构）"""
    def __init__(self, obs_dim=44, act_dim=2, lidar_dim=36, state_dim=8, hidden_dim=128, **kwargs):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim

        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # UserIntent 口径：user_intent_ego = [ux, uy]
        # - ux: 自车正前方为正（forward +X）
        # - uy: 自车左侧为正（left +Y）
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_merge = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar_feat = x[:, :self.lidar_dim]
        state_feat = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]
        lidar_out = self.lidar_branch(lidar_feat)
        state_out = self.state_branch(state_feat)
        merge_feat = torch.cat([lidar_out, state_out], dim=1)
        merge_feat = torch.tanh(self.fc_merge(merge_feat))
        mean = self.mean_layer(merge_feat)
        log_std = self.log_std_layer(merge_feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class SimpleFCSFTValue(BaseValue):
    """SFT价值网络（ReLU分支结构）"""
    def __init__(self, obs_dim=44, lidar_dim=36, state_dim=8, hidden_dim=128, **kwargs):
        super().__init__(obs_dim=obs_dim)
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim

        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_merge = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lidar_feat = x[:, :self.lidar_dim]
        state_feat = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]
        lidar_out = self.lidar_branch(lidar_feat)
        state_out = self.state_branch(state_feat)
        merge_feat = torch.cat([lidar_out, state_out], dim=1)
        merge_feat = torch.tanh(self.fc_merge(merge_feat))
        return self.value_layer(merge_feat)
