"""
Essay Base Network Architecture
论文基础架构：1D Conv + LSTM
"""
import torch
import torch.nn as nn
from .base import BasePolicy, BaseValue


class EssayBasePolicy(BasePolicy):
    """Essay Base策略网络（1D Conv + LSTM架构）"""
    def __init__(self, obs_dim=44, act_dim=2, lidar_dim=36, state_dim=8, **kwargs):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        
        # LiDAR分支：1D Conv处理
        # 1D Conv 1: kernel=5, out_channels=8, stride=2
        self.conv1d_1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=0
        )
        
        # 1D Conv 2: kernel=3, out_channels=4, stride=2
        self.conv1d_2 = nn.Conv1d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
        # 计算Conv后的特征维度
        # Conv1: (36-5)/2+1 = 16
        # Conv2: (16-3)/2+1 = 7
        # 所以conv_output_size = 7 * 4 = 28
        conv_output_size = 7 * 4
        
        # FC0: 将Conv输出映射到64维
        self.fc0 = nn.Linear(conv_output_size, 64)
        
        # LSTM: 融合LiDAR特征(64) + 状态特征(8) = 72维输入
        self.lstm = nn.LSTM(
            input_size=64 + state_dim,  # FC0输出(64) + 状态特征(8)
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # FC1: LSTM输出后的全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # 输出层
        self.mean_layer = nn.Linear(64, act_dim)
        self.log_std_layer = nn.Linear(64, act_dim)
        
        # 动作标准差限制
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        # 拆分输入：前36维是LiDAR，后8维是状态特征
        lidar_feat = x[:, :self.lidar_dim]    # (batch, 36)
        state_feat = x[:, self.lidar_dim:]    # (batch, 8)
        
        # LiDAR分支：1D Conv处理
        # Reshape: (batch, 36) -> (batch, 1, 36)
        lidar_reshaped = lidar_feat.unsqueeze(1)  # (batch, 1, 36)
        
        # 1D Conv 1
        conv1_out = torch.relu(self.conv1d_1(lidar_reshaped))  # (batch, 8, 16)
        
        # 1D Conv 2
        conv2_out = torch.relu(self.conv1d_2(conv1_out))  # (batch, 4, 7)
        
        # Flatten
        conv_flat = conv2_out.view(conv2_out.size(0), -1)  # (batch, 28)
        
        # FC0
        fc0_out = torch.tanh(self.fc0(conv_flat))  # (batch, 64)
        
        # 融合：FC0输出 + 状态特征
        fused_feat = torch.cat([fc0_out, state_feat], dim=1)  # (batch, 72)
        
        # Reshape为序列输入LSTM: (batch, 72) -> (batch, 1, 72)
        lstm_input = fused_feat.unsqueeze(1)  # (batch, 1, 72)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)  # (batch, 1, 64)
        lstm_out = lstm_out.squeeze(1)  # (batch, 64)
        
        # 数值稳定性：裁剪LSTM输出，防止过大值
        lstm_out = torch.clamp(lstm_out, min=-10.0, max=10.0)
        
        # FC1
        fc1_out = self.fc1(lstm_out)  # (batch, 64)
        
        # 数值稳定性：裁剪FC1输出
        fc1_out = torch.clamp(fc1_out, min=-10.0, max=10.0)
        
        # 输出动作分布
        mean = self.mean_layer(fc1_out)
        log_std = self.log_std_layer(fc1_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # 数值稳定性：裁剪mean输出
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        
        return mean, log_std


class EssayBaseValue(BaseValue):
    """Essay Base价值网络（1D Conv + LSTM架构）"""
    def __init__(self, obs_dim=44, lidar_dim=36, state_dim=8, **kwargs):
        super().__init__(obs_dim=obs_dim)
        
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        
        # LiDAR分支：1D Conv处理（与Policy共享结构）
        # 1D Conv 1: kernel=5, out_channels=8, stride=2
        self.conv1d_1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=0
        )
        
        # 1D Conv 2: kernel=3, out_channels=4, stride=2
        self.conv1d_2 = nn.Conv1d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
        # 计算Conv后的特征维度
        conv_output_size = 7 * 4
        
        # FC0: 将Conv输出映射到64维
        self.fc0 = nn.Linear(conv_output_size, 64)
        
        # LSTM: 融合LiDAR特征(64) + 状态特征(8) = 72维输入
        self.lstm = nn.LSTM(
            input_size=64 + state_dim,  # FC0输出(64) + 状态特征(8)
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # FC1: LSTM输出后的全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # 输出层
        self.value_layer = nn.Linear(64, 1)

    def forward(self, x):
        # 拆分输入：前36维是LiDAR，后8维是状态特征
        lidar_feat = x[:, :self.lidar_dim]    # (batch, 36)
        state_feat = x[:, self.lidar_dim:]    # (batch, 8)
        
        # LiDAR分支：1D Conv处理
        # Reshape: (batch, 36) -> (batch, 1, 36)
        lidar_reshaped = lidar_feat.unsqueeze(1)  # (batch, 1, 36)
        
        # 1D Conv 1
        conv1_out = torch.relu(self.conv1d_1(lidar_reshaped))  # (batch, 8, 16)
        
        # 1D Conv 2
        conv2_out = torch.relu(self.conv1d_2(conv1_out))  # (batch, 4, 7)
        
        # Flatten
        conv_flat = conv2_out.view(conv2_out.size(0), -1)  # (batch, 28)
        
        # FC0
        fc0_out = torch.tanh(self.fc0(conv_flat))  # (batch, 64)
        
        # 融合：FC0输出 + 状态特征
        fused_feat = torch.cat([fc0_out, state_feat], dim=1)  # (batch, 72)
        
        # Reshape为序列输入LSTM: (batch, 72) -> (batch, 1, 72)
        lstm_input = fused_feat.unsqueeze(1)  # (batch, 1, 72)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)  # (batch, 1, 64)
        lstm_out = lstm_out.squeeze(1)  # (batch, 64)
        
        # FC1
        fc1_out = self.fc1(lstm_out)  # (batch, 64)
        
        # 输出状态价值
        value = self.value_layer(fc1_out)
        return value
