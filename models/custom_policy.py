"""
WaffleDrive 专属自定义网络架构 (对接 skrl)
100% 纯 PyTorch 实现：共享 1D CNN + LSTM，独立分叉 Actor/Critic
"""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

class WaffleSharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        # 初始化 skrl 模型基类
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std=True, min_log_std=-20, max_log_std=2)
        DeterministicMixin.__init__(self, clip_actions)

        # ==========================================================
        # 1. 共享特征提取层 (Shared Feature Extractor)
        # ==========================================================
        # 1D CNN 处理 LiDAR: 36维 -> conv1 -> 17维 -> conv2 -> 9维
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 32 channels * 9 features = 288
        self.fc0 = nn.Sequential(nn.Linear(288, 64), nn.ReLU())
        
        # 拼接 8 维状态 (User Input[2] + Base Vel[2] + 2 Last Actions[4])
        self.shared_fc_out = nn.Sequential(nn.Linear(64 + 8, 128), nn.ReLU())

        # ==========================================================
        # 2. 共享记忆层 (Shared LSTM)
        # ==========================================================
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)

        # ==========================================================
        # 3. 独立分叉输出层 (Decoupled Actor & Critic)
        # ==========================================================
        # Actor 专属 MLP
        self.actor_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.actor_mean = nn.Linear(64, self.num_actions)
        self.actor_log_std = nn.Parameter(torch.zeros(self.num_actions))

        # Critic 专属 MLP
        self.critic_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.critic_value = nn.Linear(64, 1)

    def get_specification(self):
        # 向 skrl 注册我们的 LSTM 结构：一层，隐藏层维度 256 (h_0, c_0)
        return {"rnn": {"sequence_length": 1, "sizes": [(1, 256), (1, 256)]}}

    # ==========================================================
    # 【架构级修复】：显式路由 act 方法，打破多重继承导致的误解析
    # ==========================================================
    def act(self, inputs, role=""):
        if role == "policy":
            # Actor 需要概率分布探索，严格调用 GaussianMixin
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            # Critic 需要确定性评分，严格调用 DeterministicMixin
            return DeterministicMixin.act(self, inputs, role)
        else:
            return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        """100% 运行在 GPU 上的前向传播管线"""
        obs = inputs["states"] 
        
        # 1. 空间与状态特征提取
        lidar_data = obs[:, :36].unsqueeze(1) # shape: (N, 1, 36)
        state_data = obs[:, 36:]              # shape: (N, 8)

        cnn_features = self.cnn(lidar_data)
        fc0_features = self.fc0(cnn_features)
        
        combined_features = torch.cat((fc0_features, state_data), dim=1)
        shared_features = self.shared_fc_out(combined_features)

        # 2. 时序 LSTM 计算
        rnn_states = inputs.get("rnn", [torch.zeros(1, obs.size(0), 256, device=self.device),
                                        torch.zeros(1, obs.size(0), 256, device=self.device)])
        h_0, c_0 = rnn_states[0], rnn_states[1]

        lstm_out, (h_n, c_n) = self.lstm(shared_features.unsqueeze(1), (h_0, c_0))
        lstm_out = lstm_out.squeeze(1)

        # ==========================================================
        # 3. 角色分流输出 (严格匹配对应 Mixin 期望的返回值数量)
        # ==========================================================
        if role == "policy":
            x = self.actor_mlp(lstm_out)
            # GaussianMixin 期望 3 个返回值：(均值, 对数方差, 附加状态)
            return self.actor_mean(x), self.actor_log_std, {"rnn": [h_n, c_n]}
            
        elif role == "value":
            x = self.critic_mlp(lstm_out)
            # DeterministicMixin 期望 2 个返回值：(确定性数值, 附加状态)
            return self.critic_value(x), {"rnn": [h_n, c_n]}
            
        else:
            x = self.actor_mlp(lstm_out)
            return self.actor_mean(x), self.actor_log_std, {"rnn": [h_n, c_n]}