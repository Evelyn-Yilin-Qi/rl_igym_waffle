"""
LiDAR + sequence-backbone policies for 360-ray + low-dim state (e.g. intent + base_vel).

Architectures:
  - cnn_lstm_sft: 1D CNN on LiDAR -> concat state -> LSTM(seq=1) -> FC -> Gaussian head
  - cnn_gru_sft:  same with GRU instead of LSTM
  - fc_lstm_sft:  MLP on LiDAR -> concat state -> LSTM -> FC -> head

Input layout matches SimpleFCSFTPolicy: x[:, :lidar_dim], x[:, lidar_dim : lidar_dim + state_dim].
"""
import torch
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


class SimpleFCSFTPolicy(BasePolicy):
    """SFT策略网络（ReLU分支结构）"""

    def __init__(self, obs_dim=364, act_dim=2, lidar_dim=360, state_dim=4, hidden_dim=128, **kwargs):
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

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class _CNNEncoder360(nn.Module):
    """Two Conv1d stages + adaptive pool + linear to `out_dim`."""

    _FLAT = 32 * 16

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = int(out_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(self._FLAT, self.out_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, lidar_bchw):
        return self.encoder(lidar_bchw)


class CNNLSTMSFTPolicy(BasePolicy):
    def __init__(
            self,
            obs_dim=364,
            act_dim=2,
            lidar_dim=360,
            state_dim=4,
            cnn_vec_dim=512,
            lstm_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.cnn_body = _CNNEncoder360(out_dim=cnn_vec_dim)
        lstm_in = cnn_vec_dim + self.state_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)
        feat = self.head(h)
        mean = self.mean_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), self.log_std_min, self.log_std_max)
        mean = torch.clamp(mean, -10.0, 10.0)
        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class CNNLSTMSFTValue(BaseValue):
    def __init__(
            self,
            obs_dim=364,
            lidar_dim=360,
            state_dim=4,
            cnn_vec_dim=512,
            lstm_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.cnn_body = _CNNEncoder360(out_dim=cnn_vec_dim)
        lstm_in = cnn_vec_dim + self.state_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )
        self.value_layer = nn.Linear(head_hidden, 1)

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        feat = self.head(h)
        return self.value_layer(feat)


class CNNGRUSFTPolicy(BasePolicy):
    def __init__(
            self,
            obs_dim=364,
            act_dim=2,
            lidar_dim=360,
            state_dim=4,
            cnn_vec_dim=512,
            rnn_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.cnn_body = _CNNEncoder360(out_dim=cnn_vec_dim)
        lstm_in = cnn_vec_dim + self.state_dim
        self.gru = nn.GRU(
            input_size=lstm_in,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, head_hidden),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.gru(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)
        feat = self.head(h)
        mean = self.mean_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), self.log_std_min, self.log_std_max)
        mean = torch.clamp(mean, -10.0, 10.0)
        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class CNNGRUSFTValue(BaseValue):
    def __init__(
            self,
            obs_dim=364,
            lidar_dim=360,
            state_dim=4,
            cnn_vec_dim=512,
            rnn_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.cnn_body = _CNNEncoder360(out_dim=cnn_vec_dim)
        lstm_in = cnn_vec_dim + self.state_dim
        self.gru = nn.GRU(
            input_size=lstm_in,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, head_hidden),
            nn.Tanh(),
        )
        self.value_layer = nn.Linear(head_hidden, 1)

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.gru(fused)
        h = h.squeeze(1)
        feat = self.head(h)
        return self.value_layer(feat)


class FCLSTMSFTPolicy(BasePolicy):
    def __init__(
            self,
            obs_dim=364,
            act_dim=2,
            lidar_dim=360,
            state_dim=4,
            lidar_emb_dim=128,
            lstm_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.lidar_mlp = nn.Sequential(
            nn.Linear(self.lidar_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, lidar_emb_dim),
            nn.ReLU(inplace=False),
        )
        lstm_in = lidar_emb_dim + self.state_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.lidar_mlp(lidar)
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)
        feat = self.head(h)
        mean = self.mean_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), self.log_std_min, self.log_std_max)
        mean = torch.clamp(mean, -10.0, 10.0)
        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class FCLSTMSFTValue(BaseValue):
    def __init__(
            self,
            obs_dim=364,
            lidar_dim=360,
            state_dim=4,
            lidar_emb_dim=128,
            lstm_hidden=128,
            head_hidden=128,
            **kwargs,
    ):
        super().__init__(obs_dim=obs_dim)
        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)
        self.lidar_mlp = nn.Sequential(
            nn.Linear(self.lidar_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, lidar_emb_dim),
            nn.ReLU(inplace=False),
        )
        lstm_in = lidar_emb_dim + self.state_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )
        self.value_layer = nn.Linear(head_hidden, 1)

    def forward(self, x):
        lidar = x[:, : self.lidar_dim]
        state = x[:, self.lidar_dim: self.lidar_dim + self.state_dim]
        z = self.lidar_mlp(lidar)
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        feat = self.head(h)
        return self.value_layer(feat)


if __name__ == "__main__":
    import numpy as np

    a = SimpleFCSFTPolicy()
    a.load_state_dict(torch.load("ppo_mixed_simple_fc_sft_epoch_020.pth", map_location="cpu")['policy_state_dict'])
    b = CNNLSTMSFTPolicy()
    b.load_state_dict(torch.load("ppo_mixed_cnn_lstm_sft_nodoor_epoch_020.pth", map_location="cpu")['policy_state_dict'])
    c = CNNLSTMSFTPolicy()
    c.load_state_dict(torch.load("ppo_mixed_cnn_lstm_sft_epoch_020.pth", map_location="cpu")['policy_state_dict'])
    d = CNNGRUSFTPolicy()
    d.load_state_dict(torch.load("ppo_mixed_cnn_gru_sft_epoch_020.pth", map_location="cpu")['policy_state_dict'])
    e = FCLSTMSFTPolicy()
    e.load_state_dict(torch.load("ppo_mixed_fc_lstm_sft_epoch_020.pth", map_location="cpu")['policy_state_dict'])

    model = [a, b, c, d, e]
    #lidar 360
    lidar_obs = np.ones(360, dtype=np.float32)*3
    # 状态特征：UserIntent(2)+BaseVel(2)
    user_obs = np.ones(2, dtype=np.float32)
    base_v_obs = np.zeros(2, dtype=np.float32)
    # 合并为完整obs（364维）
    test_obs = np.concatenate([lidar_obs, user_obs, base_v_obs], axis=0)
    # 转换为tensor并增加batch维度（模型输入需要batch维度）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0).to(device)

    for m in model:
        m = m.to(device)
        with torch.no_grad():
            mean, log_std = m(test_obs_tensor)
            std = log_std.exp()

            # 输出动作均值（归一化到[-1,1]）
            action_mean = mean.cpu().numpy().flatten()
            # 反归一化到实际速度范围（和训练时一致）
            max_v = 1.0  # 替换成你训练时的max_v
            max_w = 1.0  # 替换成你训练时的max_w
            v_cmd = action_mean[0] * max_v
            w_cmd = action_mean[1] * max_w

            # 速度裁剪（和训练时一致）
            # v_cmd = np.clip(v_cmd, 0.0, max_v)
            # w_cmd = np.clip(w_cmd, -max_w, max_w)
            print(v_cmd, w_cmd)
