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

from .base import BasePolicy, BaseValue


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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
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
        state = x[:, self.lidar_dim : self.lidar_dim + self.state_dim]
        z = self.lidar_mlp(lidar)
        fused = torch.cat([z, state], dim=1).unsqueeze(1)
        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        feat = self.head(h)
        return self.value_layer(feat)
