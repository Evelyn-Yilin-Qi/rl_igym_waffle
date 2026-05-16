      
import torch
import numpy as np
import os
import torch.nn as nn

# ==================== 1. 复用原代码的网络定义（必须保留） ====================
class Actor(nn.Module):
    """PPO策略网络（拆分输入分支：LiDAR+状态特征）"""

    def __init__(self, lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128, share_bb=None):
        super(Actor, self).__init__()

        self.share_bb = share_bb  # 可选：共享基底网络（如果需要共享权重）
        # LiDAR特征提取分支（36维输入）
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 状态特征分支（6维：UserIntent(2)+BaseVel(2)+ActionHistory(2)）
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )

        # 合并特征后输出动作分布
        self.fc_merge = nn.Sequential(
            nn.Tanh(),  # 线性层前的tanh
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim)
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        # 动作标准差限制（避免方差过大/过小）
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        # 拆分输入：前36维是LiDAR，后6维是状态特征
        lidar_feat = x[:, :36]  # (batch, 36)
        state_feat = x[:, 36:]  # (batch, 6)

        # 分支特征提取
        lidar_out = self.lidar_branch(lidar_feat)  # (batch, 32)
        state_out = self.state_branch(state_feat)  # (batch, 32)

        # 合并特征
        merge_feat = torch.cat([lidar_out, state_out], dim=1)  # (batch, 64)
        merge_feat = torch.tanh(self.fc_merge(merge_feat))  # (batch, 64)

        # 输出动作分布
        mean = self.mean_layer(merge_feat)
        log_std = self.log_std_layer(merge_feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict_mean(self, x):
        """仅预测动作均值（用于监督学习）"""
        mean, _ = self.forward(x)
        return mean


# ==================== 2. 极简加载+推理demo ====================
def load_ppo_ckpt_and_infer(ckpt_path):
    """
    加载PPO模型ckpt并运行推理
    :param ckpt_path: 你的ckpt文件路径，比如 "ckpts/ppo_ckpt_step_10000.pth"
    """
    # 1. 设备配置（和训练时保持一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 初始化网络（和训练时的参数一致）
    actor = Actor(lidar_dim=36, state_dim=4, act_dim=2, hidden_dim=128).to(device)

    # 3. 加载ckpt文件
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt文件不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)  # map_location自动适配设备
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"成功加载ckpt: {ckpt_path}")
    print(f"该ckpt保存时的训练步数: {checkpoint['step']}")

    # 4. 构建测试输入（模拟obs：36维LiDAR + 6维状态特征）
    # LiDAR部分：全1表示无障碍物，距离最大
    lidar_obs = np.ones(36, dtype=np.float32)
    # 状态特征：UserIntent(2)+BaseVel(2)
    state_obs = np.zeros(4, dtype=np.float32)
    # 合并为完整obs（40维）
    test_obs = np.concatenate([lidar_obs, state_obs], axis=0)
    # 转换为tensor并增加batch维度（模型输入需要batch维度）
    test_obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0).to(device)

    # 5. 推理（关闭梯度计算，提升速度）
    actor.eval()  # 切换到评估模式
    with torch.no_grad():
        mean, log_std = actor(test_obs_tensor)
        std = log_std.exp()

        # 输出动作均值（归一化到[-1,1]）
        action_mean = mean.cpu().numpy().flatten()
        # 反归一化到实际速度范围（和训练时一致）
        max_v = 1.0  # 替换成你训练时的max_v
        max_w = 1.0  # 替换成你训练时的max_w
        v_cmd = action_mean[0] * max_v
        w_cmd = action_mean[1] * max_w

        # 速度裁剪（和训练时一致）
        v_cmd = np.clip(v_cmd, 0.0, max_v)
        w_cmd = np.clip(w_cmd, -max_w, max_w)

    # 6. 打印输出结果
    print("\n=== 推理结果 ===")
    print(f"归一化动作均值: v={action_mean[0]:.4f}, w={action_mean[1]:.4f}")
    print(f"实际速度指令: v={v_cmd:.4f} m/s, w={w_cmd:.4f} rad/s")


# ==================== 3. 运行demo ====================
if __name__ == "__main__":
    # 替换成你的ckpt文件路径
    CKPT_PATH = "supervised_ckpt_step_200000.pth"
    load_ppo_ckpt_and_infer(CKPT_PATH)，