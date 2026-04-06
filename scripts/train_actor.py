      
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from pathlib import Path

# seed = 35 # for 36
seed = 27 # for 360
rng = np.random.default_rng(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# GPU随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== 1. 复用原代码的网络定义（必须保留） ====================
class Actor(nn.Module):
    """PPO策略网络（拆分输入分支：LiDAR+状态特征）"""

    def __init__(self, lidar_dim=360, state_dim=4, act_dim=2, hidden_dim=128, share_bb=None):
        super(Actor, self).__init__()
        self.lidar_dim = lidar_dim
        self.share_bb = share_bb  # 可选：共享基底网络（如果需要共享权重）
        # LiDAR特征提取分支（360维输入）
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 状态特征分支（4维：UserIntent(2)+BaseVel(2)
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        init.normal_(self.mean_layer.weight, mean=0.0, std=0.01)
        init.constant_(self.mean_layer.bias, 0.0)
        init.normal_(self.log_std_layer.weight, mean=0.0, std=0.01)
        init.constant_(self.log_std_layer.bias, 0.0)

        # 动作标准差限制（避免方差过大/过小）
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        # 拆分输入：前360维是LiDAR，后6维是状态特征
        lidar_feat = x[:, :self.lidar_dim]  # (batch, 360)
        state_feat = x[:, self.lidar_dim:]  # (batch, 4)

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
def test_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 初始化网络（和训练时的参数一致）
    actor = Actor(lidar_dim=360, state_dim=4, act_dim=2, hidden_dim=128).to(device)

    w_list = []
    v_list = []
    total_count = 10000
    for i in range(total_count):
        # 4. 构建测试输入（模拟obs：360维LiDAR + 4维状态特征）
        # LiDAR部分：全1表示无障碍物，距离最大
        lidar_obs = torch.rand(360, dtype=torch.float32, device=device)

        # 状态特征：UserIntent(2)+BaseVel(2)
        user_v = torch.rand(1, dtype=torch.float32, device=device)
        user_w = torch.empty(1, dtype=torch.float32, device=device).uniform_(-1, 1)
        base_v = torch.rand(1, dtype=torch.float32, device=device)
        base_w = torch.empty(1, dtype=torch.float32, device=device).uniform_(-1, 1)

        test_obs = torch.cat([lidar_obs, user_v, user_w, base_v, base_w], dim=0)

        # 增加 batch 维度（模型输入）
        test_obs_tensor = test_obs.unsqueeze(0)

        # 5. 推理（关闭梯度计算，提升速度）
        actor.eval()  # 切换到评估模式
        with torch.no_grad():
            mean, log_std = actor(test_obs_tensor)
            std = log_std.exp()

            # 输出动作均值（归一化到[-1,1]）
            action_mean = mean.cpu().numpy().flatten()
            # 反归一化到实际速度范围（和训练时一致）
            max_v = 1.0
            max_w = 1.0
            v_cmd = action_mean[0] * max_v
            w_cmd = action_mean[1] * max_w
        w_list.append(w_cmd)
        v_list.append(w_cmd)

    w_count = sum(1 for x in w_list if x > 0)
    v_count = sum(1 for x in w_list if x > 0)
    print('w_count:', w_count / total_count)
    print('v_count:', v_count / total_count)
    return actor


class RobotDataset(Dataset):
    def __init__(self, obs_np, v_np, w_np, max_v=1.0, max_w=1.0, lidar_dim=360, state_dim=4):
        """
        把npz数据包装成PyTorch数据集
        obs: (N, 1, 40) → 自动 squeeze 成 (N,40)
        v, w: (N,)
        """
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        self.obs_dim = self.lidar_dim + self.state_dim
        self.obs = torch.tensor(np.squeeze(obs_np), dtype=torch.float32)[:, :self.obs_dim]
        self.v = torch.tensor(v_np, dtype=torch.float32)
        self.w = torch.tensor(w_np, dtype=torch.float32)

        # 【关键】把真实v/w归一化到[-1,1]，和网络输出范围匹配
        self.v = self.v / max_v
        self.w = self.w / max_w

        # 拼接成 [v, w]
        self.actions = torch.stack([self.v, self.w], dim=1)  # [N,2]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# ==================== 【新增：训练函数】 ====================
def train_actor():
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 2. 超参数
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    MAX_V = 1.0
    MAX_W = 1.0
    SAVE_PATH = "best_actor.pth"
    # 数据根目录：遍历data下所有子目录的npz文件
    DATA_ROOT = "data"

    # 3. 遍历所有npz文件并合并数据
    print(f"正在扫描目录: {DATA_ROOT} 下的所有 .npz 文件...")
    # 获取所有npz文件（递归遍历子目录）
    npz_files = list(Path(DATA_ROOT).rglob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"在 {DATA_ROOT} 目录下未找到任何 .npz 文件！")

    print(f"找到 {len(npz_files)} 个数据文件，开始加载...")

    # 初始化空列表存储所有数据
    all_obs = []
    all_v = []
    all_w = []

    # 逐个加载文件
    for file_path in npz_files:
        try:
            data = np.load(str(file_path))
            obs = data['obs']
            v = data['v']
            w = data['w']

            # 过滤0速度数据
            mask = ~((v == 0) & (w == 0))
            obs = obs[mask]
            v = v[mask]
            w = w[mask]

            all_obs.append(obs)
            all_v.append(v)
            all_w.append(w)

            print(f"✅ 加载成功: {file_path} | 有效数据: {len(obs)}")
        except Exception as e:
            print(f"❌ 加载失败: {file_path} | 错误: {str(e)}")

    # 合并所有数据
    obs_all = np.concatenate(all_obs, axis=0)
    v_all = np.concatenate(all_v, axis=0)
    w_all = np.concatenate(all_w, axis=0)
    print(f"\n数据合并完成！总数据量: {obs_all.shape[0]} 条")
    print(f"obs shape: {obs_all.shape} | v shape: {v_all.shape} | w shape: {w_all.shape}")

    # 4. 构建数据集 & 加载器
    dataset = RobotDataset(obs_all, v_all, w_all, max_v=MAX_V, max_w=MAX_W)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 5. 初始化模型、优化器、损失
    actor = Actor(lidar_dim=360, state_dim=4, act_dim=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=LR)
    criterion = nn.MSELoss()  # 回归任务用MSE

    # 6. 开始训练
    print("\n========== 开始训练 ==========")
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        actor.train()
        total_loss = 0.0
        for batch_obs, batch_act in dataloader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            # 前向：只需要动作均值（监督学习）
            pred_act = actor.predict_mean(batch_obs)

            # 损失
            loss = criterion(pred_act, batch_act)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.size(0)

        # 每个epoch打印
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch + 1:2d}/{EPOCHS}] | Loss: {avg_loss:.6f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(actor.state_dict(), SAVE_PATH)
            print(f"✅ 最优模型已保存: {SAVE_PATH} | 最佳Loss: {best_loss:.6f}")

    print("\n🎉 训练完成！")

    # 测试推理 (已修复 detach() 问题)
    import random
    n_test = 10
    print("\n========== 随机测试推理结果 ==========")
    actor.eval()  # 切换到评估模式
    with torch.no_grad():  # 关闭梯度计算，更安全高效
        for i in range(n_test):
            num = random.randint(0, len(dataset) - 1)
            obs_tensor = dataset.obs[num].unsqueeze(0).to(device)
            # 修复：添加 .detach()
            pred = actor.predict_mean(obs_tensor).detach().cpu().numpy()[0]
            print(
                f'测试 {i + 1}: 预测[v,w] = [{pred[0]:.4f}, {pred[1]:.4f}] | 真实[v,w] = [{v_all[num]:.4f}, {w_all[num]:.4f}]')

    return actor


# ==================== 3. 运行demo ====================
if __name__ == "__main__":
    # 初始化测试
    actor = test_init()
    # 训练（自动加载所有npz数据）
    actor = train_actor()

    