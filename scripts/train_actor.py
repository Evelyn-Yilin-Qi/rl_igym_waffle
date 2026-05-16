import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Project root (for core_network)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_network import create_policy


# ==================== 日志保存 ====================
class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        print(f"📝 日志将自动保存到: {self.log_path}")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


sys.stdout = Logger()

seed = 27
rng = np.random.default_rng(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 与 RL / assemble 对齐：360 LiDAR + 4 维状态（intent2 + base_vel2）
LIDAR_DIM = 360
STATE_DIM = 4
OBS_DIM = LIDAR_DIM + STATE_DIM
ACT_DIM = 2

# 监督预训练变体：(network_type, exclude_door_data, create_policy 额外 kwargs)
PRETRAIN_VARIANTS = [
    ("simple_fc_sft", False, {"hidden_dim": 128}),
    ("cnn_lstm_sft", False, {}),
    ("cnn_gru_sft", False, {}),
    ("fc_lstm_sft", False, {}),
    ("cnn_lstm_sft_nodoor", True, {}),
]


def collect_npz_paths(data_root: str, exclude_door: bool):
    paths = sorted(Path(data_root).rglob("*.npz"))
    if exclude_door:
        paths = [p for p in paths if "door" not in p.as_posix().lower()]
    return paths


def load_merged_from_paths(paths):
    all_obs, all_v, all_w = [], [], []
    for file_path in paths:
        try:
            data = np.load(str(file_path))
            obs = data["obs"]
            v = data["v"]
            w = data["w"]
            mask = ~((v == 0) & (w == 0))
            obs = obs[mask]
            v = v[mask]
            w = w[mask]
            all_obs.append(obs)
            all_v.append(v)
            all_w.append(w)
            print(f"✅ 加载: {file_path} | 有效: {len(obs)}")
        except Exception as e:
            print(f"❌ 失败: {file_path} | {e}")
    if not all_obs:
        raise RuntimeError("没有可用的 npz 数据（检查路径或 door 过滤是否过严）")
    obs_all = np.concatenate(all_obs, axis=0)
    v_all = np.concatenate(all_v, axis=0)
    w_all = np.concatenate(all_w, axis=0)
    return obs_all, v_all, w_all


class RobotDataset(Dataset):
    def __init__(self, obs_np, v_np, w_np, max_v=1.0, max_w=1.0, lidar_dim=LIDAR_DIM, state_dim=STATE_DIM):
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        self.obs_dim = self.lidar_dim + self.state_dim
        self.obs = torch.tensor(np.squeeze(obs_np), dtype=torch.float32)[:, : self.obs_dim]
        self.v = torch.tensor(v_np, dtype=torch.float32) / max_v
        self.w = torch.tensor(w_np, dtype=torch.float32) / max_w
        self.actions = torch.stack([self.v, self.w], dim=1)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def train_one_variant(
    network_type: str,
    save_path: str,
    obs_all: np.ndarray,
    v_all: np.ndarray,
    w_all: np.ndarray,
    device: torch.device,
    extra_kw: dict,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
):
    dataset = RobotDataset(obs_all, v_all, w_all, max_v=1.0, max_w=1.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    policy = create_policy(
        network_type,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        lidar_dim=LIDAR_DIM,
        state_dim=STATE_DIM,
        **extra_kw,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    print(f"\n========== 监督预训练 | {network_type} ==========")
    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            pred = policy.predict_mean(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch + 1:3d}/{epochs}] | Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), save_path)
            print(f"  ✅ 保存: {save_path} | best_loss={best_loss:.6f}")

    return policy, best_loss


def test_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = create_policy(
        "simple_fc_sft",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        lidar_dim=LIDAR_DIM,
        state_dim=STATE_DIM,
        hidden_dim=128,
    ).to(device)
    policy.eval()
    x = torch.randn(4, OBS_DIM, device=device)
    with torch.no_grad():
        m, ls = policy(x)
    print(f"[smoke] simple_fc_sft forward OK | mean {m.shape} log_std {ls.shape}")
    return policy


def train_all_variants():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    data_root = "data"
    all_paths = collect_npz_paths(data_root, exclude_door=False)
    if not all_paths:
        raise FileNotFoundError(f"在 {data_root} 下未找到 .npz")

    print(f"共发现 {len(all_paths)} 个 npz（未过滤 door）")

    for network_type, exclude_door, extra_kw in PRETRAIN_VARIANTS:
        paths = collect_npz_paths(data_root, exclude_door=exclude_door)
        if not paths:
            print(f"⚠️ 跳过 {network_type}: 过滤后无数据")
            continue
        print(f"\n--- 变体 {network_type} | exclude_door={exclude_door} | 文件数={len(paths)} ---")
        obs_all, v_all, w_all = load_merged_from_paths(paths)
        print(f"合并后 N={obs_all.shape[0]} | obs {obs_all.shape}")

        save_path = os.path.join(str(PROJECT_ROOT), f"best_actor_{network_type}.pth")
        train_one_variant(
            network_type, save_path, obs_all, v_all, w_all, device=device, extra_kw=extra_kw
        )

        if network_type == "simple_fc_sft":
            legacy = os.path.join(str(PROJECT_ROOT), "best_actor.pth")
            shutil.copy2(save_path, legacy)
            print(f"📎 兼容拷贝 -> {legacy}")

    print("\n🎉 全部变体预训练完成。")


if __name__ == "__main__":
    test_init()
    train_all_variants()
