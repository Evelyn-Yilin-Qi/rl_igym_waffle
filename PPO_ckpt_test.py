import torch
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt


# ==================== 1. 复用原代码的网络定义 ====================
class Actor(nn.Module):
    """PPO策略网络（拆分输入分支：LiDAR+状态特征）"""

    def __init__(self, lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128):
        super(Actor, self).__init__()

        # LiDAR特征提取分支（36维输入）
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )

        # 状态特征分支（6维：UserIntent(2)+BaseVel(2)+ActionHistory(2)）
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh()
        )

        # 合并特征后输出动作分布
        self.fc_merge = nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim)
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


# ==================== 2. 特征贡献分析核心函数 ====================
def calculate_feature_contribution(actor, test_obs_tensor, device):
    """
    计算每个原始特征对输出的贡献值
    方法1：梯度法（Saliency）- 输出对输入的梯度绝对值代表特征重要性
    方法2：消融法 - 屏蔽单个特征后输出的变化量代表特征贡献
    """
    actor.eval()

    # -------------------- 方法1：梯度法（Saliency Map） --------------------
    # 开启输入的梯度计算
    test_obs_tensor.requires_grad = True

    # 前向传播获取输出（以动作均值为例，也可以选log_std）
    mean, _ = actor(test_obs_tensor)

    # 计算输出对输入的梯度（针对v和w两个动作维度）
    feature_contrib_grad = []
    for act_idx in range(mean.shape[1]):  # 遍历每个动作维度（v和w）
        # 清空之前的梯度
        if test_obs_tensor.grad is not None:
            test_obs_tensor.grad.zero_()

        # 对第act_idx个动作的均值求梯度
        mean[:, act_idx].backward(retain_graph=True if act_idx < mean.shape[1] - 1 else False)

        # 梯度绝对值代表特征贡献（梯度越大，特征对输出影响越大）
        # 修复：梯度tensor需要先detach再转numpy
        grad_abs = torch.abs(test_obs_tensor.grad[0]).detach().cpu().numpy()
        feature_contrib_grad.append(grad_abs)

    # 合并两个动作维度的梯度贡献（取均值）
    feature_contrib_grad = np.mean(feature_contrib_grad, axis=0)

    # -------------------- 方法2：消融法（Feature Ablation） --------------------
    # 先获取原始输出（基准值）
    with torch.no_grad():
        mean_original, _ = actor(test_obs_tensor)
        mean_original = mean_original.cpu().numpy()[0]  # (2,)

    # 修复核心：对需要grad的tensor，必须先detach再转numpy
    obs_np = test_obs_tensor.detach().cpu().numpy()[0]
    feature_contrib_ablation = np.zeros_like(obs_np)

    # 逐个屏蔽特征（将特征值置为0），计算输出变化量
    with torch.no_grad():
        for feat_idx in range(test_obs_tensor.shape[1]):
            # 复制测试输入并屏蔽当前特征
            obs_ablation = test_obs_tensor.clone()
            obs_ablation[0, feat_idx] = 0.0  # 屏蔽该特征

            # 获取屏蔽后的输出
            mean_ablation, _ = actor(obs_ablation)
            mean_ablation = mean_ablation.cpu().numpy()[0]  # (2,)

            # 计算输出变化量（L1距离）作为贡献值
            contrib = np.sum(np.abs(mean_original - mean_ablation))
            feature_contrib_ablation[feat_idx] = contrib

    # -------------------- 特征命名与结果整理 --------------------
    # 特征名称（方便可视化）
    feature_names = [f"Lidar_{i + 1}" for i in range(36)] + \
                    ["UserIntent_1", "UserIntent_2", "BaseVel_1", "BaseVel_2",
                     "ActionHistory_1", "ActionHistory_2"]

    # 归一化贡献值（便于对比）
    feature_contrib_grad_norm = feature_contrib_grad / np.sum(feature_contrib_grad)
    feature_contrib_ablation_norm = feature_contrib_ablation / np.sum(feature_contrib_ablation)

    return {
        "feature_names": feature_names,
        "grad_contrib": feature_contrib_grad,  # 原始梯度贡献
        "grad_contrib_norm": feature_contrib_grad_norm,  # 归一化梯度贡献
        "ablation_contrib": feature_contrib_ablation,  # 原始消融贡献
        "ablation_contrib_norm": feature_contrib_ablation_norm,  # 归一化消融贡献
        "mean_original": mean_original  # 原始输出均值
    }


def visualize_feature_contribution(contrib_result, top_k=10):
    """可视化特征贡献（展示Top-K重要特征）"""
    # 合并两种方法的贡献值（取均值）
    combined_contrib = (contrib_result["grad_contrib_norm"] + contrib_result["ablation_contrib_norm"]) / 2

    # 排序获取Top-K特征
    sorted_idx = np.argsort(combined_contrib)[::-1]
    top_idx = sorted_idx[:top_k]

    top_names = [contrib_result["feature_names"][i] for i in top_idx]
    top_grad = contrib_result["grad_contrib_norm"][top_idx]
    top_ablation = contrib_result["ablation_contrib_norm"][top_idx]
    top_combined = combined_contrib[top_idx]

    # 绘图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(top_names))
    width = 0.25

    plt.bar(x - width, top_grad, width, label='梯度法贡献', color='#1f77b4')
    plt.bar(x, top_ablation, width, label='消融法贡献', color='#ff7f0e')
    plt.bar(x + width, top_combined, width, label='综合贡献', color='#2ca02c')

    plt.xlabel('特征名称')
    plt.ylabel('归一化贡献值')
    plt.title(f'Top-{top_k} 重要特征贡献分析')
    plt.xticks(x, top_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 打印详细数值
    print("\n=== 特征贡献详细数值 ===")
    print(f"{'特征名称':<15} {'梯度贡献(归一化)':<20} {'消融贡献(归一化)':<20} {'综合贡献':<20}")
    print("-" * 75)
    for i in top_idx:
        name = contrib_result["feature_names"][i]
        grad = contrib_result["grad_contrib_norm"][i]
        ablation = contrib_result["ablation_contrib_norm"][i]
        combined = combined_contrib[i]
        print(f"{name:<15} {grad:<20.6f} {ablation:<20.6f} {combined:<20.6f}")


# ==================== 3. 加载模型并分析特征贡献 ====================
def load_ppo_ckpt_and_analyze(ckpt_path):
    """加载模型并分析特征贡献"""
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 初始化网络
    actor = Actor(lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128).to(device)

    # 3. 加载ckpt文件
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt文件不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"成功加载ckpt: {ckpt_path}")
    print(f"该ckpt保存时的训练步数: {checkpoint['step']}")

    # 4. 构建测试输入
    lidar_obs = np.ones(36, dtype=np.float32) * 0.3 # LiDAR全1
    state_obs = np.ones(6, dtype=np.float32) * 0.5  # 状态特征0.5
    test_obs = np.concatenate([lidar_obs, state_obs], axis=0)
    test_obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0).to(device)

    # 5. 计算特征贡献
    print("\n正在计算特征贡献...")
    contrib_result = calculate_feature_contribution(actor, test_obs_tensor, device)

    # 6. 可视化结果
    visualize_feature_contribution(contrib_result, top_k=15)

    # 7. 输出原始输出值
    print(f"\n原始输出动作均值: v={contrib_result['mean_original'][0]:.6f}, w={contrib_result['mean_original'][1]:.6f}")


# ==================== 4. 主函数 ====================
if __name__ == "__main__":
    # 替换成你的ckpt文件路径
    CKPT_PATH = "ppo_ckpt_step_10000.pth"  # 或绝对路径
    load_ppo_ckpt_and_analyze(CKPT_PATH)