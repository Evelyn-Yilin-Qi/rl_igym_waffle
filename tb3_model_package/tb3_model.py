"""
TB3模型推理类
独立使用，不依赖原工程
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class EssayBasePolicy(nn.Module):
    """Essay Base策略网络（1D Conv + LSTM架构）"""
    def __init__(self, obs_dim=44, act_dim=2, lidar_dim=36, state_dim=8):
        super().__init__()
        
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        
        # LiDAR分支：1D Conv处理
        self.conv1d_1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=0
        )
        
        self.conv1d_2 = nn.Conv1d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
        conv_output_size = 7 * 4
        
        # FC0: 将Conv输出映射到64维
        self.fc0 = nn.Linear(conv_output_size, 64)
        
        # LSTM: 融合LiDAR特征(64) + 状态特征(8) = 72维输入
        self.lstm = nn.LSTM(
            input_size=64 + state_dim,
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
        lidar_feat = x[:, :self.lidar_dim]
        state_feat = x[:, self.lidar_dim:]
        
        # LiDAR分支：1D Conv处理
        lidar_reshaped = lidar_feat.unsqueeze(1)
        
        conv1_out = torch.relu(self.conv1d_1(lidar_reshaped))
        conv2_out = torch.relu(self.conv1d_2(conv1_out))
        
        conv_flat = conv2_out.view(conv2_out.size(0), -1)
        
        fc0_out = torch.tanh(self.fc0(conv_flat))
        
        # 融合：FC0输出 + 状态特征
        fused_feat = torch.cat([fc0_out, state_feat], dim=1)
        
        # Reshape为序列输入LSTM
        lstm_input = fused_feat.unsqueeze(1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out.squeeze(1)
        
        # 数值稳定性：裁剪LSTM输出
        lstm_out = torch.clamp(lstm_out, min=-10.0, max=10.0)
        
        # FC1
        fc1_out = self.fc1(lstm_out)
        
        # 数值稳定性：裁剪FC1输出
        fc1_out = torch.clamp(fc1_out, min=-10.0, max=10.0)
        
        # 输出动作分布
        mean = self.mean_layer(fc1_out)
        log_std = self.log_std_layer(fc1_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # 数值稳定性：裁剪mean输出
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        
        return mean, log_std


class TB3ModelInference:
    """
    TB3模型推理类
    用于加载和使用训练好的模型
    """
    def __init__(self, model_path=None, device="cpu"):
        """
        初始化模型
        
        Args:
            model_path: 模型文件路径（.pth文件），如果为None则自动查找同目录下的.pth文件
            device: 计算设备，"cpu" 或 "cuda"
        """
        import os
        import glob
        
        self.device = torch.device(device)
        
        # 如果未指定模型路径，自动查找同目录下的.pth文件
        if model_path is None:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 查找.pth文件
            pth_files = glob.glob(os.path.join(current_dir, "*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"未找到模型文件，请在当前目录放置.pth文件，或指定model_path参数")
            if len(pth_files) > 1:
                print(f"⚠️  警告: 找到多个.pth文件，使用第一个: {pth_files[0]}")
            model_path = pth_files[0]
        
        # 如果是相对路径，转换为绝对路径（相对于当前文件所在目录）
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model_path)
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建模型结构
        self.policy = EssayBasePolicy(
            obs_dim=44,
            act_dim=2,
            lidar_dim=36,
            state_dim=8
        ).to(self.device)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()  # 设置为评估模式
        
        print(f"✅ 模型加载成功: {os.path.basename(model_path)}")
        print(f"   设备: {self.device}")
    
    def predict(self, observation, deterministic=False):
        """
        根据观察预测动作
        
        Args:
            observation: numpy数组，形状为(44,)或(batch, 44)
                - 维度0-35: LiDAR数据（36维）
                - 维度36-37: UserIntent（2维）
                - 维度38-39: BaseVelocity（2维）
                - 维度40-43: ActionHistory（4维）
            deterministic: 是否使用确定性策略（True返回均值，False从分布采样）
        
        Returns:
            action: numpy数组，形状为(2,)或(batch, 2)
                - 维度0: linear_velocity（线性速度）
                - 维度1: angular_velocity（角速度）
        """
        # 转换为torch tensor
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float()
        else:
            obs_tensor = observation.float()
        
        # 确保是2D (batch, features)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # 移到设备
        obs_tensor = obs_tensor.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            mean, log_std = self.policy(obs_tensor)
            
            if deterministic:
                # 确定性策略：直接返回均值
                action = mean
            else:
                # 随机策略：从分布中采样
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
        
        # 转回numpy
        action_np = action.cpu().numpy()
        
        # 如果是单样本，去掉batch维度
        if observation.ndim == 1:
            action_np = action_np[0]
        
        return action_np
    
    def get_value(self, observation):
        """
        获取状态价值（如果需要的话）
        注意：当前模型只加载了policy，value网络未包含在推理类中
        """
        raise NotImplementedError("Value网络未包含在推理类中，如需使用请扩展此类")


if __name__ == "__main__":
    # 测试示例
    print("TB3模型推理类测试")
    print("=" * 50)
    
    try:
        # 自动查找同目录下的模型文件
        model = TB3ModelInference(device="cpu")
        
        # 创建测试观察数据
        test_obs = np.random.randn(44).astype(np.float32)
        
        # 预测动作
        action = model.predict(test_obs, deterministic=True)
        print(f"\n✅ 测试成功！")
        print(f"观察形状: {test_obs.shape}")
        print(f"动作形状: {action.shape}")
        print(f"动作值: linear={action[0]:.3f}, angular={action[1]:.3f}")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n使用方法:")
        print("1. 将模型文件(.pth)放在同一目录")
        print("2. from tb3_model import TB3ModelInference")
        print("3. model = TB3ModelInference()  # 自动查找.pth文件")
        print("   或 model = TB3ModelInference('ppo_final_20260312_16.pth')  # 指定文件名")
        print("4. action = model.predict(observation)")
