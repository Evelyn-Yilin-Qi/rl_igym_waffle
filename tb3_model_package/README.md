# TB3模型使用说明

## 📦 文件夹内容

- `ppo_final_*.pth`: 训练好的模型权重文件（需要手动复制）
- `tb3_model.py`: 模型推理类（包含模型结构定义和加载逻辑）
- `requirements.txt`: Python依赖包
- `README.md`: 本使用说明
- `copy_model.sh`: 复制模型文件的辅助脚本（可选）

## 🚀 准备模型文件

模型文件（`.pth`）已经包含在文件夹中，可以直接使用。

如果需要更新模型文件，可以手动复制新的`.pth`文件到此文件夹。

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
```

### 2. 使用模型

```python
from tb3_model import TB3ModelInference
import numpy as np

# 初始化模型（自动查找同目录下的.pth文件）
model = TB3ModelInference(device="cpu")
# 或者指定模型文件名: model = TB3ModelInference("ppo_final_20260312_16.pth", device="cpu")
# 如果有GPU，可以使用 device="cuda"

# 准备观察数据（44维）
observation = np.array([...], dtype=np.float32)  # 形状: (44,)

# 预测动作
action = model.predict(observation, deterministic=False)
# action是numpy数组，形状为(2,)，包含[linear_velocity, angular_velocity]
```

## 观察数据格式

观察向量共**44维**，按顺序为：

- **维度0-35**: LiDAR数据（36维）
- **维度36-37**: UserIntent（2维）- 用户意图方向
- **维度38-39**: BaseVelocity（2维）- 当前线速度和角速度
- **维度40-43**: ActionHistory（4维）- 历史动作

## 动作输出格式

动作向量共**2维**：

- **维度0**: `linear_velocity` - 线性速度（m/s）
- **维度1**: `angular_velocity` - 角速度（rad/s）

## API说明

### `TB3ModelInference(model_path=None, device="cpu")`

初始化模型推理类。

**参数：**
- `model_path`: 模型文件路径（.pth文件），如果为`None`则自动查找同目录下的.pth文件
- `device`: 计算设备，`"cpu"` 或 `"cuda"`

### `predict(observation, deterministic=False)`

根据观察预测动作。

**参数：**
- `observation`: numpy数组，形状为`(44,)`或`(batch, 44)`，dtype为`float32`
- `deterministic`: 
  - `True`: 返回均值动作（确定性策略）
  - `False`: 从分布中采样（随机策略，训练时使用）

**返回：**
- `action`: numpy数组，形状为`(2,)`或`(batch, 2)`，包含`[linear_velocity, angular_velocity]`

## 使用示例

### 示例1：单次预测

```python
from tb3_model import TB3ModelInference
import numpy as np

# 加载模型（自动查找同目录下的.pth文件）
model = TB3ModelInference()
# 或者指定文件名: model = TB3ModelInference("ppo_final_20260312_16.pth")

# 准备观察数据（假设你已经从传感器获取了数据）
lidar_data = np.random.randn(36).astype(np.float32)  # 36维LiDAR
user_intent = np.array([0.5, 0.8], dtype=np.float32)  # 2维用户意图
base_vel = np.array([0.2, 0.1], dtype=np.float32)  # 2维当前速度
action_history = np.array([0.2, 0.1, 0.15, 0.05], dtype=np.float32)  # 4维历史动作

# 组装观察向量
observation = np.concatenate([lidar_data, user_intent, base_vel, action_history])
assert observation.shape == (44,), f"观察维度错误: {observation.shape}"

# 预测动作
action = model.predict(observation, deterministic=True)
print(f"预测动作: linear={action[0]:.3f}, angular={action[1]:.3f}")
```

### 示例2：批量预测

```python
# 批量预测（多个观察）
observations = np.random.randn(10, 44).astype(np.float32)  # 10个观察
actions = model.predict(observations, deterministic=True)
print(f"批量动作形状: {actions.shape}")  # (10, 2)
```

### 示例3：集成到ROS节点

```python
#!/usr/bin/env python3
import rospy
from tb3_model import TB3ModelInference
import numpy as np

class TB3RLController:
    def __init__(self):
        # 加载模型（自动查找同目录下的.pth文件）
        self.model = TB3ModelInference()
        # 或者指定文件名: self.model = TB3ModelInference("ppo_final_20260312_16.pth")
        
        # ROS订阅和发布
        # self.lidar_sub = rospy.Subscriber(...)
        # self.cmd_vel_pub = rospy.Publisher(...)
    
    def callback(self, lidar_msg, user_intent, base_vel, action_history):
        # 组装观察
        observation = np.concatenate([
            lidar_msg.ranges[:36],  # LiDAR数据
            user_intent,  # 用户意图
            base_vel,  # 当前速度
            action_history  # 历史动作
        ]).astype(np.float32)
        
        # 预测动作
        action = self.model.predict(observation, deterministic=True)
        
        # 发布控制命令
        # cmd_vel.linear.x = action[0]
        # cmd_vel.angular.z = action[1]
        # self.cmd_vel_pub.publish(cmd_vel)

if __name__ == "__main__":
    rospy.init_node("tb3_rl_controller")
    controller = TB3RLController()
    rospy.spin()
```

## 注意事项

1. **观察数据格式**：确保观察数据格式正确（44维，float32）
2. **设备选择**：默认使用CPU，如果有GPU可以指定`device="cuda"`
3. **确定性vs随机性**：
   - `deterministic=True`: 返回均值动作，适合实际部署
   - `deterministic=False`: 从分布中采样，更接近训练时的行为
4. **数值范围**：模型输出的动作值已经过处理，可以直接使用
5. **模型文件**：确保`.pth`文件与`tb3_model.py`在同一目录，如果不指定`model_path`参数，会自动查找同目录下的.pth文件

## 常见问题

**Q: 模型加载失败？**
A: 检查模型文件路径是否正确，确保文件存在。

**Q: 观察维度不匹配？**
A: 确保观察向量严格为44维，按顺序：LiDAR(36) + UserIntent(2) + BaseVel(2) + ActionHistory(4)。

**Q: 动作值异常？**
A: 检查观察数据是否包含NaN或Inf，确保数据格式为float32。

**Q: 如何集成到ROS？**
A: 参考"示例3：集成到ROS节点"，根据你的实际ROS话题调整订阅和发布。

## 联系

如有问题，请联系模型提供者。
