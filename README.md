# SymbioDrive - WaffleDrive RL Environment

本项目是为 SymbioDrive (Intent-Aware Shared Autonomy Solution for Senior Mobility) 项目打造的底层张量化强化学习环境。
基于 NVIDIA Isaac Sim 2022.2.1 和 OmniIsaacGymEnvs (OIGE) 框架，实现了 Waffle Pi 机器人的高吞吐量并发训练。

## 📁 工程目录结构指南

* `assets/`: 存放 Waffle Pi 的 USD 3D 模型文件。
* `cfg/`: Hydra 配置文件中枢。
  * `task/`: 物理引擎、环境参数（控制频率、奖励函数权重、安全距离）的总控台。
  * `train/`: 存放 skrl 算法和 PPO 相关的超参配置。
* `envs/`: 强化学习环境核心逻辑。
  * `igym_waffle_env.py`: 包含了四大场景（Empty, Cylinder, Box, Door）的动态生成、物理重置机制（地下停车场盲盒法）以及定制化的奖励与惩罚计算。
* `models/`: 自定义神经网络架构。
  * `custom_policy.py`: 适用于 Waffle Pi 的 CNN + LSTM + MLP 共享网络架构。
* `scripts/`: 执行脚本的入口。
  * `train_stage1.py` / `train_stage2.py`: 训练启动脚本。
  * `eval_stage1.py` / `eval_stage2.py`: 包含可视化界面的检阅评估脚本。
* `utils/`: 工具链。
  * `skrl_utils.py`: 负责搭建环境、分配显存、组装训练器的“组装工厂”。
  