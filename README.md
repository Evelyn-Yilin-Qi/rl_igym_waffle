由于加了很多新的东西，且我要保证通路的正确性，其实很难做到比较标准的架构，哎真的不太可能我也不强迫我自己了，想下去只会让自己过拟合；
我现在能保证的是两个stage的训练的代码都能跑通且我的参数都设置的很小，sft和rl的过程都能完全接入了，sft作为一个可选过程直接可以使用true false选择是否要加入，当然参数如何设置以及sft rule 等等怎么能弄的更好就是后话了。
 我尽可能保留 冯 的一些结构且所有参数名字的变化我后面全都加注释了。 
readme里面我已经把比较重要的地方都标❗❗❗了
如果还有啥看不懂的还是随时说吧。
在目前加入了完整的stage1 stage2之后那肯定是要重新调整参数了。



把SFT加到了stage1和stage2的训练过程里面，且保留了更改的过程，请主要使用 sft_rl_tb3_V3_modular；stage2_rl_sft；stage1_rl_sft 这三个代码！
关注我写的❗❗❗的地方都比较重要，其他地方能不改就不改了吧
```
rl_igym_waffle/
├── core_network/        ❗❗❗如果后续要用别的架构就要改这里   # 核心网络架构模块
│   ├── __init__.py      # 工厂函数：create_policy(), create_value()
│   ├── base.py          # 基础接口：BasePolicy, BaseValue
│   ├── simple_fc.py     # 简单全连接架构（LiDAR分支 + 状态分支）
│   └── essay_base.py    # 论文架构（1D Conv + LSTM）
│    （或许我们还有很多别的架构）
│
├── algorithms/          # RL算法模块
│   ├── __init__.py      # 工厂函数：create_algorithm()
│   ├── base.py          # 算法基础接口：BaseRLAlgorithm
│   └── ppo.py           # PPO算法实现
│
├── rewards/             ❗❗❗如果要改奖励函数的参数请动这里 # 奖励函数模块
│   ├── __init__.py      # 导出所有奖励组件
│   ├── base.py          # 奖励组件基类：BaseRewardComponent
│   ├── composer.py      # 奖励组合器：RewardComposer（统一管理）  ❗组装所有enable为 Ture的奖励函数
│   ├── obstacle.py      # 障碍物惩罚
│   ├── heading.py       # 航向惩罚
│   ├── smoothness.py    # 动作平滑惩罚
│   ├── static.py        # 静态惩罚（静止不动）
│   └── centrifugal.py   # 离心力惩罚（可选）❗暂时没用上
│
├── envs/                # 环境观察模块
│   ├── __init__.py
│   ├── observations.py  # 观察向量组装：LiDAR、碰撞检测、44维观察
│   └── user_intent.py   # 用户意图计算（目标方向）
│
├── sim/                 # 仿真环境模块
│   ├── robot/           # 机器人相关
│   │   ├── tb3_config.py    # TB3配置（速度限制、尺寸等）
│   │   └── ...
│   └── scenes/          # 场景管理
│       ├── scene_manager.py  # 场景管理器（创建、重置障碍物）
│       ├── scene_base.py     # 场景基类和常量
│       └── ...
│
├── process_settings/    # 训练流程设置模块
│   ├── __init__.py
│   └── env_setup.py     # 环境初始化类：EnvironmentSetup
│
├── sft/                 ❗❗❗SFT相关逻辑都在这里
│   ├── rule_controller.py  # 规则控制器：监督label怎么生成
│   ├── supervised.py       # 监督训练：MSE训练策略网络   要是觉得需要改就动这里
│   ├── logging.py          # SFT/RL打印工具 正式的Stage里面就不print了
│   └── config.py           # SFT相关配置读取工具（轻量）
│
├── config/              ❗❗❗ 改要使用的re function； network；RL 策略等# 配置模块（YAML格式）  
│   ├── network_config.yaml    # 网络架构配置（支持enabled切换）
│   ├── algorithm_config.yaml  # RL算法配置（支持enabled切换）
│   └── reward_config.yaml     # 奖励函数配置（支持enabled/weight）
│
├── utils/               # 工具函数模块
│   └── config_utils.py  # 配置读取工具：get_enabled_component()
│
├── scripts/              # 训练脚本   （有关SFT以及RL的一些参数代码的命名我都改了，注释里面写了原名和现名，改是为了统一加上好理解
│   ├── stage1_rl_sft.py       # ❗❗❗❗❗Stage1 SFT + RL 之后修改stage1主要用这个  只有空场景 目前env中写死了16个
│   │                             只能说跑通了,效果完全没验证过                         env改的时候建议【4，16，32，64这种的】
│   ├── stage2_rl_sft.py       # ❗❗❗❗❗Stage2 SFT + RL 之后修改stage2主要用这个  四个场景 目前env中写死了16个
│   │                             只能说跑通了,效果完全没验证过                         env改的时候建议【4，16，32，64这种的】
│   ├── sft_rl_tb3_V3_modular.py ❗❗❗❗❗ 如果要做任何运行测试，可以在这里做，但只运行一个env，四个场景都可选择可以在代码中切换，main注释写清楚了
│   │                           
│   ├── sft_rl_tb3_V3.py       # 冯的版本对齐到V3  这个可以运行，但是只是box单一场景 (test)
│   ├── sft_rl_tb3.py          # 冯原始的脚本  (test)
│   └── PPO_ckpt_test.py       # ❗❗❗checkpoint打包/测试脚本  反正现在没有可用的模型我就没动，这个代码没有做适配
│
│   # 旧脚本（别用了）
│   ├── recent_old_version/stage1_sft_rl.py
│   ├── recent_old_version/stage2_sft_rl.py
│   └── test/stage1_rl.py, test/stage2_rl.py, test/*.py
│
└── cfg/                 # 别再用了，只是为了几个老脚本留着了,新脚本都没引用 
    ├── WaffleDrive.yaml  # 
```

---
