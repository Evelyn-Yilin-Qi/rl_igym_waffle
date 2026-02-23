"""
WaffleDrive Stage 1 (skrl 纯 GPU 版): 无障碍空旷环境基础训练
纯净版：彻底干掉 @hydra.main 装饰器，杜绝乱建 outputs 文件夹！
"""
from hydra import initialize, compose
from omegaconf import DictConfig

# ⛔️ 绝对不能在文件顶部 import 任何自定义的 envs 或 omni 模块！


def main():
    print("📜 正在解析配置...")
    
    # 1. 显式读取 YAML，绝对不触发 Hydra 建立 outputs 文件夹的行为
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        
    print("🔥 [Stage 1] 启动 4090 纯张量引擎：Empty Environment...")
    
    # 2. 在这里局部导入工厂，工厂内部会负责完美的启动顺序
    from utils.skrl_utils import setup_waffledrive_training
    
    # 3. 开始组装并训练
    trainer, agent = setup_waffledrive_training(cfg, stage_level=1)
    trainer.train()
    
    # 4. 保存模型到根目录
    agent.save("./waffle_ppo_stage1.pt")
    print("✅ [Stage 1] 纯 GPU 训练完成！权重已保存为 waffle_ppo_stage1.pt")

if __name__ == '__main__':
    main()