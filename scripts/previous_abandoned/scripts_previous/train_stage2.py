"""
WaffleDrive Stage 2 (skrl 纯 GPU 版): 四大混合场景进阶避障训练
纯净版：加载 Stage 1 基础权重，并在高压环境中完成终极进化
"""
import os
from hydra import initialize, compose
from omegaconf import DictConfig

def main():
    print("📜 正在解析配置 (Stage 2 混合地形模式)...")
    
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        
    print("🚧 [Stage 2] 启动 4090 纯张量引擎：四大混合场景进阶训练...")
    
    # 局部导入，保证底层依赖不发生时序冲突
    from utils.skrl_utils import setup_waffledrive_training
    
    # 1. 指定 Stage 1 产出的“初级大脑”路径
    stage1_path = "./waffle_ppo_stage1.pt"
    if not os.path.exists(stage1_path):
        print(f"⚠️ 警告：未在根目录找到 {stage1_path}！如果是首次跑，请先完成 Stage 1。")
    
    # 2. 传参：stage_level=2 并且带上预训练权重
    trainer, agent = setup_waffledrive_training(cfg, stage_level=2, load_path=stage1_path)
    
    # 3. 开启终极炼丹
    trainer.train()
    
    # 4. 存档
    agent.save("./waffle_ppo_stage2.pt")
    print("🎉 [Stage 2] 终极课程训练通关！跨界大脑已保存为 waffle_ppo_stage2.pt")

if __name__ == '__main__':
    main()