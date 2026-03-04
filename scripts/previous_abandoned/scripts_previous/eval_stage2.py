"""
WaffleDrive Stage 2 检阅脚本 (加载权重并在物理显示器上开启可视化)
四大混合场景 (Empty, Cylinder, Box, Door) 终极测试
"""
from hydra import initialize, compose
from omegaconf import DictConfig

def main():
    print("📜 正在解析配置 (Stage 2 评估模式)...")
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        
    # 强行设置评估环境数量。16台刚好是 4x4，每种地形精确分配 4 台！
    cfg.task.env.numEnvs = 16 
    
    # 强制关闭无头模式，要求引擎必须渲染弹窗！
    cfg.train.trainer.headless = False
    
    # 导入工厂 (杜绝死锁，必须在 cfg 加载后导入)
    from utils.skrl_utils import setup_waffledrive_training
    
    print("🎬 [World A] 开启视觉检阅模式，高压混合环境加载中...")
    
    # 指向我们在 train_stage2.py 中保存的最新模型
    model_path = "./waffle_ppo_stage2.pt"
    
    try:
        trainer, agent = setup_waffledrive_training(cfg, stage_level=2, load_path=model_path)
    except Exception as e:
        print(f"⚠️ 无法加载环境或权重。\n报错详情: {e}")
        return

    # 切换到测试/评估模式 (关闭高斯噪声探索，执行确定性最优策略)
    agent.set_running_mode("eval")
    
    print("🚀 开始闭环推理循环！(请在 Isaac Sim 窗口中观察四大场景的表现)")
    trainer.eval()

if __name__ == '__main__':
    main()