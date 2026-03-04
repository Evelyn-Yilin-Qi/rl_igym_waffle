"""
WaffleDrive Stage 1 专属检阅脚本 (加载权重并在物理显示器上开启可视化)
纯空旷环境 (Empty Environment) 测试
"""
from hydra import initialize, compose
from omegaconf import DictConfig

def main():
    print("📜 正在解析配置 (Stage 1 评估模式)...")
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        
    # 为了人眼观赏，强制把环境数量降到 4 台
    cfg.task.env.numEnvs = 16
    
    # 强制关闭无头模式，要求引擎必须渲染弹窗！
    cfg.train.trainer.headless = False
    
    # 导入工厂
    from utils.skrl_utils import setup_waffledrive_training
    
    print("🎬 正在加载 Stage 1 初级大脑...")
    trainer, agent = setup_waffledrive_training(cfg, stage_level=1, load_path="./waffle_ppo_stage1.pt")
    
    # 切换到测试/评估模式 (关闭高斯噪声探索，执行确定性最优策略)
    agent.set_running_mode("eval")
    
    print("🚀 开始播放 3D 测试画面！(请在弹出的 Isaac Sim 窗口中观察)")
    # 使用 skrl 自带的 trainer 进行优雅的推理循环
    trainer.eval()

if __name__ == '__main__':
    main()