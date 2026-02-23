"""
WaffleDrive 全局引擎点火器
纯净的底层启动程序，负责解析配置并安全拉起 Isaac Sim 物理引擎
"""
import sys
from hydra import initialize, compose
from omni.isaac.kit import SimulationApp

def start_engine(config_path="../cfg", config_name="config", force_headless=None):
    """
    阻塞式启动引擎，完成后返回配置和引擎句柄
    """
    # 1. 解析全局配置，允许终端参数覆盖 (sys.argv)
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name, overrides=sys.argv[1:])
    
    # 2. 处理特殊覆盖 (例如 eval_sim.py 强行要求开画面)
    if force_headless is not None:
        cfg.train.trainer.headless = force_headless
        
    # 3. 阻塞式拉起 C++ 物理引擎
    simulation_app = SimulationApp({"headless": cfg.train.trainer.headless})
    
    return cfg, simulation_app