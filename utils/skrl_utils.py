"""
WaffleDrive 组装工厂 - 物理引擎扩容版
解决并发环境过多导致的 PhysX GPU 显存缓冲区溢出
动态支持任意 Stage，彻底消除 if/else 硬编码！
"""
import os
from omegaconf import DictConfig, OmegaConf
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch.sequential import SequentialTrainer

def setup_waffledrive_training(cfg: DictConfig, stage_level: int, load_path: str = None):
    OmegaConf.set_struct(cfg, False)
    
    default_oige_args = {
        "device_id": 0, "sim_device": "gpu", "rl_device": "cuda:0",
        "graphics_device_id": 0, "headless": cfg.train.trainer.headless, "test": False, 
        "seed": 42, "enable_livestream": False, "enable_cameras": False
    }

    for k, v in default_oige_args.items():
        if k not in cfg: cfg[k] = v
        if k not in cfg.task: cfg.task[k] = v

    # 动态将外部传入的 stage_level 压入配置
    cfg.task.env.randomization.stage_level = stage_level
    
    if "sim" not in cfg.task:
        cfg.task.sim = {}
    if "physx" not in cfg.task.sim:
        cfg.task.sim.physx = {}
        
    cfg.task.sim.physx.gpu_found_lost_pairs_capacity = 1048576    
    cfg.task.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1048576
    cfg.task.sim.physx.gpu_total_aggregate_pairs_capacity = 1048576
    cfg.task.sim.physx.gpu_max_rigid_contact_count = 1048576
    cfg.task.sim.physx.gpu_max_rigid_patch_count = 1048576
    cfg.task.sim.physx.gpu_heap_capacity = 67108864
    cfg.task.sim.physx.gpu_temp_buffer_capacity = 16777216
    cfg.task.sim.physx.gpu_max_num_partitions = 8

    print(f"⚙️ 正在挂载向量化管理器并点火底层引擎...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)
    
    print("🔧 正在强行唤醒 Replicator 域随机化扩展...")
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.isaac")
    enable_extension("omni.replicator.core")
    
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from envs.igym_waffle_env import WaffleDriveTask 
    from models.custom_policy import WaffleSharedModel
    
    sim_config_obj = SimConfig(cfg)
    
    print(f"📦 正在挂载自定义任务逻辑 [Stage {stage_level}]...")
    task = WaffleDriveTask(name="WaffleDrive", sim_config=sim_config_obj, env=vec_env)
    
    vec_env.set_task(task=task, sim_params=sim_config_obj.get_physics_params(), backend="torch", init_sim=True)
    
    env = wrap_env(vec_env, wrapper="omniverse-isaacgym")
    device = env.device
    
    rollouts = cfg.train.agent.rollouts
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)
    
    shared_model = WaffleSharedModel(env.observation_space, env.action_space, device)
    models = {"policy": shared_model, "value": shared_model}
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    agent_cfg = OmegaConf.to_container(cfg.train.agent, resolve=True)
    cfg_ppo.update(agent_cfg)

    # =================================================================
    # 【架构解耦】：完全抛弃 if/else 硬编码，使用动态 F-String 处理多阶段！
    # =================================================================
    if "experiment" not in cfg_ppo:
        cfg_ppo["experiment"] = {}
    cfg_ppo["experiment"]["directory"] = "runs"
    # 实验名动态生成：Waffle_Stage1, Waffle_Stage2, Waffle_Stage3...
    cfg_ppo["experiment"]["experiment_name"] = f"Waffle_Stage{stage_level}" 
    cfg_ppo["experiment"]["write_interval"] = env.num_envs * rollouts
    
    agent = PPO(models=models, memory=memory, cfg=cfg_ppo, 
                observation_space=env.observation_space, action_space=env.action_space, device=device)
    
    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"⚠️ 找不到预训练权重 {load_path}！")
        print(f"📥 正在注入已有灵魂权重: {load_path}")
        agent.load(load_path)
    
    # 动态读取 epoch：去 YAML 里找 "stageX_epochs" 字段，找不到默认给 100
    epoch_key = f"stage{stage_level}_epochs"
    epochs = cfg.train.trainer.get(epoch_key, 100) 
    
    total_timesteps = epochs * rollouts * env.num_envs
    
    cfg_trainer = {"timesteps": total_timesteps, "headless": cfg.train.trainer.headless}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    return trainer, agent