"""
最简单的测试脚本：4个环境，机器人能显示并落到地面
参考 train_stage1.py 的加载方式，使用 VecEnvRLGames 来管理应用生命周期
"""
# ⛔️ 绝对不能在文件顶部 import 任何自定义的 envs 或 omni 模块！
import numpy as np
import torch

def main():
    # ⚠️ 关键：在main函数内部导入omni相关模块
    # 参考 train_stage1.py 和 skrl_utils.py 的模式
    # 正确的顺序：先导入VecEnvRLGames类，实例化它（初始化Isaac Sim），然后才能导入SimConfig
    print("📦 导入基础模块...")
    from omegaconf import OmegaConf
    
    # 机器人USD路径
    robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
    
    # 创建4个环境，每个环境间距4米
    num_envs = 4
    env_spacing = 4.0
    
    # 创建最小配置（参考 train_stage1.py 和 skrl_utils.py 的方式）
    print("⚙️ 创建配置...")
    cfg = OmegaConf.create({
        "task": {
            "name": "SimpleWaffle",
            "physics_engine": "physx",
            "env": {
                "numEnvs": num_envs,
                "envSpacing": env_spacing,
                "maxEpisodeLength": 1000,
            },
            "sim": {
                "dt": 0.025,
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
                }
            }
        },
        "sim_device": "gpu",
        "device_id": 0,
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": False,
        "test": False,  # SimConfig 需要这个键
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    })
    
    # 确保配置结构正确（参考 skrl_utils.py）
    # 必须在创建 SimConfig 之前设置，并且确保所有必需的键都存在
    OmegaConf.set_struct(cfg, False)
    
    # 确保所有必需的顶层键都存在（参考 skrl_utils.py 的 default_oige_args）
    # 这些键是 SimConfig 需要的
    if "test" not in cfg:
        cfg["test"] = False
    if "seed" not in cfg:
        cfg["seed"] = 42
    if "enable_livestream" not in cfg:
        cfg["enable_livestream"] = False
    if "enable_cameras" not in cfg:
        cfg["enable_cameras"] = False
    
    # 关键：先导入VecEnvRLGames类（不导入SimConfig）
    print("🚀 导入VecEnvRLGames...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    
    # 实例化VecEnvRLGames（这会初始化 Isaac Sim）
    print("🔥 创建向量化环境管理器（这会初始化 Isaac Sim）...")
    vec_env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.sim_device)
    
    # 现在 Isaac Sim 已经初始化，可以安全导入 SimConfig 和其他模块
    print("📦 导入OmniIsaacGym模块（Isaac Sim已初始化）...")
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    
    # 在创建 SimConfig 之前，再次确保所有必需的键都存在
    # 参考 skrl_utils.py 的做法
    OmegaConf.set_struct(cfg, False)
    
    # 确保所有必需的顶层键都存在（SimConfig 需要这些键）
    required_top_level_keys = {
        "device_id": 0,
        "sim_device": "gpu",
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": False,
        "test": False,
        "seed": 42,
        "enable_livestream": False,
        "enable_cameras": False,
    }
    for k, v in required_top_level_keys.items():
        if k not in cfg:
            cfg[k] = v
    
    # 创建SimConfig
    print("🔧 创建SimConfig...")
    sim_config = SimConfig(cfg)
    
    # 创建一个简单的任务类来加载机器人
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    
    class SimpleWaffleTask(RLTask):
        def __init__(self, name, sim_config, env, offset=None):
            self._sim_config = sim_config
            self._cfg = sim_config.config
            self._task_cfg = sim_config.task_config
            self._num_envs = self._task_cfg["env"]["numEnvs"]
            self._env_spacing = self._task_cfg["env"]["envSpacing"]
            self._num_observations = 1
            self._num_actions = 2
            super().__init__(name, env, offset)
        
        def set_up_scene(self, scene):
            # 在每个环境中加载机器人
            add_reference_to_stage(usd_path=robot_path, prim_path=self.default_zero_env_path + "/WafflePi")
            super().set_up_scene(scene)
            self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/WafflePi", name="waffle_view")
            scene.add(self.robots)
        
        def get_observations(self):
            self.obs_buf = torch.zeros((self._num_envs, self._num_observations), device=self._device)
            return {self.name: {"obs": self.obs_buf}}
        
        def pre_physics_step(self, actions):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
            # 不应用任何动作，让机器人自然落下
        
        def reset_idx(self, env_ids):
            num_resets = len(env_ids)
            env_pos = self._env_pos[env_ids]
            robot_positions = env_pos.clone()
            robot_positions[:, 2] = 0.2  # Z=0.2米，让它落下来
            robot_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(num_resets, 1)
            indices = env_ids.to(dtype=torch.int32)
            self.robots.set_world_poses(robot_positions, robot_rotations, indices=indices)
            self.robots.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
            self.robots.set_joint_velocities(torch.zeros((num_resets, self.robots.num_dof), device=self._device), indices=indices)
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
        
        def post_reset(self):
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)
        
        def calculate_metrics(self):
            self.rew_buf[:] = 0.0
        
        def is_done(self):
            self.reset_buf[:] = 0
    
    # 创建任务
    print("📦 创建测试任务...")
    task = SimpleWaffleTask(name="SimpleWaffle", sim_config=sim_config, env=vec_env)
    
    # 设置任务并初始化仿真（参考 train_stage1.py）
    print("🔧 初始化仿真环境...")
    vec_env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)
    
    print(f"✅ 已创建 {num_envs} 个环境")
    print("🔄 开始物理模拟，让机器人落到地面...")
    
    # 运行物理模拟，让机器人落到地面
    for step in range(200):
        # 创建零动作
        actions = torch.zeros((num_envs, 2), device=task._device)
        
        # 执行一步（VecEnvRLGames内部会处理应用更新）
        task.pre_physics_step(actions)
        
        # 更新进度缓冲区
        task.progress_buf += 1
        
        # 计算奖励和完成状态
        task.calculate_metrics()
        task.is_done()
        
        # 渲染（VecEnvRLGames内部处理）
        vec_env.render()
        
        if step % 20 == 0:
            # 获取机器人位置
            positions, _ = task.robots.get_world_poses()
            print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
    
    # 获取最终位置
    final_positions, _ = task.robots.get_world_poses()
    print("\n✅ 测试完成！最终机器人Z位置：")
    for i in range(num_envs):
        print(f"   环境 {i}: Z = {final_positions[i, 2]:.4f} 米")
    
    print("\n💡 提示：VecEnvRLGames会自动保持应用运行")
    print("按Ctrl+C退出...")
    
    # 保持运行 - VecEnvRLGames内部会处理应用生命周期
    # 我们只需要定期调用相关方法
    try:
        import time
        step = 200
        while True:
            actions = torch.zeros((num_envs, 2), device=task._device)
            task.pre_physics_step(actions)
            task.progress_buf += 1
            task.calculate_metrics()
            task.is_done()
            vec_env.render()
            
            step += 1
            if step % 200 == 0:
                positions, _ = task.robots.get_world_poses()
                print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()