"""
最简单的测试脚本：4个环境，机器人能显示并落到地面
每个环境的小车在初始化时随机朝向（不会reset）
参考 train_stage1.py 的加载方式，使用 VecEnvRLGames 来管理应用生命周期
"""
# ⛔️ 绝对不能在文件顶部 import 任何自定义的 envs 或 omni 模块！
import numpy as np
import torch
import math

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
                "maxEpisodeLength": 100000,  # 设置很大的值，避免自动重置
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
            # ⚠️ 关键修复：确保 env_ids 在正确的设备上
            env_ids = env_ids.to(device=self._device)
            
            env_pos = self._env_pos[env_ids]
            robot_positions = env_pos.clone()
            robot_positions[:, 2] = 0.2  # Z=0.2米，让它落下来
            
            # ⚠️ 关键修改：在 reset_idx 中，如果已经有 initial_rotations，使用它们
            # 否则使用默认旋转（在 post_reset 中会生成随机旋转）
            if hasattr(self, 'initial_rotations'):
                robot_rotations = self.initial_rotations[env_ids].clone()
            else:
                # 如果还没有初始化旋转，使用默认旋转（在 post_reset 中会生成）
                robot_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(num_resets, 1)
            
            indices = env_ids.to(dtype=torch.int32)
            self.robots.set_world_poses(robot_positions, robot_rotations, indices=indices)
            self.robots.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
            self.robots.set_joint_velocities(torch.zeros((num_resets, self.robots.num_dof), device=self._device), indices=env_ids)
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0
        
        def post_reset(self):
            # ⚠️ 关键修改：在初始化时生成随机旋转（绕Z轴，水平面旋转）
            # 每个环境都有不同的随机朝向，且只在初始化时生成一次
            print("🔄 为每个环境生成随机初始朝向...")
            random_angles = torch.rand(self._num_envs, device=self._device) * 2 * math.pi  # 0 到 2π 的随机角度
            
            # 使用 Isaac Sim 的旋转工具函数生成绕Z轴旋转的四元数
            from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats
            
            # 将Z轴角度转换为欧拉角 [roll, pitch, yaw]，然后转换为四元数
            euler_angles = torch.zeros((self._num_envs, 3), device=self._device)
            euler_angles[:, 2] = random_angles  # yaw (绕Z轴)
            self.initial_rotations = euler_angles_to_quats(euler_angles)
            
            # ⚠️ 关键修复：确保 initial_rotations 在正确的设备上
            if self.initial_rotations.device != self._device:
                self.initial_rotations = self.initial_rotations.to(device=self._device)
            
            # 打印每个环境的初始朝向（角度）
            print("📐 各环境初始朝向（角度）：")
            for i in range(self._num_envs):
                angle_deg = random_angles[i].item() * 180.0 / math.pi
                print(f"   环境 {i}: {angle_deg:.1f}°")
            
            # ⚠️ 关键：在 post_reset 中调用 reset_idx 来设置位置
            # 注意：此时 initial_rotations 已经生成，所以 reset_idx 会使用它们
            indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)
            
            # 标记：旋转将在第一次物理步进后再次设置（确保物理引擎完全初始化后）
            self._rotations_applied = False
        
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
    # ⚠️ 关键：在物理步进过程中，确保旋转不被重置
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
        
        # ⚠️ 关键修复：在 render() 之后，重新应用初始旋转
        # 执行顺序分析：
        # 1. vec_env.set_task() 调用 post_reset() - 此时生成随机旋转并调用 reset_idx() 设置旋转
        # 2. 但物理引擎可能还未完全初始化，所以旋转可能被忽略
        # 3. 第一次物理步进后（第一次 render() 之后），物理引擎完全初始化，此时再设置旋转
        # 4. 之后每个物理步进后都保持旋转
        if hasattr(task, 'initial_rotations'):
            positions, _ = task.robots.get_world_poses()
            all_indices = torch.arange(num_envs, dtype=torch.int32, device=task._device)
            
            # 第一次物理步进后，强制设置旋转（确保物理引擎完全初始化后）
            if not hasattr(task, '_rotations_applied') or not task._rotations_applied:
                task.robots.set_world_poses(positions, task.initial_rotations, indices=all_indices)
                task._rotations_applied = True
                print(f"✅ Step {step}: 首次应用随机旋转（物理引擎已完全初始化）")
            else:
                # 之后每个步进后都保持旋转
                task.robots.set_world_poses(positions, task.initial_rotations, indices=all_indices)
        
        if step % 20 == 0:
            # 获取机器人位置和旋转
            positions, rotations = task.robots.get_world_poses()
            print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
            # 打印旋转信息
            if hasattr(task, 'initial_rotations'):
                for i in range(min(2, num_envs)):  # 只打印前2个环境
                    actual_q = rotations[i].cpu().numpy()
                    expected_q = task.initial_rotations[i].cpu().numpy()
                    actual_angle = 2 * math.atan2(actual_q[3], actual_q[0]) * 180.0 / math.pi
                    expected_angle = 2 * math.atan2(expected_q[3], expected_q[0]) * 180.0 / math.pi
                    print(f"   环境 {i}: 期望角度={expected_angle:.1f}°, 实际角度={actual_angle:.1f}°")
    
    # 获取最终位置和旋转
    final_positions, final_rotations = task.robots.get_world_poses()
    print("\n✅ 测试完成！最终机器人状态：")
    for i in range(num_envs):
        # 从四元数提取Z轴旋转角度
        q = final_rotations[i].cpu().numpy()
        # 四元数转欧拉角（绕Z轴）
        angle_rad = 2 * math.atan2(q[3], q[0])  # 提取Z轴旋转
        angle_deg = angle_rad * 180.0 / math.pi
        print(f"   环境 {i}: Z = {final_positions[i, 2]:.4f} 米, 朝向 = {angle_deg:.1f}°")
    
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
            
            # ⚠️ 关键修复：在无限循环中也要保持旋转
            if hasattr(task, 'initial_rotations'):
                positions, _ = task.robots.get_world_poses()
                all_indices = torch.arange(num_envs, dtype=torch.int32, device=task._device)
                task.robots.set_world_poses(positions, task.initial_rotations, indices=all_indices)
            
            step += 1
            if step % 200 == 0:
                positions, rotations = task.robots.get_world_poses()
                print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
                # 打印旋转信息
                if hasattr(task, 'initial_rotations'):
                    for i in range(min(2, num_envs)):
                        actual_q = rotations[i].cpu().numpy()
                        expected_q = task.initial_rotations[i].cpu().numpy()
                        actual_angle = 2 * math.atan2(actual_q[3], actual_q[0]) * 180.0 / math.pi
                        expected_angle = 2 * math.atan2(expected_q[3], expected_q[0]) * 180.0 / math.pi
                        print(f"   环境 {i}: 期望角度={expected_angle:.1f}°, 实际角度={actual_angle:.1f}°")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()
