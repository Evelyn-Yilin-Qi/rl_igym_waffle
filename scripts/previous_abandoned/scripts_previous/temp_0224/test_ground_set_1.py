"""
测试脚本：验证4个独立环境是否正确创建
每个环境中加载一个TB3小车，不做任何操作，仅验证环境设置
"""
# ⛔️ 绝对不能在文件顶部 import 任何自定义的 envs 或 omni 模块！
# 这些模块必须在 Isaac Sim 启动后的上下文中导入
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf


def create_simple_waffle_task():
    """创建SimpleWaffleTask类（延迟导入以避免模块加载问题）"""
    # 在这里导入，确保在Isaac Sim上下文中
    from omniisaacgymenvs.tasks.base.rl_task import RLTask
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    
    class SimpleWaffleTask(RLTask):
        """最简单的TB3小车测试任务，仅用于验证环境创建"""
        
        def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        # 最小化的观测和动作空间
        self._num_observations = 1
        self._num_actions = 2
        
        # 调用父类初始化
        super().__init__(name, env, offset)
    
    def set_up_scene(self, scene) -> None:
        """设置场景：在每个环境中加载TB3小车"""
        robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
        
        # 在默认环境路径下添加机器人USD引用
        # default_zero_env_path 是基类提供的，指向 "/World/envs/env_0"
        # 框架会自动为每个环境复制这个结构
        add_reference_to_stage(
            usd_path=robot_path,
            prim_path=self.default_zero_env_path + "/WafflePi"
        )
        
        # 调用父类方法完成场景设置
        super().set_up_scene(scene)
        
        # 创建ArticulationView来管理所有环境中的机器人
        self.robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/WafflePi",
            name="waffle_view"
        )
        scene.add(self.robots)
    
    def get_observations(self) -> dict:
        """获取观测：返回最简单的观测（仅用于测试）"""
        # 创建一个简单的观测缓冲区
        # 这里只返回一个占位值，因为我们只是测试环境创建
        self.obs_buf = torch.zeros((self._num_envs, self._num_observations), 
                                   device=self._device)
        
        return {
            self.name: {
                "obs": self.obs_buf
            }
        }
    
    def pre_physics_step(self, actions) -> None:
        """物理步进前处理：这里不做任何操作，让小车保持静止"""
        # 检查是否需要重置环境
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # 不应用任何动作，让小车保持静止
        # 这是测试脚本，我们只想验证环境是否正确创建
    
    def reset_idx(self, env_ids):
        """重置指定环境：将小车重置到初始位置"""
        num_resets = len(env_ids)
        
        # 获取这些环境的基础位置
        env_pos = self._env_pos[env_ids]
        
        # 设置机器人位置：在每个环境的原点上方
        robot_positions = env_pos.clone()
        robot_positions[:, 2] += 0.05  # 稍微抬高，避免碰撞地面
        
        # 设置初始旋转（四元数：w, x, y, z）
        robot_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        robot_rotations = robot_rotations.repeat(num_resets, 1)
        
        # 应用重置
        indices = env_ids.to(dtype=torch.int32)
        self.robots.set_world_poses(robot_positions, robot_rotations, indices=indices)
        self.robots.set_velocities(
            torch.zeros((num_resets, 6), device=self._device),
            indices=indices
        )
        self.robots.set_joint_velocities(
            torch.zeros((num_resets, self.robots.num_dof), device=self._device),
            indices=indices
        )
        
        # 重置缓冲区
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def post_reset(self):
        """后重置处理：初始化所有环境"""
        # 重置所有环境
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    
    def calculate_metrics(self) -> None:
        """计算奖励：测试脚本中不需要奖励"""
        self.rew_buf[:] = 0.0
    
        def is_done(self) -> None:
            """判断是否完成：测试脚本中永远不完成"""
            # 不设置任何重置条件，让环境持续运行
            self.reset_buf[:] = 0
    
    return SimpleWaffleTask


def create_test_config():
    """创建测试用的最小配置"""
    config = OmegaConf.create({
        "task": {
            "name": "SimpleWaffle",
            "physics_engine": "physx",
            "env": {
                "numEnvs": 4,
                "envSpacing": 4.0,
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
        "headless": False,  # 设置为False以便观察
    })
    return config


def main():
    """主函数：创建并运行测试环境"""
    print("=" * 60)
    print("测试脚本：验证4个独立环境的创建")
    print("=" * 60)
    
    # ⚠️ 关键：在main函数内部导入omni相关模块
    # 这确保在Isaac Sim的上下文中加载
    print("\n[0/5] 导入OmniIsaacGym模块...")
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    
    # 创建SimpleWaffleTask类（延迟导入）
    SimpleWaffleTask = create_simple_waffle_task()
    
    # 创建配置
    print("[1/5] 创建测试配置...")
    cfg = create_test_config()
    
    # 创建向量化环境
    print("[2/5] 创建向量化环境管理器...")
    vec_env = VecEnvRLGames(
        headless=cfg.headless,
        sim_device=cfg.sim_device
    )
    
    # 创建SimConfig对象
    print("[3/5] 创建仿真配置对象...")
    sim_config = SimConfig(cfg)
    
    # 创建任务
    print("[4/5] 创建测试任务...")
    task = SimpleWaffleTask(
        name="SimpleWaffle",
        sim_config=sim_config,
        env=vec_env
    )
    
    # 设置任务并初始化仿真
    print("[5/5] 初始化仿真环境...")
    vec_env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=True
    )
    
    print("\n✅ 环境创建完成！")
    print(f"   环境数量: {task._num_envs}")
    print(f"   环境间距: {task._env_spacing}")
    
    # 获取初始观测，验证环境正常工作
    print("\n[验证] 获取初始观测...")
    obs = task.get_observations()
    print(f"   观测形状: {obs[task.name]['obs'].shape}")
    
    # 检查小车位置
    if hasattr(task, 'robots'):
        robot_positions, _ = task.robots.get_world_poses()
        print(f"   小车位置 (前2个环境):")
        for i in range(min(2, task._num_envs)):
            print(f"      Env {i}: ({robot_positions[i][0]:.2f}, {robot_positions[i][1]:.2f}, {robot_positions[i][2]:.2f})")
    
    print("\n💡 提示：")
    print("   - 你应该能在仿真器中看到4个独立的TB3小车")
    print("   - 每个小车应该位于不同的位置（间距4.0米）")
    print("   - 小车应该保持静止（没有应用任何动作）")
    print("\n按 Ctrl+C 退出...")
    
    # 运行简单的仿真循环
    # 注意：在OmniIsaacGym中，物理仿真会自动运行
    # 我们只需要定期调用相关方法，让环境保持活跃
    try:
        import time
        step = 0
        
        while True:
            # 创建零动作（不移动小车）
            actions = torch.zeros((task._num_envs, task._num_actions), 
                                 device=task._device)
            
            # 调用pre_physics_step（虽然不应用动作，但这是标准流程）
            task.pre_physics_step(actions)
            
            # 手动更新进度缓冲区（防止超时重置）
            # TODO: 需要确认OmniIsaacGym是否自动处理progress_buf
            task.progress_buf += 1
            
            # 计算奖励和完成状态（虽然我们不需要，但保持完整性）
            task.calculate_metrics()
            task.is_done()
            
            # 渲染（如果headless=False，这会更新视图）
            if hasattr(vec_env, 'render'):
                vec_env.render()
            
            step += 1
            if step % 200 == 0:
                print(f"仿真步数: {step}")
            
            # 稍微延迟，避免占用过多CPU
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("清理资源...")
        try:
            if hasattr(vec_env, 'close'):
                vec_env.close()
        except:
            pass


if __name__ == "__main__":
    main()
