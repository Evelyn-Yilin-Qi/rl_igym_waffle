"""
最简单的测试脚本：4个环境，机器人能显示并落到地面
"""
import numpy as np
from omni.isaac.kit import SimulationApp

# 启动Isaac Sim
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from pxr import UsdGeom

def main():
    # 创建世界
    world = World(stage_units_in_meters=1.0)
    
    # 机器人USD路径
    robot_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"
    
    # 创建4个环境，每个环境间距4米
    num_envs = 4
    env_spacing = 4.0
    
    # 为每个环境添加机器人
    robot_views = []
    for i in range(num_envs):
        # 计算环境位置
        x_pos = i * env_spacing
        
        # 环境路径
        env_path = f"/World/env_{i}"
        
        # 机器人路径
        robot_prim_path = f"{env_path}/WafflePi"
        
        # 添加机器人到场景
        add_reference_to_stage(usd_path=robot_path, prim_path=robot_prim_path)
        
        # 设置机器人初始位置（稍微高一点，让它落下来）
        from omni.isaac.core.utils.prims import get_prim_at_path
        robot_prim = get_prim_at_path(robot_prim_path)
        if robot_prim:
            # 设置初始位置
            xform = UsdGeom.Xformable(robot_prim)
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set((x_pos, 0.0, 0.2))  # Z=0.2米，让它落下来
    
    # 创建ArticulationView来管理所有机器人
    robot_view = ArticulationView(
        prim_paths_expr="/World/env_.*/WafflePi",
        name="robot_view"
    )
    
    # 初始化世界
    world.scene.add(robot_view)
    world.reset()
    
    print(f"✅ 已创建 {num_envs} 个环境")
    print("🔄 开始物理模拟，让机器人落到地面...")
    
    # 运行物理模拟，让机器人落到地面
    for step in range(200):
        # 更新应用（必须调用，否则窗口会无响应）
        simulation_app.update()
        
        # 执行物理步进
        if world.is_playing():
            world.step(render=False)
        
        if step % 20 == 0:
            # 获取机器人位置
            positions = robot_view.get_world_poses()[0]
            print(f"Step {step}: 机器人Z位置 = {positions[:, 2].cpu().numpy()}")
    
    # 获取最终位置
    final_positions = robot_view.get_world_poses()[0]
    print("\n✅ 测试完成！最终机器人Z位置：")
    for i in range(num_envs):
        print(f"   环境 {i}: Z = {final_positions[i, 2]:.4f} 米")
    
    print("\n按Ctrl+C退出...")
    
    # 保持运行 - 使用正确的API
    # 在Isaac Sim中，必须定期调用simulation_app.update()来保持应用运行
    while simulation_app.is_running():
        # 更新应用（处理窗口事件、渲染等）
        simulation_app.update()
        
        # 如果世界正在播放，则执行物理步进
        if world.is_playing():
            world.step(render=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 退出中...")
    finally:
        simulation_app.close()