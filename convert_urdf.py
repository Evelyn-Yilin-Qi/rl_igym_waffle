"""
Isaac Sim 无头 (Headless) URDF 转换脚本
将 Waffle Pi 的 URDF 和 Meshes 自动打包为 USD 三维资产
"""
from omni.isaac.kit import SimulationApp

# 1. 启动纯后台仿真引擎 (不加载任何渲染和窗口，速度极快)
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
from omni.isaac.urdf import _urdf

# 2. 设置输入输出路径 (请确保与你的实际路径完全一致)
urdf_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/turtlebot3_waffle_pi.urdf"
dest_usd_path = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

# 3. 获取 URDF 导入接口
urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()

# ==========================================
# 4. 关键物理属性配置
# ==========================================
import_config.fix_base = False               # 极其重要！移动机器人绝不能固定底盘在半空
import_config.import_inertia_tensor = True   # 导入你在 URDF 里写的惯性张量 (inertia)
import_config.make_default_prim = True       # 将小车设为 USD 的默认根节点

print(f"⏳ 正在后台解析 URDF 模型: {urdf_path} ...")
print(f"⚙️ 正在打包 Meshes 材质与碰撞体 ...")

# 5. 执行转换指令
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path=dest_usd_path
)

print(f"✅ 转换大功告成！")
print(f"📁 原生资产已生成: {dest_usd_path}")

# 6. 优雅关闭引擎
simulation_app.close()