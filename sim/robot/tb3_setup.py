"""
TB3 Robot Setup Functions
TB3 机器人加载和配置函数
"""
import omni.usd
from pxr import UsdPhysics, Gf
from .tb3_config import BASE_MASS, COM_X, COM_Z


def apply_massapi_all_tb3(base_mass=BASE_MASS, com_x=COM_X, com_z=COM_Z):
    """
    为所有 TB3 机器人应用 MassAPI（质量和质心配置）
    
    Args:
        base_mass: 底盘质量 (kg)
        com_x: 质心 X 偏移 (m)
        com_z: 质心 Z 偏移 (m)
    """
    stage = omni.usd.get_context().get_stage()
    base_cnt = 0
    imu_cnt = 0

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        name = prim.GetName()

        if name == "a__namespace_base_link":
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(float(base_mass))
            api.CreateCenterOfMassAttr(Gf.Vec3f(float(com_x), 0.0, float(com_z)))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.02, 0.02, 0.02))
            base_cnt += 1

        elif name == "a__namespace_imu_link":
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(0.01)
            api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
            imu_cnt += 1

    if base_cnt > 0:
        print(f"[INFO] MassAPI applied: base_link={base_cnt}, imu_link={imu_cnt}")


def configure_wheel_joints(robots, wheel_kp=None, wheel_kd=None, wheel_max_effort=None):
    """
    配置轮子关节参数（KP, KD, max_effort）
    
    Args:
        robots: ArticulationView 对象
        wheel_kp: 位置增益（默认使用 tb3_config.WHEEL_KP）
        wheel_kd: 速度增益（默认使用 tb3_config.WHEEL_KD）
        wheel_max_effort: 最大力矩（默认使用 tb3_config.WHEEL_MAX_EFFORT）
    
    Returns:
        left_idx, right_idx: 左右轮 DOF 索引
    """
    from .tb3_config import WHEEL_KP, WHEEL_KD, WHEEL_MAX_EFFORT
    
    if wheel_kp is None:
        wheel_kp = WHEEL_KP
    if wheel_kd is None:
        wheel_kd = WHEEL_KD
    if wheel_max_effort is None:
        wheel_max_effort = WHEEL_MAX_EFFORT
    
    # 查找左右轮 DOF 索引
    dof_names = robots.dof_names
    left_idx = None
    right_idx = None
    
    for i, n in enumerate(dof_names):
        s = n.lower()
        if left_idx is None and "wheel_left_joint" in s:
            left_idx = i
        if right_idx is None and "wheel_right_joint" in s:
            right_idx = i
    
    if left_idx is None or right_idx is None:
        print(f"[WARN] Could not find wheel joints. dof_names: {dof_names}")
        return None, None
    
    # 设置关节增益
    kps, kds = robots.get_gains()
    kps[:, left_idx] = wheel_kp
    kps[:, right_idx] = wheel_kp
    kds[:, left_idx] = wheel_kd
    kds[:, right_idx] = wheel_kd
    robots.set_gains(kps=kps, kds=kds)
    
    # 设置最大力矩
    try:
        max_eff = robots.get_max_efforts()
        if max_eff is not None:
            max_eff[:, left_idx] = wheel_max_effort
            max_eff[:, right_idx] = wheel_max_effort
            robots.set_max_efforts(max_eff)
    except Exception as e:
        print(f"[WARN] set_max_efforts failed: {e}")
    
    return left_idx, right_idx
