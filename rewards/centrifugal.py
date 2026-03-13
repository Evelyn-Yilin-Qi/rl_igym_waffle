"""
Centrifugal Force Penalty Component
离心力惩罚组件：抑制过大的离心力（提高舒适性）
"""
import numpy as np
from .base import BaseRewardComponent


class CentrifugalPenalty(BaseRewardComponent):
    """离心力惩罚：抑制过大的离心力"""
    def compute(self, v_measured, w_measured, mass=None, **kwargs):
        """
        Args:
            v_measured: (num_envs,) 实际测量的线速度（m/s）
            w_measured: (num_envs,) 实际测量的角速度（rad/s）
            mass: 机器人质量（kg），如果为None则使用配置中的值
        Returns:
            reward: (num_envs,) 离心力惩罚
            info: dict
        """
        if not self.enabled:
            return np.zeros(len(v_measured), dtype=np.float32), {}
        
        v_measured = np.asarray(v_measured, dtype=np.float32)
        w_measured = np.asarray(w_measured, dtype=np.float32)
        
        # 获取质量（优先使用传入参数，否则从配置读取）
        if mass is None:
            mass = self.config.get('params', {}).get('mass', 1.373)  # 默认TB3质量
        
        # 获取惩罚参数
        params = self.config.get('params', {})
        penalty_type = params.get('penalty_type', 'linear')  # 'linear', 'threshold', 'quadratic'
        scale = params.get('scale', 1.0)
        threshold = params.get('threshold', 0.5)  # 离心加速度阈值（m/s²）
        
        # 计算离心加速度（m/s²）
        # 对于差速驱动机器人：a_centrifugal = v * |w|
        centrifugal_accel = v_measured * np.abs(w_measured)
        
        # 计算离心力（N）
        centrifugal_force = mass * centrifugal_accel
        
        # 根据惩罚类型计算惩罚
        reward = np.zeros(len(v_measured), dtype=np.float32)
        
        if penalty_type == 'linear':
            # 线性惩罚：penalty = -scale * centrifugal_force
            reward = -scale * centrifugal_force * self.weight
        
        elif penalty_type == 'threshold':
            # 阈值惩罚：超过阈值时施加惩罚
            exceed_mask = centrifugal_accel > threshold
            reward[exceed_mask] = -scale * (centrifugal_accel[exceed_mask] - threshold) * self.weight
        
        elif penalty_type == 'quadratic':
            # 二次惩罚：penalty = -scale * (centrifugal_accel - threshold)²
            exceed_mask = centrifugal_accel > threshold
            reward[exceed_mask] = -scale * (centrifugal_accel[exceed_mask] - threshold) ** 2 * self.weight
        
        return reward, {
            "centrifugal_pen": reward,
            "centrifugal_accel": centrifugal_accel,
            "centrifugal_force": centrifugal_force
        }
