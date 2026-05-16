"""
Robot Configuration Module
TB3 机器人配置和设置
"""
from .tb3_config import *
from .tb3_setup import apply_massapi_all_tb3, configure_wheel_joints

__all__ = [
    'TB3_USD', 'WHEEL_RADIUS', 'WHEEL_BASE', 'BASE_MASS', 'COM_X', 'COM_Z',
    'BASE_LINK_DIAGONAL_INERTIA',
    'MAX_V', 'MAX_W',
    'WHEEL_KP', 'WHEEL_KD', 'WHEEL_MAX_EFFORT',
    'apply_massapi_all_tb3', 'configure_wheel_joints'
]
