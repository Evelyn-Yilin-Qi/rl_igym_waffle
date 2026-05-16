"""
Reward Functions Module
奖励函数模块
"""
from .base import BaseRewardComponent
from .obstacle import ObstaclePenalty
from .heading import HeadingPenalty
from .smoothness import SmoothnessPenalty
from .static import StaticPenalty
from .centrifugal import CentrifugalPenalty
from .distance import DistanceReward
from .goal import GoalReward
from .composer import RewardComposer

__all__ = [
    'BaseRewardComponent',
    'ObstaclePenalty',
    'HeadingPenalty',
    'SmoothnessPenalty',
    'StaticPenalty',
    'CentrifugalPenalty',
    'DistanceReward',
    'GoalReward',
    'RewardComposer',
]
