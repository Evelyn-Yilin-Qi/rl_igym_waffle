import random
from typing import FrozenSet, Tuple

import numpy as np


def _sector_index_sets(num_rays: int) -> Tuple[FrozenSet[int], FrozenSet[int], FrozenSet[int], FrozenSet[int]]:
    """
    Mirror the original 36-beam layout (10° per bin): 1 front + 8 left + 19 back + 8 right.
    Map bin edges onto `num_rays` evenly spaced rays (angle_i = 2π·i/n, i=0 is +x forward).
    """
    n = int(num_rays)
    if n < 1:
        raise ValueError("num_rays must be >= 1")
    scale = n / 36.0

    def span(lo: float, hi: float) -> FrozenSet[int]:
        i0 = int(np.floor(lo * scale))
        i1 = int(np.ceil(hi * scale))
        i0 = max(0, min(i0, n))
        i1 = max(0, min(i1, n))
        if i1 <= i0:
            return frozenset()
        return frozenset(range(i0, i1))

    front = span(0.0, 1.0)
    left = span(1.0, 9.0)
    back = span(9.0, 28.0)
    right = span(28.0, 36.0)
    return front, left, back, right


def _front_gap_compare_indices(num_rays: int) -> Tuple[int, int]:
    """~±45° from +x (same role as old lidar_ranges[1] vs [-1] on 36-beam layout)."""
    n = int(num_rays)
    two_pi_over_n = 2.0 * np.pi / n
    i_left = int(round((np.pi / 4.0) / two_pi_over_n)) % n
    i_right = int(round((7.0 * np.pi / 4.0) / two_pi_over_n)) % n
    return i_left, i_right


class RuleBasedController:
    """
    Rule-based controller for SFT data collection (any LiDAR count; geometry matches 36-ray sectors).
    Inputs:
      - lidar_ranges (num_rays,) — same ordering as compute_lidar_ranges
      - user_intent (ego frame: forward ux, left uy)
      - base_vel: normalized [v_norm, w_norm] in [-1, 1]
    """

    def __init__(
        self,
        max_v=0.5,
        max_w=1.0,
        safe_dist=1.0,
        collision_dist=0.2,
        v_gain=1.0,
        w_gain=1.5,
        obstacle_v_gain=1.0,
        obstacle_w_gain=2.0,
        num_rays: int = 36,
    ):
        self.max_v = max_v
        self.max_w = max_w
        self.safe_dist = safe_dist
        self.collision_dist = collision_dist
        self.v_gain = v_gain
        self.w_gain = w_gain
        self.obstacle_v_gain = obstacle_v_gain
        self.obstacle_w_gain = obstacle_w_gain
        self.num_rays = int(num_rays)

        self.front_sectors, self.left_sectors, self.back_sectors, self.right_sectors = _sector_index_sets(
            self.num_rays
        )
        self._idx_front_left_clear, self._idx_front_right_clear = _front_gap_compare_indices(self.num_rays)

    def get_obstacle_direction(self, lidar_ranges):
        min_dist = np.min(lidar_ranges)
        min_idx = int(np.argmin(lidar_ranges))
        if min_idx in self.front_sectors and min_dist < self.safe_dist:
            return "front", min_dist
        if min_idx in self.left_sectors and min_dist < self.safe_dist:
            return "left", min_dist
        if min_idx in self.right_sectors and min_dist < self.safe_dist:
            return "right", min_dist
        if min_dist < self.safe_dist:
            return "back", min_dist
        return None, min_dist

    def compute_velocity_commands(self, lidar_ranges, user_intent, base_vel):
        lidar_ranges = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
        if lidar_ranges.shape[0] != self.num_rays:
            raise ValueError(
                f"lidar_ranges length {lidar_ranges.shape[0]} != RuleBasedController.num_rays ({self.num_rays})"
            )

        obstacle_side, min_dist = self.get_obstacle_direction(lidar_ranges)
        if np.linalg.norm(user_intent) < 1e-6:
            target_angle = 0.0
        else:
            target_angle = np.arctan2(user_intent[1], user_intent[0])

        if min_dist < self.collision_dist:
            v_cmd = 0.0
            if obstacle_side == "left":
                w_cmd = -self.max_w * 0.8
            elif obstacle_side == "right":
                w_cmd = self.max_w * 0.8
            elif obstacle_side == "front":
                w_cmd = self.max_w * np.sign(np.random.uniform(-1, 1))
            else:
                w_cmd = 0.0
        elif obstacle_side is not None:
            dist_ratio = (min_dist - self.collision_dist) / (self.safe_dist - self.collision_dist)
            if obstacle_side == "left":
                w_cmd = np.clip(self.obstacle_w_gain * (-2.0 + target_angle / 2), -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            elif obstacle_side == "right":
                w_cmd = np.clip(self.obstacle_w_gain * (2.0 + target_angle / 2), -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            elif obstacle_side == "front":
                left = float(lidar_ranges[self._idx_front_left_clear])
                right = float(lidar_ranges[self._idx_front_right_clear])
                w_cmd = 1.0 if left > right else -1.0
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            else:
                w_cmd = np.clip(self.w_gain * target_angle, -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
        else:
            intent_magnitude = np.linalg.norm(user_intent)
            v_cmd = np.clip(self.v_gain * intent_magnitude, 0, self.max_v)
            w_cmd = np.clip(self.w_gain * target_angle, -self.max_w, self.max_w)

        v_cmd = 0.8 * v_cmd + 0.2 * base_vel[0]
        w_cmd = 0.8 * w_cmd + 0.2 * base_vel[1]
        v_cmd = np.clip(v_cmd, 0, self.max_v)
        w_cmd = np.clip(w_cmd, -self.max_w, self.max_w)
        return v_cmd, w_cmd
