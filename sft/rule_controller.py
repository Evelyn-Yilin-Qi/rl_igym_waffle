import random
import numpy as np


class RuleBasedController:
    """
    Rule-based controller for SFT data collection.
    Inputs:
      - lidar_ranges (36)
      - user_intent (ego frame: forward, left)
      - base_vel
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
    ):
        self.max_v = max_v
        self.max_w = max_w
        self.safe_dist = safe_dist
        self.collision_dist = collision_dist
        self.v_gain = v_gain
        self.w_gain = w_gain
        self.obstacle_v_gain = obstacle_v_gain
        self.obstacle_w_gain = obstacle_w_gain

        # 36 rays, 10 degrees each
        self.front_sectors = [0]
        self.left_sectors = list(range(1, 9))
        self.right_sectors = list(range(28, 36))
        self.back_sectors = list(range(9, 28))

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
                left = lidar_ranges[1]
                right = lidar_ranges[-1]
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
