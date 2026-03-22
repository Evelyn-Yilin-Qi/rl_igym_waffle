import numpy as np


def print_phase_status(train_mode, step_count, supervised_steps, supervised_obs_count, update_count, total_epochs, avg_reward):
    if train_mode == "supervised":
        print("\n=== 监督学习数据收集 ===")
        print(
            f"当前阶段: SUPERVISED | step={step_count}/{supervised_steps} | "
            f"已收集样本: {supervised_obs_count}"
        )
    else:
        print("\n=== PPO策略更新 ===")
        print(
            f"当前阶段: PPO | step={step_count} | "
            f"update={update_count}/{total_epochs} | 最近100步平均奖励: {avg_reward:.2f}"
        )


def print_reward_breakdown(reward_info, env_idx=0):
    """
    Unified reward logging: print directly from reward_info (no manual recomputation).
    """
    reward_name_map = {
        "distance_reward": "距离奖励",
        "goal_reward": "目标奖励",
        "static_pen": "静止惩罚",
        "obstacle_pen": "障碍物惩罚",
        "collision_pen": "碰撞惩罚",
        "heading_pen": "航向惩罚",
        "smooth_pen": "动作平滑惩罚",
    }
    preferred_order = [
        "distance_reward",
        "goal_reward",
        "static_pen",
        "obstacle_pen",
        "collision_pen",
        "heading_pen",
        "smooth_pen",
    ]

    def _reward_scalar(v):
        arr = np.asarray(v)
        if arr.size == 0:
            return 0.0
        flat = arr.reshape(-1)
        pick = min(env_idx, flat.shape[0] - 1)
        return float(flat[pick])

    printed = set()
    for key in preferred_order:
        if key in reward_info:
            zh_name = reward_name_map.get(key, key)
            print(f"{zh_name} ({key}): {_reward_scalar(reward_info[key]):.4f}")
            printed.add(key)

    for key, value in reward_info.items():
        if key in printed:
            continue
        zh_name = reward_name_map.get(key, key)
        print(f"{zh_name} ({key}): {_reward_scalar(value):.4f}")
