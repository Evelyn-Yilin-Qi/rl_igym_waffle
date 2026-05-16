"""
Classical local navigation: APF (artificial potential field), VO-style speed cap,
and ORCA-style half-plane projection on a 2D holonomic velocity proxy mapped to unicycle (v, w).

Coordinates match the rest of the stack: robot +X forward, +Y left; LiDAR angles
0 .. 2pi from compute_lidar_ranges (ray_dir = [cos(a), sin(a)]).
"""
from __future__ import annotations

import numpy as np

from sim.scenes import wrap_to_pi


def _lidar_ray_dirs(num_rays: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False, dtype=np.float64)
    c = np.cos(angles)
    s = np.sin(angles)
    return np.stack([c, s], axis=1)


class APFVOOrcaController:
    """
    mode:
      - "apf": attractive + LiDAR repulsive resultant -> (v, w)
      - "apf_vo": APF then scale forward speed using VO / time-to-collision style margin
      - "orca": APF gives holonomic reference; static ORCA half-planes (u·v <= b) project
                velocity in the plane, then map to unicycle (v, w)
    """

    def __init__(
        self,
        max_v: float,
        max_w: float,
        k_att: float = 1.2,
        k_rep: float = 0.35,
        rho0: float = 1.8,
        d_safe: float = 0.22,
        robot_radius: float = 0.14,
        vo_tau: float = 2.0,
        vo_margin: float = 0.12,
        orca_iters: int = 24,
        w_align_gain: float = 2.2,
        side_rep_scale: float = 0.42,
        rep_vec_norm_cap: float = 2.8,
        narrow_corridor_lateral_sim_thresh: float = 0.55,
        narrow_lateral_max: float = 1.45,
        vo_half_fov_scale: float = 0.48,
        v_floor_in_narrow_scale: float = 0.22,
        mode: str = "apf_vo",
    ):
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.rho0 = float(rho0)
        self.d_safe = float(d_safe)
        self.robot_radius = float(robot_radius)
        self.vo_tau = float(vo_tau)
        self.vo_margin = float(vo_margin)
        self.orca_iters = int(orca_iters)
        self.w_align_gain = float(w_align_gain)
        self.side_rep_scale = float(side_rep_scale)
        self.rep_vec_norm_cap = float(rep_vec_norm_cap)
        self.narrow_corridor_lateral_sim_thresh = float(narrow_corridor_lateral_sim_thresh)
        self.narrow_lateral_max = float(narrow_lateral_max)
        self.vo_half_fov_scale = float(vo_half_fov_scale)
        self.v_floor_in_narrow_scale = float(v_floor_in_narrow_scale)
        self.mode = str(mode)
        if self.mode not in ("apf", "apf_vo", "orca"):
            raise ValueError(f"Unknown mode {self.mode!r}; use 'apf', 'apf_vo', or 'orca'.")

    def _lateral_clearance_mins(self, lidar_ranges: np.ndarray) -> tuple[float, float, float]:
        """Approx min range on left / right / front sectors (robot +x forward)."""
        n = int(lidar_ranges.shape[0])
        if n < 8:
            d = float(np.min(lidar_ranges))
            return d, d, d
        q = n // 4
        left = float(np.min(lidar_ranges[max(0, n // 4 - q // 2) : min(n, n // 4 + q // 2)]))
        right = float(np.min(lidar_ranges[max(0, 3 * n // 4 - q // 2) : min(n, 3 * n // 4 + q // 2)]))
        w = max(2, n // 12)
        front = float(np.min(np.concatenate([lidar_ranges[:w], lidar_ranges[-w:]])))
        return left, right, front

    def _narrow_corridor(self, lidar_ranges: np.ndarray) -> bool:
        """True near door / corridor: both sides close to wall and similar range."""
        left, right, _ = self._lateral_clearance_mins(lidar_ranges)
        if min(left, right) > self.narrow_lateral_max:
            return False
        return abs(left - right) < self.narrow_corridor_lateral_sim_thresh

    def _apf_force(self, lidar_ranges: np.ndarray, user_intent: np.ndarray) -> np.ndarray:
        n = int(lidar_ranges.shape[0])
        dirs = _lidar_ray_dirs(n)
        F = np.zeros(2, dtype=np.float64)
        ui = np.asarray(user_intent, dtype=np.float64).reshape(2)
        nu = np.linalg.norm(ui)
        u_goal = (ui / nu) if nu > 1e-9 else np.zeros(2, dtype=np.float64)
        if nu > 1e-9:
            F += self.k_att * u_goal

        F_rep = np.zeros(2, dtype=np.float64)
        for i in range(n):
            d = float(lidar_ranges[i])
            if d >= self.rho0 or d < 1e-4:
                continue
            d_eff = max(d, self.d_safe)
            rho = self.rho0
            mag = self.k_rep * (1.0 / d_eff - 1.0 / rho) * (1.0 / (d_eff * d_eff))
            mag = min(mag, 50.0)
            ray = dirs[i]
            ang = float(np.arctan2(ray[1], ray[0]))
            if abs(ang) > (np.pi / 4.0):
                mag *= self.side_rep_scale
            F_rep += mag * (-ray)
        nr = float(np.linalg.norm(F_rep))
        if nr > self.rep_vec_norm_cap and nr > 1e-9:
            F_rep *= self.rep_vec_norm_cap / nr
        F += F_rep
        # Bias along goal in narrow passages (reduces APF null / sideways equilibrium at door)
        if nu > 1e-9 and self._narrow_corridor(lidar_ranges):
            F += 0.85 * self.k_att * u_goal
        return F

    def _force_to_twist(
        self, F: np.ndarray, user_intent: np.ndarray, lidar_ranges: np.ndarray
    ) -> tuple[float, float]:
        ui = np.asarray(user_intent, dtype=np.float64).reshape(2)
        nu = float(np.linalg.norm(ui))
        gx = float(ui[0] / nu) if nu > 1e-9 else 0.0
        gy = float(ui[1] / nu) if nu > 1e-9 else 0.0

        fn = float(np.linalg.norm(F))
        if fn < 1e-8:
            if nu < 1e-9:
                return 0.0, 0.0
            ang_g = float(np.arctan2(gy, gx))
            ang_w = float(wrap_to_pi(ang_g))
            v_cmd = self.max_v * 0.28 * max(0.0, float(np.cos(ang_w)))
            w_cmd = self.max_w * np.clip(self.w_align_gain * ang_w / (np.pi / 3.0), -1.0, 1.0)
            return float(v_cmd), float(w_cmd)

        ang = float(np.arctan2(F[1], F[0]))
        ang_w = float(wrap_to_pi(ang))
        cos_align = max(0.0, float(np.cos(ang_w)))
        v_cmd = self.max_v * cos_align * min(1.0, fn / (self.k_att + 1e-6))
        w_cmd = self.max_w * np.clip(self.w_align_gain * ang_w / (np.pi / 3.0), -1.0, 1.0)

        if self._narrow_corridor(lidar_ranges) and nu > 1e-9 and gx > 0.12:
            v_floor = self.max_v * self.v_floor_in_narrow_scale * gx
            v_cmd = max(float(v_cmd), float(v_floor))
        return float(v_cmd), float(w_cmd)

    def _vo_limit_v(self, lidar_ranges: np.ndarray, v_cmd: float, _w_cmd: float) -> float:
        n = int(lidar_ranges.shape[0])
        dirs = _lidar_ray_dirs(n)
        v_lim = float(v_cmd)
        half_fov = self.vo_half_fov_scale * (np.pi / 2.0)
        for i in range(n):
            d = float(lidar_ranges[i])
            if d >= self.rho0:
                continue
            ray = dirs[i]
            ang = float(np.arctan2(ray[1], ray[0]))
            if abs(ang) > half_fov:
                continue
            denom = max(float(np.cos(ang)), 0.22)
            ttc = (d - self.robot_radius - self.vo_margin) / (v_lim * denom + 1e-6)
            if ttc < self.vo_tau and v_lim > 1e-6:
                v_allow = (d - self.robot_radius - self.vo_margin) / (self.vo_tau * denom + 1e-6)
                v_lim = max(0.0, min(v_lim, v_allow))
        if self._narrow_corridor(lidar_ranges):
            v_lim = max(v_lim, self.max_v * 0.08)
        return v_lim

    def _orca_project_velocity(self, lidar_ranges: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
        """Static-obstacle ORCA-style linear constraints: u_i·v <= b_i with u_i = ray direction."""
        n = int(lidar_ranges.shape[0])
        dirs = _lidar_ray_dirs(n)
        v = np.array([float(v_ref[0]), float(v_ref[1])], dtype=np.float64)
        tau = max(self.vo_tau, 1e-3)
        for _ in range(self.orca_iters):
            max_viol = 0.0
            worst_u = None
            worst_b = None
            for i in range(n):
                d = float(lidar_ranges[i])
                if d >= self.rho0 * 1.1:
                    continue
                u = dirs[i]
                b = (d - self.robot_radius - self.vo_margin) / tau
                slack = float(np.dot(u, v) - b)
                if slack > max_viol:
                    max_viol = slack
                    worst_u = u
                    worst_b = b
            if worst_u is None or max_viol <= 1e-5:
                break
            v = v - (max_viol + 1e-6) * worst_u
        vn = float(np.linalg.norm(v))
        if vn > self.max_v + 1e-6:
            v *= self.max_v / vn
        return v

    def compute_velocity_commands(
        self,
        lidar_ranges: np.ndarray,
        user_intent: np.ndarray,
        base_vel: np.ndarray | None = None,
    ) -> tuple[float, float]:
        _ = base_vel
        ui = np.asarray(user_intent, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(ui)) < 1e-6:
            return 0.0, 0.0

        lidar_ranges = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
        F = self._apf_force(lidar_ranges, ui)

        if self.mode == "orca":
            fn = float(np.linalg.norm(F))
            if fn < 1e-8:
                return self._force_to_twist(np.zeros(2), ui, lidar_ranges)
            v_ref = (F / fn) * min(self.max_v, fn)
            v_proj = self._orca_project_velocity(lidar_ranges, v_ref)
            ang = float(np.arctan2(v_proj[1], v_proj[0] + 1e-9))
            ang_w = float(wrap_to_pi(ang))
            v_cmd = min(self.max_v, float(np.linalg.norm(v_proj))) * max(0.0, float(np.cos(ang_w)))
            w_cmd = self.max_w * np.clip(self.w_align_gain * ang_w / (np.pi / 3.0), -1.0, 1.0)
            nu = float(np.linalg.norm(ui))
            gx = float(ui[0] / nu) if nu > 1e-9 else 0.0
            if self._narrow_corridor(lidar_ranges) and gx > 0.12:
                v_cmd = max(float(v_cmd), self.max_v * self.v_floor_in_narrow_scale * gx)
            return float(v_cmd), float(w_cmd)

        v_cmd, w_cmd = self._force_to_twist(F, ui, lidar_ranges)
        if self.mode == "apf_vo":
            v_cmd = self._vo_limit_v(lidar_ranges, v_cmd, w_cmd)
        return float(v_cmd), float(w_cmd)
