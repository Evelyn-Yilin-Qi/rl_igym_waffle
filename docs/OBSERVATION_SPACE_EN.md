# Observation Space Detailed Documentation

## Overview

The WaffleDrive RL environment's observation space is a **44-dimensional vector** composed of four components:

```
Observation = [LiDAR(36D) | User Input(2D) | Base Velocity(2D) | Action History(4D)]
```

---

## 1. LiDAR Distance Measurements (Dimensions 0-35, 36 dimensions)

### Basic Information
- **Dimension Indices**: `[0:36]`
- **Data Type**: `float32`
- **Value Range**: `[0.0, 1.0]`
- **Physical Unit**: Normalized distance (original unit: meters)

### Normalization Formula

```python
lidar_normalized = clip(lidar_ranges / lidar_max_range, 0.0, 1.0)
```

Where:
- `lidar_ranges`: Raw LiDAR distance measurements (unit: meters)
- `lidar_max_range`: Maximum measurement range = **3.0 meters** (configured in `cfg/task/WaffleDrive.yaml`)

### Physical Meaning

- **Number of Rays**: 36 rays
- **Scanning Angle**: Uniformly distributed in `[0, 2π)` range, i.e., one ray every 10°
- **Ray Direction**: Relative to robot's local coordinate frame, 0° corresponds to robot's forward direction (positive x-axis)
- **Measurement Method**: Uses `omni.physx.get_physx_scene_query_interface().raycast_closest()` for real ray casting

### Value Interpretation

| Value | Physical Meaning | Description |
|-------|------------------|-------------|
| `0.0` | Distance = 0 m | Ray directly hits obstacle or boundary wall |
| `0.0 < x < 1.0` | Distance = x × 3.0 m | Obstacle detected within measurement range |
| `1.0` | Distance ≥ 3.0 m | Beyond maximum range, no obstacle detected |

### Examples

- `lidar[0] = 0.5` → Obstacle 1.5 m ahead of robot
- `lidar[9] = 1.0` → No obstacle within 3 m in the 90° right direction
- `lidar[18] = 0.0` → Direct collision behind robot

---

## 2. User Input User Intent Vector (Dimensions 36-37, 2 dimensions)

### Basic Information
- **Dimension Indices**: `[36:38]`
- **Data Type**: `float32`
- **Value Range**: `ux, uy ∈ [-1, 1]`, and **√(ux² + uy²) = 1** (unit vector)
- **Physical Meaning**: Unit direction vector from robot's current position to goal position (in robot's local coordinate frame)

### Calculation Formula

```python
# 1. Compute direction vector in environment coordinate frame
robot_local = robot_position - env_origin
goal_local = goal_position - env_origin
dir_vec = goal_local[:2] - robot_local[:2]  # (N, 2)

# 2. Normalize to unit vector
dist = ||dir_vec||
norm = max(dist, 1e-8)  # Avoid division by zero
dir_unit = dir_vec / norm  # (ux, uy)

# 3. Transform to robot's local coordinate frame
# Rotate based on robot's current orientation (yaw angle)
ux, uy = rotate_to_robot_frame(dir_unit, robot_yaw)
```

### Physical Meaning

- **Coordinate Frame**: Robot's local coordinate frame (x-axis: forward, y-axis: left)
- **Unit Vector**: Magnitude is always 1, only represents direction, not distance
- **Direction Definition**:
  - `ux > 0`: Goal is ahead of robot
  - `ux < 0`: Goal is behind robot
  - `uy > 0`: Goal is to the left of robot
  - `uy < 0`: Goal is to the right of robot

### Value Interpretation

| Value Combination | Physical Meaning | Description |
|-------------------|------------------|-------------|
| `ux = 1.0, uy = 0.0` | Goal directly ahead | Robot only needs to move forward |
| `ux = 0.0, uy = 1.0` | Goal directly to the left | Robot needs to turn left 90° |
| `ux = -1.0, uy = 0.0` | Goal directly behind | Robot needs to turn around |
| `ux = 0.707, uy = 0.707` | Goal at 45° left-front | Robot needs to move forward and turn left |
| `ux = -0.5, uy = -0.866` | Goal at 120° right-rear | Robot needs to move backward and turn right |

### Constraints

- **Unit Vector Constraint**: `ux² + uy² = 1.0` (negative values allowed)
- **Negative Values Allowed**: Both `ux` and `uy` can be negative, indicating goal is behind or to the right of robot

---

## 3. Base Velocity (Dimensions 38-39, 2 dimensions)

### Basic Information
- **Dimension Indices**: `[38:40]`
- **Data Type**: `float32`
- **Value Range**: `[-1.0, 1.0]`
- **Physical Meaning**: Measured linear and angular velocities at current time step (normalized)

### Normalization Formula

```python
# 1. Extract from robot velocity tensor
v = velocities[:, 0]  # Linear velocity v (robot local frame x-direction)
w = velocities[:, 5]  # Angular velocity w (around z-axis)

# 2. Normalize
v_norm = clip(v / max_v, -1.0, 1.0)
w_norm = clip(w / max_w, -1.0, 1.0)
```

Where:
- `max_v = 0.5 m/s` (maximum linear velocity, configured in `cfg/task/WaffleDrive.yaml`)
- `max_w = 1.0 rad/s` (maximum angular velocity, configured in `cfg/task/WaffleDrive.yaml`)

### Physical Meaning

#### v (Linear Velocity)
- **Physical Unit**: meters/second (m/s)
- **Coordinate Frame**: Robot's local coordinate frame x-direction (forward direction)
- **Value Interpretation**:
  - `v_norm > 0`: Robot moving forward
  - `v_norm < 0`: Robot moving backward
  - `v_norm = 0`: Robot stationary (in linear velocity direction)

#### w (Angular Velocity)
- **Physical Unit**: radians/second (rad/s)
- **Rotation Axis**: Around z-axis (vertical upward)
- **Value Interpretation**:
  - `w_norm > 0`: Counterclockwise rotation (viewed from above)
  - `w_norm < 0`: Clockwise rotation (viewed from above)
  - `w_norm = 0`: No rotation

### Value Interpretation

| Value | Physical Velocity | Description |
|-------|-------------------|-------------|
| `v_norm = 1.0` | `v = 0.5 m/s` | Maximum forward speed |
| `v_norm = -1.0` | `v = -0.5 m/s` | Maximum backward speed |
| `v_norm = 0.0` | `v = 0.0 m/s` | Stationary |
| `w_norm = 1.0` | `w = 1.0 rad/s` | Maximum counterclockwise angular velocity (≈ 57.3°/s) |
| `w_norm = -1.0` | `w = -1.0 rad/s` | Maximum clockwise angular velocity |
| `w_norm = 0.0` | `w = 0.0 rad/s` | No rotation |

### Examples

- `[v_norm=0.6, w_norm=0.3]` → Robot moving forward at 0.3 m/s while rotating counterclockwise at 0.3 rad/s
- `[v_norm=-0.2, w_norm=-0.8]` → Robot moving backward at 0.1 m/s while rotating clockwise at 0.8 rad/s
- `[v_norm=0.0, w_norm=1.0]` → Robot rotating in place counterclockwise

### Important Notes

- **Measured vs Commanded**: Base Velocity represents **actual measured velocities**, not commands sent to the robot. Due to physical limitations, friction, etc., measured values may differ from commanded values.
- **Source**: Obtained from `robots.get_velocities()`, which returns `[vx, vy, vz, wx, wy, wz]`, where `vx` is linear velocity and `wz` is angular velocity.

---

## 4. Action History (Dimensions 40-43, 4 dimensions)

### Basic Information
- **Dimension Indices**: `[40:44]`
- **Data Type**: `float32`
- **Value Range**: `[-1.0, 1.0]`
- **Physical Meaning**: Measured linear and angular velocities at historical time steps (normalized)

### Normalization Formula

```python
# 1. Get measured velocities at historical time steps (physical values)
v_t_minus_1 = prev_measured_v  # Measured linear velocity at t-1
w_t_minus_1 = prev_measured_w  # Measured angular velocity at t-1
v_t_minus_2 = action_history[:, 2]  # Measured linear velocity at t-2 (from history)
w_t_minus_2 = action_history[:, 3]  # Measured angular velocity at t-2 (from history)

# 2. Normalize
action_history_normalized[:, 0] = clip(v_t_minus_1 / max_v, -1.0, 1.0)  # v_t-1
action_history_normalized[:, 1] = clip(w_t_minus_1 / max_w, -1.0, 1.0)  # w_t-1
action_history_normalized[:, 2] = clip(v_t_minus_2 / max_v, -1.0, 1.0)  # v_t-2
action_history_normalized[:, 3] = clip(w_t_minus_2 / max_w, -1.0, 1.0)  # w_t-2
```

Where:
- `max_v = 0.5 m/s`
- `max_w = 1.0 rad/s`

### Dimension Composition

```
Action History = [v_t-1, w_t-1, v_t-2, w_t-2]
```

- **`[40]`: v_t-1** - Measured linear velocity at t-1 (previous time step)
- **`[41]`: w_t-1** - Measured angular velocity at t-1 (previous time step)
- **`[42]`: v_t-2** - Measured linear velocity at t-2 (two steps ago)
- **`[43]`: w_t-2** - Measured angular velocity at t-2 (two steps ago)

### Physical Meaning

- **Time Steps**: Assuming control frequency of 40 Hz, time step `dt = 0.025 s`
  - `t-1`: 25 milliseconds ago
  - `t-2`: 50 milliseconds ago
- **Measured Values**: Action History stores **measured velocities at historical time steps**, not commanded actions. This is consistent with Base Velocity, both are actual measured velocity values.

### Value Interpretation

Same as Base Velocity, each value has the following meaning:

| Dimension | Value | Physical Meaning |
|-----------|-------|------------------|
| `v_t-1` | `1.0` | Moving forward at maximum speed at previous time step |
| `v_t-1` | `-1.0` | Moving backward at maximum speed at previous time step |
| `w_t-1` | `1.0` | Rotating counterclockwise at maximum angular velocity at previous time step |
| `w_t-1` | `-1.0` | Rotating clockwise at maximum angular velocity at previous time step |

### Update Logic

```python
# At each time step:
# 1. Get current measured velocities
current_v = robot_velocities[:, 0]
current_w = robot_velocities[:, 5]

# 2. Update history (sliding window)
action_history[:, 2] = action_history[:, 0]  # t-2 = old t-1
action_history[:, 3] = action_history[:, 1]  # t-2 = old t-1
action_history[:, 0] = prev_measured_v       # t-1 = measured value from previous step
action_history[:, 1] = prev_measured_w       # t-1 = measured value from previous step

# 3. Save current measured values for next step
prev_measured_v = current_v
prev_measured_w = current_w
```

### Important Notes

- **Measured vs Commanded**: Action History stores **historical measured velocities**, not historical commanded actions. This is consistent with Base Velocity, both are actual measured velocity values.
- **Purpose**: Provides temporal sequence information about velocities, helping the RL model understand the robot's motion trends and acceleration information.

---

## Complete Observation Vector Example

Suppose the complete observation vector at a certain time step is:

```python
obs = [
    # LiDAR (36 dimensions)
    0.8, 0.9, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1,  # Front and left
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # Rear
    1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,  # Right
    0.2, 0.1, 0.0, 0.1, 0.2, 0.3,                       # Back to front
    
    # User Input (2 dimensions)
    0.707, 0.707,  # Goal at 45° left-front
    
    # Base Velocity (2 dimensions)
    0.6, 0.3,  # Currently moving forward at 0.3 m/s, rotating counterclockwise at 0.3 rad/s
    
    # Action History (4 dimensions)
    0.5, 0.2,  # t-1: 0.25 m/s forward, 0.2 rad/s counterclockwise
    0.4, 0.1   # t-2: 0.2 m/s forward, 0.1 rad/s counterclockwise
]
```

---

## Summary

| Component | Dimensions | Value Range | Physical Meaning | Normalization Basis |
|-----------|------------|-------------|------------------|---------------------|
| **LiDAR** | 36 | [0, 1] | Obstacle distances in 36 directions | `max_range = 3.0 m` |
| **User Input** | 2 | [-1, 1], magnitude=1 | Goal direction unit vector | Unit vector |
| **Base Velocity** | 2 | [-1, 1] | Current measured velocities (v, w) | `max_v=0.5 m/s`, `max_w=1.0 rad/s` |
| **Action History** | 4 | [-1, 1] | Historical measured velocities (v_t-1, w_t-1, v_t-2, w_t-2) | Same as above |
| **Total** | **44** | - | - | - |

---

## Reference Configuration

Relevant configuration parameters are located in `cfg/task/WaffleDrive.yaml`:

```yaml
robot_limits:
  max_v: 0.5  # Maximum linear velocity m/s
  max_w: 1.0  # Maximum angular velocity rad/s

lidar:
  num_rays: 36
  max_range: 3.0  # meters
```

---

## Code Implementation Locations

- **Observation Assembly**: `envs/observations.py` - `assemble_observations()`
- **Base Velocity Extraction**: `envs/observations.py` - `get_base_velocity_from_tensor()`
- **User Input Calculation**: `envs/user_intent.py` - `compute_user_intent_torch()`
- **LiDAR Calculation**: `scripts/2_RL_stage1_test.py` - `compute_lidar_ranges()` (test script) or `envs/igym_waffle_env.py` - `_compute_lidar_ranges()` (RL environment)
