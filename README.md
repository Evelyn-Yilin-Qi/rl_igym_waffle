# rl_igym_waffle

Reinforcement learning training for **TurtleBot3 (Waffle Pi)** in **Isaac Sim**: **SFT (supervised fine-tuning) + PPO** across multiple navigation scenes, with support for comparing several LiDAR policy backbones.

---

## Requirements

- **Isaac Sim** (default Python launcher: `/isaac-sim/python.sh`)
- **OmniIsaacGymEnvs** (`OmniIsaacGymEnvs/` submodule; install per upstream instructions)
- Python dependencies in [`requirements.txt`](requirements.txt) (install into Isaac Sim’s bundled Python environment)

---

## Training Scenario Demonstration

## Quick start

From the **repository root**:

```bash
chmod +x run_rl_tb3_scene_models_isaac_sequential.sh
./run_rl_tb3_scene_models_isaac_sequential.sh
```

The script **runs jobs sequentially**, each in its own Isaac process, training different `--model` backbones. By default only the **mixed** scene runs (5 experiments); `box` / `door` / `col` commands are commented out in the script—uncomment them as needed.

If any step fails, `set -e` stops the remaining jobs.

### Custom Isaac Python path

If Isaac is not installed under `/isaac-sim`, set an environment variable before running:

```bash
export ISAAC_PYTHON_LAUNCHER=/path/to/your/python.sh
# or
export ISAAC_PYTHON=/path/to/your/python.sh

./run_rl_tb3_scene_models_isaac_sequential.sh
```

You can also edit `DEFAULT_ISAAC_PY` inside the shell script.

### Single-scene / single-model training

Without the batch script, invoke training directly with Isaac’s Python, for example:

```bash
/isaac-sim/python.sh scripts/rl_tb3_mixed.py --model cnn_lstm_sft
/isaac-sim/python.sh scripts/rl_tb3_box.py   --model simple_fc_sft
/isaac-sim/python.sh scripts/rl_tb3_door.py  --model cnn_gru_sft
/isaac-sim/python.sh scripts/rl_tb3_col.py   --model fc_lstm_sft
```

---

## Directory layout

```
rl_igym_waffle/
├── run_rl_tb3_scene_models_isaac_sequential.sh   # Batch sequential training (recommended)
├── run_rl_tb3_sequential.sh                      # Other sequential-training variant
├── requirements.txt
├── convert_urdf.py                               # URDF conversion utility
│
├── config/                    # YAML (network / algorithm / reward)
│   ├── network_config.yaml
│   ├── algorithm_config.yaml
│   └── reward_config.yaml
│
├── core_network/              # Policy and value architectures
│   ├── __init__.py            # create_policy(), create_value()
│   ├── base.py
│   ├── simple_fc.py
│   ├── simple_fc_sft.py
│   ├── essay_base.py
│   └── lidar_seq_models.py    # CNN+LSTM/GRU, FC+LSTM SFT backbones
│
├── algorithms/                # RL algorithms (PPO, etc.)
│   ├── base.py
│   └── ppo.py
│
├── rewards/                   # Composable reward terms
│   ├── composer.py            # RewardComposer: assembles enabled terms from config
│   ├── obstacle.py, heading.py, goal.py, distance.py, ...
│   └── base.py
│
├── envs/                      # Observations and user intent
│   ├── observations.py        # LiDAR, 44-dim observation assembly
│   └── user_intent.py
│
├── sim/                       # Isaac simulation and scenes
│   ├── robot/                 # TB3 config and spawn (tb3_config.py, tb3_setup.py)
│   └── scenes/                # Scene management (empty / cylinder / door / box)
│
├── process_settings/
│   └── env_setup.py           # EnvironmentSetup: env initialization
│
├── sft/                       # Supervised fine-tuning
│   ├── rule_controller.py     # Rule controller for supervised labels
│   ├── supervised.py          # Supervised training (MSE)
│   ├── config.py
│   └── logging.py
│
├── local_planners/            # Local planning (APF / VO / ORCA, etc.)
│
├── utils/
│   └── config_utils.py        # Read enabled components from YAML
│
├── scripts/                   # Training and test scripts
│   ├── rl_tb3_box.py          # box scene, SFT+PPO
│   ├── rl_tb3_door.py         # door scene
│   ├── rl_tb3_col.py          # cylinder scene
│   ├── rl_tb3_mixed.py        # four scenes, round-robin across envs
│   ├── stage1_rl_sft.py       # Stage1: empty scene, SFT+RL
│   ├── stage2_rl_sft.py       # Stage2: four scenes, SFT+RL
│   ├── sft_rl_tb3_V3_modular.py   # single-env debug / scene switching
│   ├── train_actor.py, sft_data_collection.py, tb3_control.py, ...
│   ├── test/                  # test scripts
│   └── recent_old_version/    # legacy scripts (do not use)
│
├── assets/                    # TurtleBot3 Waffle Pi URDF and meshes
│   └── turtlebot3_waffle_pi/
│
├── data/                      # Training data by scene
│   ├── empty/, cylinder/, door/, box/
│
├── checkpoints/               # Model checkpoints (named by scene and model)
├── runs/                      # TensorBoard logs
├── logs/                      # Plain-text training logs
│
├── cfg/                       # Legacy Hydra config (not used by new scripts)
│
└── OmniIsaacGymEnvs/          # OIGE submodule
```

---

## Usage

### 1. Recommended training flow

| Stage | Script | Description |
|-------|--------|-------------|
| Batch comparison | `run_rl_tb3_scene_models_isaac_sequential.sh` | Sequential runs over `--model`; one Isaac process per job |
| Stage1 | `scripts/stage1_rl_sft.py` | **Empty scene** only, SFT + RL (default 16 envs; use 4/16/32/64 as needed) |
| Stage2 | `scripts/stage2_rl_sft.py` | **Four scenes**, SFT + RL |
| Single-env debug | `scripts/sft_rl_tb3_V3_modular.py` | One env; switch scene in code (see `main` comments) |

> **Note:** Stage1/Stage2 have been smoke-tested only; performance has not been systematically evaluated. For Stage changes, prefer editing the scripts above.

### 2. Scenes and scripts

| Scene | Constant / notes | Training script |
|-------|------------------|-----------------|
| Empty | `SCENE_EMPTY` | `stage1_rl_sft.py` |
| Cylinders | `SCENE_CYLINDER` | `rl_tb3_col.py` |
| Door | `SCENE_DOOR` | `rl_tb3_door.py` |
| Boxes | `SCENE_BOX` | `rl_tb3_box.py` |
| All four (cycled) | empty / cylinder / door / box | `rl_tb3_mixed.py`, `stage2_rl_sft.py` |

### 3. Policy backbones (`--model`)

Batch and per-scene scripts accept these names (must match the `core_network` factory):

| `--model` | Description |
|-----------|-------------|
| `simple_fc_sft` | Fully connected + ReLU, 44-dim observations |
| `cnn_lstm_sft` | 1D CNN + LSTM |
| `cnn_gru_sft` | 1D CNN + GRU |
| `fc_lstm_sft` | FC + LSTM |
| `cnn_lstm_sft_nodoor` | CNN+LSTM variant (no door-specific design) |

See [`MODULAR_ARCHITECTURE.md`](MODULAR_ARCHITECTURE.md) for more detail on the modular design.

### 4. Configuration

Prefer YAML before changing code:

| Goal | File |
|------|------|
| Default network architecture | `config/network_config.yaml` |
| PPO and other algorithm params | `config/algorithm_config.yaml` |
| Reward toggles and weights | `config/reward_config.yaml` |

- **New network:** extend `core_network/` and register in `create_policy` / `create_value`.
- **New or updated rewards:** edit `rewards/` components and set `enabled` / `weight` in `reward_config.yaml`.

### 5. Outputs

Artifacts are typically written to:

- **Checkpoints:** `checkpoints/ppo_sft_rl_tb3_v3/{scene}_{model}/` (e.g. `mixed_cnn_lstm_sft`)
- **TensorBoard:** `runs/ppo_sft_rl_tb3_v3/`
- **Text logs:** `logs/`

View training curves with:

```bash
tensorboard --logdir runs/ppo_sft_rl_tb3_v3
```

---

## Other entry points

| File | Purpose |
|------|---------|
| `scripts/rl_tb3_*_models_train.py` | Subprocess runner: all models for one scene |
| `scripts/rl_tb3_matrix_train.py` | Matrix worker training (`--worker --scene --model`) |
| `scripts/rl_tb3_sequential_scene_models_train.py` | Python equivalent of 20 sequential runs (similar to the shell script) |

---

## Development tips

- Quick single-scene checks: use `sft_rl_tb3_V3_modular.py` (one env; switch scene in code).
- Do not build on `scripts/test/` or `scripts/recent_old_version/`.
- `cfg/WaffleDrive.yaml` is for legacy scripts only; new flows use `config/*.yaml`.
- Set parallel env count in each `scripts/*.py` `main()`; use a **multiple of 4** for mixed training (envs are assigned round-robin across four scene types).

---

## License

Follow the license terms of Isaac Sim, OmniIsaacGymEnvs, and any other bundled submodules.
