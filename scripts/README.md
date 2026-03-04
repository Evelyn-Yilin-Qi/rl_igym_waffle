# Scripts Module

Scripts for running TB3 simulations.

## Files

- **`run_tb3_4env.py`**: Main entry point for 4-environment TB3 simulation
  - Creates 4 environments (empty, box, cylinder, door)
  - Uses modular core infrastructure
  - Provides same functionality as `2_env&TB3_reset_move.py`

## Usage

```bash
cd /workspace/rl_igym_waffle
python scripts/run_tb3_4env.py
```

## Differences from Original Script

The new `run_tb3_4env.py` uses the modular `core/` infrastructure:
- Configuration is centralized in `core/config/`
- Scene logic is in `core/scenes/`
- Robot loading/control is in `core/robot/`
- Utility functions are in `core/utils/`
- Main simulation loop is in `core/simulator.py`

This makes the code:
- More maintainable
- Easier to extend
- Reusable for RL training
