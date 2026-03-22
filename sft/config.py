import os
from omegaconf import OmegaConf


def load_config(project_root, cfg_path):
    """Load config from common candidate paths."""
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend(
            [
                os.path.join(project_root, cfg_path),
                os.path.abspath(cfg_path),
                os.path.join(os.getcwd(), cfg_path),
            ]
        )
    found = None
    for path in candidates:
        if os.path.exists(path):
            found = os.path.abspath(path)
            break
    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")
    return OmegaConf.load(found)


def get_network_params_by_name(network_cfg, network_name):
    """Get network params by component name from network config."""
    components = OmegaConf.to_container(network_cfg.networks.components, resolve=True)
    for comp in components:
        if comp.get("name") == network_name:
            return comp.get("params", {}) or {}
    return {}
