import numpy as np
import torch
import torch.nn as nn


def train_supervised_policy(rl_agent, obs_np, act_np, device, epochs=50, batch_size=2048):
    """Supervised pretraining using current policy mean output."""
    if len(obs_np) == 0:
        print("[SFT] No supervised samples. Skip pretraining.")
        return None

    print(f"[SFT] Start supervised training: samples={len(obs_np)}, epochs={epochs}")
    obs_tensor = torch.FloatTensor(obs_np).to(device)
    act_tensor = torch.FloatTensor(act_np).to(device)
    criterion = nn.MSELoss()
    final_loss = None
    rl_agent.policy.train()

    for epoch in range(epochs):
        indices = np.arange(len(obs_tensor))
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_obs = obs_tensor[batch_idx]
            batch_act = act_tensor[batch_idx]
            mean, _ = rl_agent.policy(batch_obs)
            loss = criterion(mean, batch_act)
            rl_agent.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_agent.policy.parameters(), max_norm=1.0)
            rl_agent.actor_optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[SFT] Epoch {epoch:03d} | loss={final_loss:.6f}")
    return final_loss
