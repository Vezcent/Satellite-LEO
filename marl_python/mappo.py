"""
S-MAS Phase 2/3 — Tasks 2.4 / 2.5 / 3.1
MAPPO (Multi-Agent PPO) with Shared Policy.

Architecture
------------
  • Navigation Actor  : obs → MLP → μ, log_σ  (continuous Gaussian)
  • Resource Actor    : obs → MLP → logit      (discrete Bernoulli)
  • Mission Actor     : obs → MLP → logit      (discrete Bernoulli, Phase 3)
  • Shared Critic     : obs → MLP → V(s)       (single value head)

All four share the same hidden layers (Shared Policy) to minimise
VRAM and enable Batch Inference [batch, obs_dim].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
from typing import Optional, Tuple, Dict, List
from config import MAPPOConfig, ObsConfig


# ═══════════════════════════════════════════════════════════════════
#  Network building blocks
# ═══════════════════════════════════════════════════════════════════

def _make_mlp(in_dim: int, hidden_dim: int, num_layers: int,
              activation: str = "tanh") -> nn.Sequential:
    """Build a simple MLP trunk."""
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    layers = []
    dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(act())
        dim = hidden_dim
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════
#  Actor heads
# ═══════════════════════════════════════════════════════════════════

class NavigationHead(nn.Module):
    """
    Continuous actor for Navigation Agent.
    Outputs: μ (4-dim), log_σ (4-dim, learnable parameter).
    Action: thrust_x, thrust_y, thrust_z ∈ [-1,1], throttle ∈ [0,1].
    """
    def __init__(self, hidden_dim: int, action_dim: int = 4):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        mu = self.mu(features)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def sample(self, features: torch.Tensor):
        mu, std = self.forward(features)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        # Squash to valid ranges
        action = torch.tanh(raw_action)
        # Correction for tanh squashing in log_prob
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, raw_action, log_prob, entropy

    def evaluate(self, features: torch.Tensor, raw_action: torch.Tensor):
        mu, std = self.forward(features)
        dist = Normal(mu, std)
        
        # 1. Base log_prob from the Normal distribution
        log_prob = dist.log_prob(raw_action).sum(-1)
        
        # 2. Tanh correction: log(1 - tanh(x)^2)
        # Using 2 * (log(2) - x - softplus(-2x)) is more stable than log(1 - tanh(x)^2)
        action = torch.tanh(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ResourceHead(nn.Module):
    """
    Discrete actor for Resource Agent (Bus Manager).
    Outputs: logit for Bernoulli(deep_sleep).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        return self.logit(features).squeeze(-1)

    def sample(self, features: torch.Tensor):
        logit = self.forward(features)
        dist = Bernoulli(logits=logit)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, features: torch.Tensor, action: torch.Tensor):
        logit = self.forward(features)
        dist = Bernoulli(logits=logit)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class MissionHead(nn.Module):
    """
    Discrete actor for Mission Agent (Payload Manager).
    Outputs: logit for Bernoulli(payload_on).
    Phase 3: Toggles the CHRIS optical instrument.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        return self.logit(features).squeeze(-1)

    def sample(self, features: torch.Tensor):
        logit = self.forward(features)
        dist = Bernoulli(logits=logit)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, features: torch.Tensor, action: torch.Tensor):
        logit = self.forward(features)
        dist = Bernoulli(logits=logit)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


# ═══════════════════════════════════════════════════════════════════
#  Shared Actor-Critic
# ═══════════════════════════════════════════════════════════════════

class SharedActorCritic(nn.Module):
    """
    Shared-trunk network for MAPPO.

    Architecture:
        obs → trunk_MLP → features
        features → NavigationHead  (continuous 4D)
        features → ResourceHead   (discrete binary)
        features → MissionHead    (discrete binary, Phase 3)
        features → value_head     (scalar V(s))
    """

    def __init__(self, obs_dim: int, cfg: Optional[MAPPOConfig] = None):
        super().__init__()
        self.cfg = cfg or MAPPOConfig()

        # Shared or Separate Trunks
        if self.cfg.shared_policy:
            self.trunk = _make_mlp(obs_dim, self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.activation)
            self.nav_trunk = self.trunk
            self.bus_trunk = self.trunk
            self.mission_trunk = self.trunk
        else:
            # Independent Trunks for each agent task
            self.nav_trunk = _make_mlp(obs_dim, self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.activation)
            self.bus_trunk = _make_mlp(obs_dim, self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.activation)
            self.mission_trunk = _make_mlp(obs_dim, self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.activation)

        self.nav_head = NavigationHead(self.cfg.hidden_dim, action_dim=4)
        self.bus_head = ResourceHead(self.cfg.hidden_dim)
        self.mission_head = MissionHead(self.cfg.hidden_dim)
        self.value_head = nn.Linear(self.cfg.hidden_dim, 1)

    def get_features(self, obs: torch.Tensor, agent_type: str = "shared") -> torch.Tensor:
        if agent_type == "nav":
            return self.nav_trunk(obs)
        elif agent_type == "bus":
            return self.bus_trunk(obs)
        elif agent_type == "mission":
            return self.mission_trunk(obs)
        else:
            return self.nav_trunk(obs) # Default

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        # Critic typically uses the primary (nav) trunk or a mix
        features = self.get_features(obs, "nav")
        return self.value_head(features).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Dict:
        """Sample actions from all three heads."""
        nav_feat = self.get_features(obs, "nav")
        bus_feat = self.get_features(obs, "bus")
        mis_feat = self.get_features(obs, "mission")

        nav_action, raw_nav, nav_lp, nav_ent = self.nav_head.sample(nav_feat)
        bus_action, bus_lp, bus_ent = self.bus_head.sample(bus_feat)
        mission_action, mission_lp, mission_ent = self.mission_head.sample(mis_feat)
        value = self.value_head(nav_feat).squeeze(-1)

        return {
            "nav_action": nav_action,           # (B, 4) squashed
            "raw_nav": raw_nav,                 # (B, 4) UNSQUASHED
            "bus_action": bus_action,
            "mission_action": mission_action,
            "nav_log_prob": nav_lp,
            "bus_log_prob": bus_lp,
            "mission_log_prob": mission_lp,
            "entropy": nav_ent + bus_ent + mission_ent,
            "value": value,
        }

    def evaluate_actions(self, obs: torch.Tensor,
                         nav_action: torch.Tensor,
                         bus_action: torch.Tensor,
                         mission_action: torch.Tensor) -> Dict:
        """Re-evaluate log_probs and entropy for saved actions."""
        nav_feat = self.get_features(obs, "nav")
        bus_feat = self.get_features(obs, "bus")
        mis_feat = self.get_features(obs, "mission")

        nav_lp, nav_ent = self.nav_head.evaluate(nav_feat, nav_action)
        bus_lp, bus_ent = self.bus_head.evaluate(bus_feat, bus_action)
        mission_lp, mission_ent = self.mission_head.evaluate(mis_feat, mission_action)
        value = self.value_head(nav_feat).squeeze(-1)

        return {
            "nav_log_prob": nav_lp,
            "bus_log_prob": bus_lp,
            "mission_log_prob": mission_lp,
            "entropy": nav_ent + bus_ent + mission_ent,
            "value": value,
        }


# ═══════════════════════════════════════════════════════════════════
#  Rollout Buffer
# ═══════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores transitions for PPO update with GAE computation."""

    def __init__(self, capacity: int, obs_dim: int, nav_dim: int = 4):
        self.capacity = capacity
        self.obs          = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.nav_acts     = np.zeros((capacity, nav_dim), dtype=np.float32)
        self.bus_acts     = np.zeros(capacity, dtype=np.float32)
        self.mission_acts = np.zeros(capacity, dtype=np.float32)
        self.rewards      = np.zeros(capacity, dtype=np.float32)
        self.values       = np.zeros(capacity, dtype=np.float32)
        self.nav_lps      = np.zeros(capacity, dtype=np.float32)
        self.bus_lps      = np.zeros(capacity, dtype=np.float32)
        self.mission_lps  = np.zeros(capacity, dtype=np.float32)
        self.dones        = np.zeros(capacity, dtype=np.float32)
        self.returns      = np.zeros(capacity, dtype=np.float32)
        self.advantages   = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def push(self, obs, nav_act, bus_act, reward, value,
             nav_lp, bus_lp, done,
             mission_act: float = 0.0, mission_lp: float = 0.0):
        """
        Store one transition.

        Phase 2 callers omit mission_act / mission_lp (they default to 0.0).
        Phase 3 callers pass them explicitly as keyword args.
        """
        i = self.ptr
        self.obs[i] = obs
        self.nav_acts[i] = nav_act
        self.bus_acts[i] = bus_act
        self.mission_acts[i] = mission_act
        self.rewards[i] = reward
        self.values[i] = value
        self.nav_lps[i] = nav_lp
        self.bus_lps[i] = bus_lp
        self.mission_lps[i] = mission_lp
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float,
                    gamma: float = 0.99, lam: float = 0.95):
        """Generalised Advantage Estimation."""
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (self.rewards[t] +
                     gamma * next_val * (1.0 - self.dones[t]) -
                     self.values[t])
            gae = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size: int, device: str = "cpu"):
        """Yield shuffled mini-batches as torch tensors."""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            if end > self.ptr:
                break
            idx = indices[start:end]
            yield {
                "obs":          torch.tensor(self.obs[idx],          device=device),
                "nav_acts":     torch.tensor(self.nav_acts[idx],     device=device),
                "bus_acts":     torch.tensor(self.bus_acts[idx],     device=device),
                "mission_acts": torch.tensor(self.mission_acts[idx], device=device),
                "returns":      torch.tensor(self.returns[idx],      device=device),
                "advantages":   torch.tensor(self.advantages[idx],  device=device),
                "nav_lps":      torch.tensor(self.nav_lps[idx],      device=device),
                "bus_lps":      torch.tensor(self.bus_lps[idx],      device=device),
                "mission_lps":  torch.tensor(self.mission_lps[idx],  device=device),
            }

    def reset(self):
        self.ptr = 0


# ═══════════════════════════════════════════════════════════════════
#  PPO Update
# ═══════════════════════════════════════════════════════════════════

from typing import Dict, Union, List

def ppo_update(model: SharedActorCritic,
               optimizer: torch.optim.Optimizer,
               buffers: Union[RolloutBuffer, List[RolloutBuffer]],
               cfg: MAPPOConfig,
               device: str = "cpu") -> Dict[str, float]:
    """
    Run PPO update epochs on the rollout buffer(s).
    Returns dict of mean losses for logging.
    """
    if not isinstance(buffers, list):
        buffers = [buffers]

    # Merge data from all environments
    all_obs = np.concatenate([b.obs[:b.ptr] for b in buffers], axis=0)
    all_nav_acts = np.concatenate([b.nav_acts[:b.ptr] for b in buffers], axis=0)
    all_bus_acts = np.concatenate([b.bus_acts[:b.ptr] for b in buffers], axis=0)
    all_mission_acts = np.concatenate([b.mission_acts[:b.ptr] for b in buffers], axis=0)
    all_nav_lps = np.concatenate([b.nav_lps[:b.ptr] for b in buffers], axis=0)
    all_bus_lps = np.concatenate([b.bus_lps[:b.ptr] for b in buffers], axis=0)
    all_mission_lps = np.concatenate([b.mission_lps[:b.ptr] for b in buffers], axis=0)
    all_returns = np.concatenate([b.returns[:b.ptr] for b in buffers], axis=0)
    all_advantages = np.concatenate([b.advantages[:b.ptr] for b in buffers], axis=0)

    total_size = len(all_obs)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    num_updates       = 0

    for _epoch in range(cfg.num_epochs):
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        for start in range(0, total_size, cfg.batch_size):
            end = start + cfg.batch_size
            if end > total_size:
                break
            idx = indices[start:end]

            obs       = torch.tensor(all_obs[idx], device=device)
            nav_a     = torch.tensor(all_nav_acts[idx], device=device)
            bus_a     = torch.tensor(all_bus_acts[idx], device=device)
            mission_a = torch.tensor(all_mission_acts[idx], device=device)
            old_nav   = torch.tensor(all_nav_lps[idx], device=device)
            old_bus   = torch.tensor(all_bus_lps[idx], device=device)
            old_mis   = torch.tensor(all_mission_lps[idx], device=device)
            ret       = torch.tensor(all_returns[idx], device=device)
            adv       = torch.tensor(all_advantages[idx], device=device)

            # Normalise advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Re-evaluate
            out = model.evaluate_actions(obs, nav_a, bus_a, mission_a)
            new_nav = out["nav_log_prob"]
            new_bus = out["bus_log_prob"]
            new_mis = out["mission_log_prob"]
            value   = out["value"]
            entropy = out["entropy"]

            # Combined log-prob ratio (3-agent) - Log-space summation for stability
            new_log_prob_sum = new_nav + new_bus + new_mis
            old_log_prob_sum = old_nav + old_bus + old_mis
            ratio = torch.exp(new_log_prob_sum - old_log_prob_sum)

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                                1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(value, ret)

            # Total loss
            loss = (policy_loss
                    + cfg.value_loss_coeff * value_loss
                    - cfg.entropy_coeff * entropy.mean())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.mean().item()
            num_updates       += 1

    if num_updates == 0:
        return {"policy_loss": 0., "value_loss": 0., "entropy": 0.}

    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss":  total_value_loss / num_updates,
        "entropy":     total_entropy / num_updates,
    }
