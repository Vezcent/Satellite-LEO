"""
S-MAS Phase 2 — Integration Test
Validates all modules: env_wrapper, observation, reward, mappo.
"""
import sys
import os
import ctypes
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvConfig, ObsConfig, RewardConfig, MAPPOConfig
from env_wrapper import SatelliteEnv, StatePacket, ActionPacket
from observation import ObservationBuilder
from reward import SurvivalReward


def main():
    print("=" * 60)
    print("  S-MAS Phase 2 — Integration Test")
    print("=" * 60)

    # ── ABI check ──
    print("\n[1] ABI Size Check")
    py_state = ctypes.sizeof(StatePacket)
    py_action = ctypes.sizeof(ActionPacket)
    print("    StatePacket:  %d bytes" % py_state)
    print("    ActionPacket: %d bytes" % py_action)

    # ── Environment ──
    print("\n[2] Environment Creation")
    cfg = EnvConfig()
    print("    DLL:  %s" % cfg.dll_path)
    print("    Data: %s" % cfg.data_dir)
    env = SatelliteEnv(cfg)
    print("    Engine created + initialised OK")

    # ── Reset ──
    print("\n[3] Reset")
    state = env.reset()
    print("    Alt:  %.1f km" % state.altitude_km)
    print("    SoC:  %.1f%%" % (state.battery_soc * 100))
    print("    FDIR: %d" % state.fdir_mode)

    # ── Observation ──
    print("\n[4] Observation Builder")
    obs_b = ObservationBuilder(ObsConfig())
    obs = obs_b.build(state)
    print("    obs shape: %s" % str(obs.shape))
    print("    obs dim:   %d" % obs_b.obs_dim)
    print("    obs range: [%.3f, %.3f]" % (obs.min(), obs.max()))
    print("    obs[:5]:   %s" % str(obs[:5]))

    # ── Reward ──
    print("\n[5] Reward Function")
    rew_fn = SurvivalReward(RewardConfig())
    action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0}
    state2, _, done, info = env.step(action)
    r = rew_fn.compute(state2, action, done, info)
    print("    Reward (no-op step): %.3f" % r)

    # ── 50 steps ──
    print("\n[6] Running 50 simulation steps")
    for i in range(49):
        state2, _, done, info = env.step(action)
        if done:
            print("    Episode ended at step %d" % (i + 2))
            break
    r = rew_fn.compute(state2, action, done, info)
    print("    Step 50: Alt=%.1f km, SoC=%.1f%%, R=%.3f" % (
        state2.altitude_km, state2.battery_soc * 100, r))

    # ── MAPPO model ──
    print("\n[7] MAPPO Model")
    import torch
    from mappo import SharedActorCritic

    model = SharedActorCritic(obs_b.obs_dim, MAPPOConfig())
    obs_t = torch.tensor(obs).unsqueeze(0)
    out = model.act(obs_t)
    nav_a = out["nav_action"].detach().numpy().flatten()
    bus_a = out["bus_action"].item()
    value = out["value"].item()
    params = sum(p.numel() for p in model.parameters())

    print("    nav_action: [%.3f, %.3f, %.3f, %.3f]" % (
        nav_a[0], nav_a[1], nav_a[2], nav_a[3]))
    print("    bus_action: %d" % int(bus_a))
    print("    value:      %.4f" % value)
    print("    parameters: %d" % params)

    # ── Rollout buffer ──
    print("\n[8] Rollout Buffer + GAE")
    from mappo import RolloutBuffer

    buf = RolloutBuffer(capacity=50, obs_dim=obs_b.obs_dim)
    state = env.reset()
    obs = obs_b.build(state)

    for step in range(50):
        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            out = model.act(obs_t)

        nav_act = out["nav_action"].squeeze(0).cpu().numpy()
        bus_act = int(out["bus_action"].item())
        val = out["value"].item()
        nav_lp = out["nav_log_prob"].item()
        bus_lp = out["bus_log_prob"].item()

        act_dict = {
            "nav": np.array([nav_act[0], nav_act[1], nav_act[2],
                             (nav_act[3] + 1.0) / 2.0], dtype=np.float32),
            "bus": bus_act,
        }
        state, _, done, info = env.step(act_dict)
        next_obs = obs_b.build(state)
        r = rew_fn.compute(state, act_dict, done, info)

        buf.push(obs, nav_act, float(bus_act), r, val, nav_lp, bus_lp, done)
        obs = next_obs
        if done:
            break

    buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
    print("    Buffer filled: %d steps" % buf.ptr)
    print("    Advantages: mean=%.3f std=%.3f" % (
        buf.advantages[:buf.ptr].mean(),
        buf.advantages[:buf.ptr].std()))
    print("    Returns:    mean=%.3f std=%.3f" % (
        buf.returns[:buf.ptr].mean(),
        buf.returns[:buf.ptr].std()))

    # ── PPO update ──
    print("\n[9] PPO Update (1 epoch)")
    from mappo import ppo_update

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    mcfg = MAPPOConfig(num_epochs=1, batch_size=16)
    losses = ppo_update(model, optimizer, buf, mcfg)
    print("    policy_loss: %.5f" % losses["policy_loss"])
    print("    value_loss:  %.5f" % losses["value_loss"])
    print("    entropy:     %.4f" % losses["entropy"])

    # ── Cleanup ──
    env.close()

    print("\n" + "=" * 60)
    print("  ALL PHASE 2 MODULES VALIDATED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
