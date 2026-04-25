# S-MAS Customization & Algorithm Guide

## Quick Setup (5 commands)

```bash
# 1. Build C++ engine (run from VS Developer x64 prompt)
cd backend_cpp
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Create Python env & install deps
cd ../marl_python
python -m venv ../.venv && ../.venv/Scripts/activate
pip install -r ../requirements.txt

# 3. Train agents (recommended 10M steps for mission)
python train.py --phase 3 --device cpu

# 4. Export & deploy (with numerical parity check)
python export_onnx.py --checkpoint checkpoints/mappo_phase3_best.pt

# 5. Run & Analyze
cd ../controller_csharp && dotnet run -c Release -- --data-dir "../preprocessed-data" --model-dir "models" --steps 1000000 --skip 5000
python ../result/visualize/visualize.py
```

---

## Files You Can Freely Customize

### 🛰️ Satellite & Environment Parameters

| What to change | File | Key variables |
|---|---|---|
| **Atmospheric drag** | `marl_python/config.py:25` | `density_multiplier` — lower = less drag, longer life (0.01 = ~20yr) |
| **Episode length** | `marl_python/config.py:23` | `max_steps_per_episode` — 17280 = 1 day |
| **Satellite mass** | `backend_cpp/include/constants.h:21` | `SAT_MASS_KG = 94.0` |
| **Cross-section area** | `backend_cpp/include/constants.h:22` | `SAT_AREA_M2 = 0.36` |
| **Drag coefficient** | `backend_cpp/include/constants.h:23` | `SAT_CD_NOMINAL = 2.2` |
| **Solar panel power** | `backend_cpp/include/constants.h:24` | `SAT_SOLAR_POWER_W = 90.0` |
| **Battery capacity** | `backend_cpp/include/constants.h:25` | `SAT_BATTERY_CAP_J = 360000.0` (100 Wh) |
| **Payload power draw** | `backend_cpp/include/constants.h:26` | `SAT_PAYLOAD_POWER_W = 25.0` |
| **Bus power draw** | `backend_cpp/include/constants.h:27` | `SAT_BUS_POWER_W = 30.0` |
| **Target altitude** | `backend_cpp/include/constants.h:31` | `NOMINAL_ALT_KM = 600.0` |
| **Reentry threshold** | `backend_cpp/include/constants.h:32` | `REENTRY_ALT_KM = 200.0` |
| **Observation Dim** | `marl_python/config.py:44` | `obs_dim = 30` (includes `vz_norm` for 3D context) |
| **Agent Trunks** | `marl_python/config.py:111` | `shared_policy = False` (Independent brains for Nav/Mission) |
| **Battery degradation** | `backend_cpp/include/constants.h:52` | `BATT_CYCLE_DEGRAD = 0.00005` |
| **Cd drift rate** | `backend_cpp/include/constants.h:60` | `CD_DRIFT_SIGMA = 0.001` |
| **Comms loss timeout** | `backend_cpp/include/constants.h:43` | `TELEMETRY_LOSS_S = 72h` |
| **Integration timestep** | `backend_cpp/include/constants.h:11` | `DT = 5.0` seconds |

> **After editing `constants.h`, you must rebuild the C++ DLL:**
>
> ```bash
> cd backend_cpp && cmake --build build
> ```

### Reward Weights

| Parameter | File | Default | Effect |
|---|---|---|---|
| `w_alive` | `marl_python/config.py:65` | 1.0 | Reward per step alive |
| `w_fuel` | `marl_python/config.py:66` | 5.0 | Penalty per ΔV used |
| `w_dod` | `marl_python/config.py:67` | 2.0 | Penalty for battery depth-of-discharge |
| `w_fdir` | `marl_python/config.py:68` | 100.0 | Penalty when FDIR overrides agents |
| `w_fatal` | `marl_python/config.py:69` | 1000.0 | Penalty on terminal failure |
| `w_alt` | `marl_python/config.py:70` | 50.0 | Penalty for altitude deviation |
| `alt_deadband` | `marl_python/config.py:72` | 5.0 | Tolerance band (±5km) |
| `w_valid_target` | `marl_python/config.py:78` | 500.0 | Reward for valid imaging (high value = active mission) |
| `w_saa_penalty` | `marl_python/config.py:79` | 1000.0 | Penalty for payload ON in SAA |
| `w_sloth_penalty` | `marl_python/config.py:81` | 200.0 | Penalty for sleeping when battery is full over target |

### Training Hyperparameters

| Parameter | File | Default |
|---|---|---|
| `hidden_dim` | `marl_python/config.py:88` | 128 |
| `num_layers` | `marl_python/config.py:89` | 2 |
| `gamma` | `marl_python/config.py:93` | 0.99 |
| `gae_lambda` | `marl_python/config.py:94` | 0.95 |
| `clip_eps` | `marl_python/config.py:95` | 0.2 |
| `entropy_coeff` | `marl_python/config.py:96` | 0.01 |
| `lr` | `marl_python/config.py:101` | 3e-4 |
| `batch_size` | `marl_python/config.py:102` | 256 |
| `rollout_steps` | `marl_python/config.py:104` | 1176 (~1 orbit) |

---

## How to Implement a Different MARL Algorithm

The MAPPO implementation lives in **one file**: `marl_python/mappo.py`. To swap algorithms, you only need to modify **2 files**:

### Step 1: Create your algorithm file

Replace or create a new file alongside `mappo.py`. Your algorithm must provide:

```python
# my_algorithm.py — must implement these 3 interfaces:

class MyActorCritic(nn.Module):
    """Replace SharedActorCritic"""

    def __init__(self, obs_dim: int, cfg):
        # obs_dim = 29 (from ObsConfig)
        ...

    def act(self, obs: torch.Tensor) -> dict:
        # Must return:
        return {
            "nav_action":      ...,  # (batch, 4) continuous [-1, 1]
            "bus_action":      ...,  # (batch,)   discrete {0, 1}
            "mission_action":  ...,  # (batch,)   discrete {0, 1}
            "nav_log_prob":    ...,  # (batch,)
            "bus_log_prob":    ...,  # (batch,)
            "mission_log_prob":...,  # (batch,)
            "entropy":         ...,  # (batch,)
            "value":           ...,  # (batch,)
        }

    def evaluate_actions(self, obs, nav_action, bus_action, mission_action) -> dict:
        # Same keys as act() but for stored actions
        ...

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        # Return V(s) scalar per batch element
        ...


class MyRolloutBuffer:
    """Replace RolloutBuffer"""
    def push(self, obs, nav_act, bus_act, reward, value, nav_lp, bus_lp, done, mission_act, mission_lp): ...
    def compute_gae(self, last_value, gamma, lam): ...
    def get_batches(self, batch_size, device): ...
    def reset(self): ...


def my_update(model, optimizer, buffer, cfg, device) -> dict:
    """Replace ppo_update. Return {"policy_loss": ..., "value_loss": ..., "entropy": ...}"""
    ...
```

### Step 2: Edit `train.py` imports (line 44)

Change this one line:

```python
# BEFORE (MAPPO):
from mappo import SharedActorCritic, RolloutBuffer, ppo_update

# AFTER (your algorithm):
from my_algorithm import MyActorCritic as SharedActorCritic, MyRolloutBuffer as RolloutBuffer, my_update as ppo_update
```

That's it. The rest of `train.py` uses these 3 names generically.

### Algorithm Ideas to Try

| Algorithm | Key change from MAPPO |
|---|---|
| **MADDPG** | Replace PPO clipping with deterministic policy gradient + experience replay |
| **QMIX** | Replace actor-critic with Q-networks + mixing network for joint Q |
| **IPPO** | Remove shared trunk — each agent gets independent parameters |
| **HAPPO** | Sequential policy update instead of simultaneous (heterogeneous agents) |
| **MAA2C** | Remove PPO clipping, use vanilla advantage actor-critic |
| **DQN (discrete-only)** | Discretize thrust into N directions, use Q-learning + replay buffer |

---

## File Map: What Each File Does

```
marl_python/
├── config.py          ← ALL tunable parameters (edit this first)
├── train.py           ← Training loop (edit imports for new algorithms)
├── mappo.py           ← MAPPO algorithm (replace for new algorithms)
├── env_wrapper.py     ← C++ DLL bridge via ctypes (rarely edit)
├── observation.py     ← 29-dim observation normalization (rarely edit)
├── reward.py          ← Reward functions (edit to reshape rewards)
├── export_onnx.py     ← PyTorch → ONNX converter (with numerical parity check)
├── validate_tle.py    ← TLE validation script
│
result/visualize/
├── visualize.py       ← Offline analysis tool (generates PNG plots and mission CSVs)

backend_cpp/
├── include/
│   ├── constants.h    ← ALL physics constants (edit for different satellites)
│   ├── c_api.h        ← DLL export signatures
│   └── simulation_engine.h
├── src/
│   ├── simulation_engine.cpp  ← Main sim loop, power, FDIR, subsystems
│   ├── orbital_mechanics.cpp  ← RK4, J2, drag, thrust
│   ├── atmosphere.cpp         ← NRLMSISE-00 density model
│   └── c_api.cpp              ← DLL entry points

controller_csharp/
├── Interop/EngineApi.cs       ← P/Invoke bindings (edit if C API changes)
├── AI/InferenceEngine.cs      ← ONNX inference (edit if agents change)
├── Governor/FdirGovernor.cs   ← Safety rules (edit to change FDIR logic)
├── Program.cs                 ← CLI + simulation orchestrator
```

---

## Common Workflows

### "I want to simulate a different satellite"

1. Edit `backend_cpp/include/constants.h` — mass, area, power, battery
2. Rebuild DLL: `cd backend_cpp && cmake --build build`
3. Retrain: `python train.py --phase 3 --total_steps 1000000`

### "I want to change the reward function"

1. Edit `marl_python/reward.py` — modify `SurvivalReward.compute()` or `MissionReward.compute()`
2. Optionally edit weights in `marl_python/config.py` (RewardConfig)
3. Retrain

### "I want to add a 4th agent"

1. Add a new head class in `mappo.py` (like `MissionHead`)
2. Wire it into `SharedActorCritic.act()` and `evaluate_actions()`
3. Add the action field to `ActionPacket` in both C++ (`contracts.h`) and C# (`Contracts.cs`)
4. Add reward logic in `reward.py`
5. Update `export_onnx.py` to export the new head
6. Rebuild C++ DLL and retrain
