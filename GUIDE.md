# S-MAS Customization & Algorithm Guide

> **Last updated:** May 2026 — Phase A (42-dim observation, ADCS, fuel, thermal, degradation)

---

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

# 3. Train agents
python train.py

# 4. Export to ONNX (with numerical parity check)
python export_onnx.py --checkpoint checkpoints/mappo_phase3_best.pt

# 5. Run controller & visualize
cd ../controller_csharp && dotnet run -c Release -- --data-dir "../preprocessed-data" --model-dir "models" --steps 1000000 --skip 5000
```

---

## Files You Can Freely Customize

### 🛰️ Satellite & Environment Parameters

| What to change | File | Key variable | Notes |
|---|---|---|---|
| **Integration timestep** | `constants.h:11` | `DT = 5.0` | seconds; immutable during a run |
| **Satellite mass** | `constants.h:21` | `SAT_MASS_KG = 94.0` | PROBA-1 launch mass |
| **Cross-section area** | `constants.h:22` | `SAT_AREA_M2 = 0.36` | aerodynamic area (m²) |
| **Drag coefficient** | `constants.h:23` | `SAT_CD_NOMINAL = 2.2` | nominal Cd |
| **Solar panel power** | `constants.h:24` | `SAT_SOLAR_POWER_W = 90.0` | peak GaAs output (W) |
| **Battery capacity** | `constants.h:25` | `SAT_BATTERY_CAP_J = 360000.0` | 100 Wh = 360 kJ |
| **Payload power draw** | `constants.h:26` | `SAT_PAYLOAD_POWER_W = 25.0` | CHRIS instrument |
| **Bus power draw** | `constants.h:27` | `SAT_BUS_POWER_W = 30.0` | baseline bus draw |
| **Deep-sleep power** | `constants.h:28` | `SAT_SLEEP_POWER_W = 5.0` | minimum survival draw |
| **Fuel mass** | `constants.h:31` | `SAT_FUEL_KG = 5.6` | hydrazine propellant (kg) |
| **Thruster Isp** | `constants.h:32` | `THRUSTER_ISP_S = 220.0` | monoprop specific impulse |
| **Target altitude** | `constants.h:37` | `NOMINAL_ALT_KM = 600.0` | mission nominal orbit |
| **Reentry threshold** | `constants.h:38` | `REENTRY_ALT_KM = 200.0` | terminal condition |
| **Comms loss timeout** | `constants.h:49` | `TELEMETRY_LOSS_S = 72h` | terminal if exceeded |
| **Battery degradation** | `constants.h:58` | `BATT_CYCLE_DEGRAD = 0.00002` | capacity loss/cycle |
| **Cd drift rate** | `constants.h:79` | `CD_DRIFT_SIGMA = 0.001` | per-step random walk σ |
| **Panel drift rate** | `constants.h:80` | `PANEL_DRIFT_SIGMA = 0.00003` | per-step efficiency σ |
| **Atmospheric drag** | `config.py:25` | `density_multiplier = 0.01` | 0.01 ≈ 20yr PROBA-1 |
| **Episode length** | `config.py:23` | `max_steps_per_episode = 120_960` | ~1 week |

> **After editing `constants.h`, you must rebuild the C++ DLL:**
>
> ```bash
> cd backend_cpp && cmake --build build
> ```

### 🌡️ Thermal Model Constants

| Parameter | File | Default | Description |
|---|---|---|---|
| `SAT_ABSORPTIVITY` | `constants.h:63` | 0.3 | Solar absorptivity |
| `SAT_EMISSIVITY` | `constants.h:64` | 0.8 | IR emissivity |
| `SAT_RADIATOR_AREA` | `constants.h:65` | 0.25 m² | Radiator surface area |
| `SAT_THERMAL_MASS` | `constants.h:66` | 50.0 kg | Effective thermal mass |
| `HEATER_POWER_W` | `constants.h:68` | 7.0 W | Battery heater draw |
| `HEATER_ON_TEMP_C` | `constants.h:69` | -5.0°C | Auto-heater threshold |

### 🔄 Progressive Degradation (per-episode)

These compress ~10 years of aging into a 1-week episode so agents learn to adapt as hardware degrades mid-flight:

| Parameter | File | Default | Description |
|---|---|---|---|
| `panel_decay_per_orbit` | `config.py:30` | 0.0017 | panel eff loss/orbit (~18% drop/week) |
| `capacity_decay_per_orbit` | `config.py:31` | 645.0 J | battery capacity loss/orbit |
| `min_panel_eff` | `config.py:32` | 0.30 | floor (matches C++ clamp) |
| `min_capacity_j` | `config.py:33` | 80,000.0 J | battery never below this |

---

### 💰 Reward Weights

#### Survival Rewards (`RewardConfig`)

| Parameter | File | Default | Effect |
|---|---|---|---|
| `w_alive` | `config.py:83` | 5.0 | reward per step alive |
| `w_fuel` | `config.py:84` | 10.0 | thrust penalty (scales with 1/remaining_fuel) |
| `w_dod` | `config.py:85` | 30.0 | penalty for battery Depth of Discharge |
| `w_fdir` | `config.py:86` | 200.0 | penalty when FDIR overrides agents |
| `w_fatal` | `config.py:87` | 50,000.0 | massive penalty on terminal failure |
| `w_alt` | `config.py:88` | 0.3 | altitude deviation penalty |
| `alt_deadband_km` | `config.py:89` | 50.0 | tolerance band (±50 km) |
| `w_fuel_critical` | `config.py:91` | 500.0 | penalty when fuel < 10% |
| `w_coast_bonus` | `config.py:92` | 2.0 | bonus for coasting (throttle=0, fuel>0) |
| `w_thermal` | `config.py:94` | 50.0 | penalty when T_battery outside [-10, 45]°C |

#### Mission Rewards (`MissionRewardConfig`)

| Parameter | File | Default | Effect |
|---|---|---|---|
| `w_valid_target` | `config.py:100` | 100.0 | reward for valid imaging over target |
| `w_saa_penalty` | `config.py:101` | 200.0 | penalty for payload ON inside SAA |
| `w_idle_power` | `config.py:102` | 15.0 | penalty for payload ON when not over target |
| `w_sloth_penalty` | `config.py:103` | 50.0 | penalty for sleeping when battery>90% over target |
| `target_lat_min/max` | `config.py:105-106` | -60° / 60° | valid imaging latitude band |
| `target_min_solar_w` | `config.py:107` | 10.0 W | min sunlight for optical imaging |

---

### 🧠 Training Hyperparameters

| Parameter | File | Default | Notes |
|---|---|---|---|
| `hidden_dim` | `config.py:114` | 128 | MLP hidden layer width |
| `num_layers` | `config.py:115` | 2 | MLP depth |
| `activation` | `config.py:116` | tanh | tanh or relu |
| `gamma` | `config.py:119` | 0.99 | discount factor |
| `gae_lambda` | `config.py:120` | 0.95 | GAE smoothing |
| `clip_eps` | `config.py:121` | 0.2 | PPO clipping |
| `entropy_coeff` | `config.py:122` | 0.01 | exploration bonus |
| `lr` | `config.py:127` | 3e-4 | learning rate |
| `batch_size` | `config.py:128` | 4096 | scaled for 16 envs |
| `num_epochs` | `config.py:129` | 2 | PPO backprop passes |
| `rollout_steps` | `config.py:130` | 1176 | ≈ 1 orbit at dt=5s |
| `total_timesteps` | `config.py:140` | 1,000,000 | total training budget |
| `num_envs` | `config.py:24` | 16 | parallel environments |

---

## Observation Space (42 dimensions)

The agents receive a 42-dimensional observation vector built from the `StatePacket`:

| Group | Dims | Features |
|-------|------|----------|
| **Orbit** | 7 | alt, lat, lon, \|v\|, vx_norm, vy_norm, vz_norm |
| **Power** | 4 | soc, capacity_frac, solar_w, draw_w |
| **Environment** | 5 | ρ_log, flux10_log, flux30_log, eclipse, saa |
| **Communication** | 2 | gs_visible_any, time_since_contact_norm |
| **FDIR** | 4 | one-hot [NOMINAL, DEGRADED, SAFE, RECOVERY] |
| **Degradation** | 3 | panel_eff, cd_norm, cycles_norm |
| **SEU** | 1 | seu_active |
| **Fuel** | 2 | fuel_fraction, fuel_depleted |
| **Thermal** | 4 | temp_bus, temp_battery, temp_payload, heater_on |
| **Target Alt** | 1 | target_alt_norm (goal-conditioned, 550–750 km) |
| **ADCS** | 5 | sun_angle, nadir_error, wheel_momentum_x/y/z |
| **Lag Features** | 4 | kp_3h, f107_3h, kp_6h, f107_6h |
| **Total** | **42** | |

Configured in `config.py` → `ObsConfig`. The `obs_dim` is computed automatically as a `@property`.

---

## Action Space

| Agent | Action | Type | Dims | Range |
|-------|--------|------|------|-------|
| **Navigation** | 3D thrust direction + throttle | Continuous | 4 | [-1, 1]³ × [0, 1] |
| **Resource** | Deep-sleep toggle | Discrete | 1 | {0, 1} |
| **Mission** | Payload ON/OFF | Discrete | 1 | {0, 1} |

Defined in `ActionConfig` (config.py) and `ActionPacket` (contracts.h).

---

## Binary Contracts (StatePacket & ActionPacket)

Both structs use `#pragma pack(push, 1)` for zero-padding binary interop:

| Struct | Version | Size | Direction |
|--------|---------|------|-----------|
| `StatePacket` | v3 | 222 bytes | C++ → C# / Python |
| `ActionPacket` | v1 | 20 bytes | C# / Python → C++ |

**Rule:** NEVER change a field without incrementing `version`. All three runtimes (C++, C#, Python) must agree on layout.

Defined in:
- C++: `backend_cpp/include/contracts.h`
- C#: `controller_csharp/Interop/Contracts.cs`
- Python: `marl_python/env_wrapper.py` (ctypes `Structure`)

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
        # obs_dim = 42 (from ObsConfig.obs_dim)
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
    def push(self, obs, nav_act, bus_act, reward, value,
             nav_lp, bus_lp, done, mission_act, mission_lp): ...
    def compute_gae(self, last_value, gamma, lam): ...
    def get_batches(self, batch_size, device): ...
    def reset(self): ...


def my_update(model, optimizer, buffer, cfg, device) -> dict:
    """Replace ppo_update. Return {"policy_loss": ..., "value_loss": ..., "entropy": ...}"""
    ...
```

### Step 2: Edit `train.py` imports

Change this one line:

```python
# BEFORE (MAPPO):
from mappo import SharedActorCritic, RolloutBuffer, ppo_update

# AFTER (your algorithm):
from my_algorithm import MyActorCritic as SharedActorCritic, \
    MyRolloutBuffer as RolloutBuffer, my_update as ppo_update
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
├── config.py             ← ALL tunable parameters (edit this first)
├── train.py              ← Training loop entry point
├── mappo.py              ← MAPPO algorithm (replace for new algorithms)
├── env_wrapper.py        ← C++ DLL bridge via ctypes (StatePacket/ActionPacket)
├── observation.py        ← 42-dim observation normalisation
├── reward.py             ← Survival + Mission reward functions
├── export_onnx.py        ← PyTorch → ONNX converter (FP16, dynamic axes)
├── validate_tle.py       ← TLE historical validation script
├── visualize.py          ← Offline analysis (PNG plots)
├── checkpoints/          ← Saved model checkpoints
└── onnx_export/          ← Exported ONNX policy files

backend_cpp/
├── CMakeLists.txt        ← CMake build config (C++14)
├── include/
│   ├── constants.h       ← ALL physics constants (edit for different satellites)
│   ├── contracts.h       ← StatePacket v3 (222 bytes) + ActionPacket v1
│   ├── c_api.h           ← DLL export signatures
│   ├── types.h           ← C++17 shims (std::optional, std::clamp)
│   ├── simulation_engine.h
│   ├── orbital_mechanics.h
│   ├── atmosphere.h      ← NRLMSISE-00 density model
│   ├── satellite_bus.h   ← Power, battery, solar panel
│   ├── thermal.h         ← Multi-node thermal network
│   ├── attitude.h        ← 3-axis quaternion + reaction wheels
│   ├── stochastic.h      ← Noise, SEU, actuator error
│   ├── geometry.h        ← Eclipse, LoS, ground station
│   └── data_loader.h     ← Binary dataset reader
├── src/
│   ├── simulation_engine.cpp  ← Main sim loop, subsystem orchestration
│   ├── orbital_mechanics.cpp  ← RK4, J2, drag, thrust
│   ├── atmosphere.cpp         ← NRLMSISE-00 density model
│   ├── satellite_bus.cpp      ← Power management
│   ├── thermal.cpp            ← Thermal node simulation
│   ├── attitude.cpp           ← PD controller, 5x sub-stepping
│   ├── stochastic.cpp         ← Noise injection, SEU, actuator errors
│   ├── geometry.cpp           ← Shadow model, ground station visibility
│   ├── data_loader.cpp        ← Dataset I/O
│   └── c_api.cpp              ← DLL entry points

controller_csharp/
├── SmasController.csproj      ← .NET 10.0 project file
├── Program.cs                 ← CLI entry + test/assertion suite
├── AI/
│   ├── InferenceEngine.cs     ← ONNX inference (dynamic 37→42 dim slicing)
│   └── ObservationBuilder.cs  ← C# observation vector builder
├── Governor/
│   └── FdirGovernor.cs        ← FDIR state machine (4 modes)
├── Interop/
│   ├── Contracts.cs           ← C# mirror of StatePacket/ActionPacket
│   └── EngineApi.cs           ← P/Invoke bindings to smas_engine.dll
├── Telemetry/
│   ├── WebSocketServer.cs     ← WS server (ws://localhost:6969)
│   ├── TelemetryPacket.cs     ← JSON telemetry serialisation
│   ├── TelemetryLogger.cs     ← CSV/binary logging
│   ├── ReplayEngine.cs        ← Telemetry replay from logs
│   └── NetworkImpairment.cs   ← Simulated comm delays & packet loss
└── models/                    ← ONNX policy files (copied from Python)

frontend_webgpu/
├── package.json               ← Vite + React 19 + TypeScript 6
├── vite.config.ts             ← Dev server & build config
└── src/
    └── lib/
        └── telemetry.ts       ← Binary StatePacket decoder (222 bytes)
```

---

## FDIR Modes

The FDIR Governor operates across 4 dynamic modes and is fully exposed in the observation space:

| Mode | Value | Trigger | Behaviour |
|------|-------|---------|-----------|
| `NOMINAL` | 0 | Default | All subsystems active |
| `DEGRADED` | 1 | SoC < 30% or thermal warning | Reduce payload duty cycle |
| `SAFE` | 2 | SoC < 15% or SAA flux spike | Force deep-sleep, cut payload |
| `RECOVERY` | 3 | Post-SEU or post-SAFE | Gradual systems restart |

FDIR interventions apply a **massive negative reward penalty** (`w_fdir = 200.0`) to prevent reward hacking.

---

## Terminal Conditions

| Reason | Code | Trigger |
|--------|------|---------|
| Battery dead | 1 | SoC ≤ 0% |
| Telemetry loss | 2 | No ground contact > 72 hours |
| Re-entry | 3 | Altitude < 200 km |
| Fatal SEU | 4 | Unrecoverable single-event upset |
| Fuel exhaustion | 5 | Fuel depleted + altitude < 400 km |

Defined in `DoneReason` enum in `contracts.h:116`.

---

## Common Workflows

### "I want to simulate a different satellite"

1. Edit `backend_cpp/include/constants.h` — mass, area, power, battery, fuel
2. Rebuild DLL: `cd backend_cpp && cmake --build build`
3. Retrain: `cd marl_python && python train.py`

### "I want to change the reward function"

1. Edit `marl_python/reward.py` — modify `SurvivalReward.compute()` or `MissionReward.compute()`
2. Optionally edit weights in `marl_python/config.py` (`RewardConfig` / `MissionRewardConfig`)
3. Retrain

### "I want to change the observation space"

1. Add/remove features in `config.py` → `ObsConfig` (update the feature count)
2. Update `marl_python/observation.py` to extract & normalise the new feature
3. If the feature comes from a new `StatePacket` field:
   - Add the field to `contracts.h` (C++), `Contracts.cs` (C#), and `env_wrapper.py` (Python)
   - Increment `StatePacket.version`
4. Update `controller_csharp/AI/ObservationBuilder.cs` to match
5. Rebuild DLL and retrain

### "I want to add a 4th agent"

1. Add a new head class in `mappo.py` (like `MissionHead`)
2. Wire it into `SharedActorCritic.act()` and `evaluate_actions()`
3. Add the action field to `ActionPacket` in C++ (`contracts.h`), C# (`Contracts.cs`), and Python (`env_wrapper.py`)
4. Add reward logic in `reward.py`
5. Update `export_onnx.py` to export the new head
6. Rebuild C++ DLL and retrain

### "I want to change the target altitude range"

1. Edit `config.py:55-56` → `target_alt_min` / `target_alt_max` (default 550–750 km)
2. No DLL rebuild needed — this is Python-side goal conditioning
3. Retrain

### "I want to adjust degradation speed"

1. Edit `config.py:30-33` → `panel_decay_per_orbit`, `capacity_decay_per_orbit`, floors
2. For long-term drift rates, edit `constants.h:79-80` → `CD_DRIFT_SIGMA`, `PANEL_DRIFT_SIGMA`
3. Rebuild DLL if `constants.h` was changed, then retrain

---

## ONNX Export & Inference Pipeline

```
PyTorch model  ──[export_onnx.py]──►  ONNX (FP16, dynamic axes)
                                            │
                   ┌────────────────────────┘
                   ▼
        controller_csharp/models/*.onnx
                   │
                   ▼
        InferenceEngine.cs (ONNX Runtime C#)
                   │
          Dynamic input slicing:
          - If model expects 37 dims → slice first 37 from 42-dim obs
          - If model expects 42 dims → pass full vector
                   │
                   ▼
        ActionPacket → C++ engine via P/Invoke
```

The inference engine uses **dynamic observation slicing** (`InferenceEngine.cs`) to maintain backward compatibility with older 37-dim models while the current observation is 42-dim.
