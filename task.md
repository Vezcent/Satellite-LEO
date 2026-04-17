##### ACTIVATE .VENV .\.venv\Scripts\activate

# IMPLEMENTATION PLAN (ACTIONABLE TASKS): S-MAS PROJECT

[General Scientific Disclaimer: Training vs. Mission-Grade]

[Mission-Grade Operational Twin Disclaimer]
The C++ Physics Engine developed in this project bridges the gap between high-throughput MARL training and mission-grade operations.
Designed as a **High-Fidelity Operational Twin**, it goes beyond lightweight approximations by integrating stochastic realism—including sensor noise, actuator non-idealities, communication delays, and radiation anomalies.
Governed by a strict FDIR (Failure Detection, Isolation, and Recovery) state machine and validated against historical PROBA-1 TLE data, this environment is explicitly built to simulate real-world aerospace operations and stress-test Hybrid AI control systems under robust, noisy conditions.

## Phase 0: System Setup & Architecture

    Setting up the multi-language environment and defining strict global contracts.

    [ ] Task 0.1: Initialize the Git Repository and directory structure (`/backend_cpp`, `/controller_csharp`, `/marl_python`, `/frontend_webgpu`).

    [x] Task 0.2: Set up the Python venv and install dependencies (torch, numpy, onnx).

    [ ] Task 0.3: Initialize the C++ project with CMake (crucial for building .dll/.so shared libraries).

    [ ] Task 0.4: Initialize the C# project and React project (Vite + TypeScript).

    [ ] Task 0.5: Strict State/Action Contracts & Versioning (CRITICAL)
        [ ] Define the Global Physics Time Step: dt = 5.0 seconds. All integrators and reward calculations must hardcode or strictly adhere to this delta time.
        [ ] Versioning Contract: Define the Memory Layout Contract (C-Struct). Both StatePacket and ActionPacket must include a version header (e.g., `uint8_t version = 1;`). Prevent memory corruption between Python and C++ serialization.

### Phase 0.5: Data Pipeline Processing (< 50MB) [COMPLETED]

    Processing the raw environmental data (Space Weather, Orbit TLEs, Ground Stations, SPENVIS radiation) to create clean, localized datasets.

        [x] Task 0.5.1: Space Weather Cleaning
            [x] Cleaned raw OMNIWeb data, normalized Kp indices, and interpolated missing values.
            [x] Output generated: `preprocessed-data/space_weather.csv`.

        [x] Task 0.5.2: SPENVIS Radiation Heatmap Generation (SAA Localized)
            [x] Parsed the SPENVIS AP-8 MAX model for the 600km altitude shell.
            [x] Output generated: `preprocessed-data/saa_heatmap_600km.csv` (containing >10 MeV and >30 MeV integral proton fluxes).

        [x] Task 0.5.3: Ground Station Coordination
            [x] Formatted precise Lat/Lon/Alt/Elevation-Mask constraints for Redu and Kiruna ESA stations.
            [x] Output generated: `preprocessed-data/ground_stations.json`.

        [x] Task 0.5.4: Orbit TLE Ephemeris Formatting
            [x] Extracted and formatted PROBA-1 initial state vectors.
            [x] Output generated: `preprocessed-data/initial_state.txt`.

#### Phase 1: Build Engine (C++ Physics Core Backend)

    Developing the physics core with injected stochastic realism. Dynamics modules are decoupled for easier Unit Testing.

    [ ] Task 1.1: Data Parsers & Memory Loaders (C++)
        [ ] 1.1.1: Write a CSV parser to load `preprocessed-data/space_weather.csv` (F10.7, Kp/Ap) into a time-indexed std::map for NRLMSISE-00.
        [ ] 1.1.2: Write a CSV parser to load `preprocessed-data/saa_heatmap_600km.csv` into a fast 2D array/grid for spatial lookups.
        [ ] 1.1.3: Integrate `nlohmann/json` to parse `preprocessed-data/ground_stations.json` into a std::vector of GroundStation structs.
        [ ] 1.1.4: Write a simple text reader for `preprocessed-data/initial_state.txt` to extract the starting Position and Velocity vectors for the RK4 integrator.

    [ ] Task 1.2: Math & Geometry Engine
        [ ] 1.2.1: Implement Analytical Ray-Box Intersection (Eclipse calculation via Cylindrical Shadow Model).
        [ ] 1.2.2: Implement Spherical Trigonometry (LoS calculation, 5° Elevation Mask).

    [ ] Task 1.3: Atmospheric Subsystem (NRLMSISE-00)
        [ ] 1.3.1: Embed the open-source NRLMSISE-00 atmospheric model.
        [ ] 1.3.2: Write a Wrapper Interface returning Density $\rho$.

    [ ] Task 1.4: Orbital Perturbations Subsystem
        [ ] 1.4.1: Implement Aerodynamic Drag ($F_D = -0.5 \cdot \rho \cdot A \cdot C_D \cdot v^2$).
        [ ] 1.4.2: Implement J2 perturbation (Earth oblateness effect).

    [ ] Task 1.5: Numerical Integrator (ODE Solver)
        [ ] 1.5.1: Implement a Runge-Kutta 4 (RK4) solver strictly utilizing dt = 5.0s.

    [ ] Task 1.6: Power Subsystem & Realistic Degradation
        [ ] 1.6.1: Code the SatelliteBus class (State of Charge, GaAs Power Budget, Arrhenius degradation).
        [ ] 1.6.2: Implement Battery Degradation active capacity loss based on charge/discharge cycles.
        [ ] 1.6.3: Define the Failure Contract (Done = True):
            1. Power Failure: Battery SoC <= 0%.
            2. Telemetry Loss: Loss of LoS with Ground Stations for > 72 continuous hours.
            3. Re-entry: Altitude drops below 200 km.

    [ ] Task 1.7: C-API Export
        [ ] Write `extern "C"` functions adhering strictly to the Task 0.5 versioned State Contract for C# P/Invoke.

    [ ] Task 1.8: Stochastic Noise Injection Module
        [ ] 1.8.1: Inject Gaussian noise (`std::normal_distribution`) into Position, Velocity, and Power sensor readings before passing them to the AI state.
        [ ] 1.8.2: Implement random SEU anomalies triggered probabilistically during SAA transits.

    [ ] Task 1.9: Actuator Error & Execution Latency Model
        [ ] 1.9.1: Apply a ±5% deviation to thruster commands to simulate mechanical non-ideality.
        [ ] 1.9.2: Implement an Action Queue to simulate command execution delay (e.g., actions proposed at $t$ are executed at $t+1$ or $t+2$).

    [ ] Task 1.10: Epistemic Uncertainty (Model Drift)
        [ ] Implement a slow stochastic random walk (drift) for the Drag Coefficient ($C_D$) and baseline battery capacity to prevent the MARL policies from overfitting to a perfect physics model.

##### Phase 2: Core MAS & Survival Training (Python CTDE)

    Constructing the MARL (MADRL, MADL for optional) architecture, scaling strategies, and training loop.

    [x] Task 2.1: Data Preprocessing & Features
        [x] 2.1.1: Space weather data loaded via C++ engine; Python receives it through StatePacket.
        [x] 2.1.2: Lag feature placeholders (kp_3h, f107_3h, kp_6h, f107_6h) in observation vector.

    [x] Task 2.2: Observation/Action Spaces & Normalization
        [x] Min-Max for bounded (alt, SoC, flux), Robust Scaling for unbounded (atm density log10).
        [x] FDIR one-hot [NOM, DEG, SAFE, REC] exposed in 29-dim observation vector.
        [x] Action Contracts implemented:
            1. Nav Agent: Continuous 4D (thrust xyz [-1,1], throttle [0,1]) via Gaussian + tanh squash.
            2. Bus Agent: Discrete Bernoulli (deep_sleep 0/1).

    [x] Task 2.3: Survival Logic & Explicit Reward Weights
        [x] R_t = w1(alive) - w2(dV) - w3(DoD) - w4(FDIR_intervention) - w5(fatal)
        [x] Weights: w1=1.0, w2=5.0, w3=2.0, w4=100.0 (FDIR transition penalty), w5=1000.0.

    [x] Task 2.4: Scaling Strategy (Multi-Agent Architecture)
        [x] 2.4.1: Shared Policy MAPPO with MLP trunk + NavigationHead + ResourceHead + Critic.
        [x] 2.4.2: Batch Inference via tensor operations [batch_size, obs_dim].

    [x] Task 2.5: Training Strategy & Loop Execution
        [x] 2.5.1: Global seed management (PyTorch + NumPy + CUDA).
        [x] 2.5.2: Episode length = 17,280 steps (~1 day). Early truncation on Failure Contract.
        [x] 2.5.3: Full training loop: Reset -> Rollout -> GAE -> PPO Update -> JSONL logging + checkpoints.

###### Phase 3: Mission Layer & Coordination (Py)

    The opt-in behavioral layer focusing on the CHRIS instrument. **Users will implement their logic here, but a default Data Collection mission is provided.**

    [ ] Task 3.1: Mission Action Space & Hardware Constraints
        [ ] Mission Agent: Discrete/Binary [0, 1] to toggle the CHRIS optical payload.
        [ ] **Constraint Implementation: Ensure that Action 1 (ON) explicitly subtracts power from the battery state in the C++ core to simulate payload energy consumption.**

    [ ] Task 3.2: Dynamic Reward Shaping
        [ ] Implement the training reward to penalize unsafe and wasteful payload usage:
                $$R_{mission} = R_{survival} + (S_{payload} \cdot 50.0) - (S_{payload} \cdot 500.0) - (S_{payload} \cdot 5.0)$$
        [ ] Explicit Weights: +50.0 (Valid Target Imaged), -500.0 (Payload ON inside SAA boundaries), -5.0 (Power draw penalty for leaving payload ON when not over target).

    [ ] Task 3.3: Software-Defined Resiliency
        [ ] Implement Meta-Coordination: Override Payload to 0 if Bus Agent triggers Deep Sleep.

    [ ] Task 3.4: ONNX Export Pipeline
        [ ] Export models to `.onnx` with Dynamic Axes and FP16 for scalable Batch Inference.

###### Phase 4: The Controller & Operations Simulation (C# Execution Bridge)

    The orchestrator handling C++ execution, AI batch inference, FDIR safety rules, and Ops simulation.

    [ ] Task 4.1: ONNX Batch Inference Integration
        [ ] Map C# state array into a single 2D ONNX Tensor `[1111, state_dim]` utilizing the ONNX Runtime C++ API.

    [ ] Task 4.2: P/Invoke Bridge (C# <-> C++)
        [ ] Implement unmanaged memory mapping based on the versioned C-Structs.

    [ ] Task 4.3: Mission-Critical WebSocket & Network Simulation
        [ ] 4.3.1: Set up an Async WebSocket Server with a Binary Packet Schema `[Header(Version) | Payload | Checksum]`.
        [ ] 4.3.2: Implement **Network Impairment Simulator** (Circular buffer to inject 1-10s delays and random packet drops).
        [ ] 4.3.3: Implement Backpressure (Drop Stale Frames) to prevent server OOM.

    [ ] Task 4.4: Telemetry Logging & Episode Replay System
        [ ] Log states, ONNX decisions, and FDIR interventions to Parquet/Binary.
        [ ] Implement an Offline Replay Mode streaming directly to the WebGPU frontend.

    [ ] Task 4.5: FDIR & State Machine Logic (The Governor)
        [ ] Implement the strict 4-Mode State Machine:
            - NOMINAL: SoC > 20% (Normal operation).
            - DEGRADED: SoC < 20% (Force payload OFF, restrict actions).
            - SAFE: SoC < 10% (Override AI entirely, force Deep Sleep).
            - RECOVERY: Transition logic to return to NOMINAL once stable.

###### Phase 5: WebGPU Dashboard (React Frontend)

    Visual presentation with strict VRAM Lifecycle management.

    [ ] Task 5.1: WebSocket Client & Decoder
        [ ] Write a `useWebSocket` hook to ingest the Binary stream and decode the version header.

    [ ] Task 5.2: WebGPU Context & Resource Lifecycle (Critical)
        [ ] Ensure every `GPUBuffer` and `GPUTexture` invokes `.destroy()` upon React component unmounting.

    [ ] Task 5.3: Instanced Rendering (The Swarm)
        [ ] Write WGSL Shaders utilizing `@builtin(instance_index)` to draw 1111 satellites in a single Draw Call.
        [ ] Map state-driven colors: Blue (Nominal), Green (Active Payload), Red (Fatal Error).

###### Phase 6: Validation & Real Data

    [ ] Task 6.1: Real-World TLE Validation
        [ ] Compare the simulated baseline orbital decay against historical PROBA-1 TLE data to fine-tune the NRLMSISE-00 density multiplier and stochastic noise parameters.
