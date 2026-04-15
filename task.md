# IMPLEMENTATION PLAN (ACTIONABLE TASKS): S-MAS PROJECT

    [General Scientific Disclaimer: Training vs. Mission-Grade]
    The C++ Physics Engine developed in this project is explicitly designed as a "Training Approximation" environment. The priority is to achieve high-throughput simulation (millions of steps per second) necessary for MARL convergence. While equations are grounded in real physics, the models utilize lightweight approximations and static lookups. This system is optimized for training AI policies and is not intended to replace mission-grade orbital mechanics software.

    Read pipeline.md first

    Read data from dataset folder, then do the task below

## Phase 0: System Setup & Architecture

    Setting up the multi-language environment and defining strict global contracts.

        [ ] Task 0.1: Initialize the Git Repository and directory structure (/backend_cpp, /controller_csharp, /marl_python, /frontend_webgpu).

        [ ] Task 0.2: Set up the Python venv and install dependencies (torch, numpy, onnx).

        [ ] Task 0.3: Initialize the C++ project with CMake (crucial for building .dll/.so shared libraries).

        [ ] Task 0.4: Initialize the C# project and React project (Vite + TypeScript).

        [ ] Task 0.5: Strict State/Action Contracts & Versioning (CRITICAL)

            [ ] Define the Global Physics Time Step: dt = 5.0 seconds. All integrators and reward calculations must hardcode or strictly adhere to this delta time.

            [ ] Versioning Contract: Define the Memory Layout Contract (C-Struct). Both StatePacket and ActionPacket must include a version header (e.g., uint8_t version = 1;). If Python sends a v2 Action to a v1 C++ Engine, the system must catch the version mismatch and throw an error, preventing memory corruption.

### Phase 1: Build Engine (C++ Physics Core Backend)

    Developing the ultra-lightweight physics core. Dynamics modules are decoupled for easier Unit Testing.

        [ ] Task 1.1: Data Parsers & Static Heatmaps

            [ ] 1.1.1: Write a CSV parser for F10.7 and Kp/Ap indices.

            [ ] 1.1.2: Write a parser for the SPENVIS AP-8 2D Static Heatmap.

        [ ] Task 1.2: Math & Geometry Engine

            [ ] 1.2.1: Implement Analytical Ray-Box Intersection (Eclipse calculation).

            [ ] 1.2.2: Implement Spherical Trigonometry (LoS calculation, $5^\circ$ Elevation Mask).

        [ ] Task 1.3: Atmospheric Subsystem (NRLMSISE-00)

            [ ] 1.3.1: Embed the open-source NRLMSISE-00 atmospheric model.

            [ ] 1.3.2: Write a Wrapper Interface returning Density $\rho$.

        [ ] Task 1.4: Orbital Perturbations Subsystem

            [ ] 1.4.1: Implement Aerodynamic Drag ($F_D = -0.5 \cdot \rho \cdot A \cdot C_D \cdot v^2$).

            [ ] 1.4.2: Implement J2 perturbation (Earth oblateness effect).

        [ ] Task 1.5: Numerical Integrator (ODE Solver)

            [ ] 1.5.1: Implement a Runge-Kutta 4 (RK4) solver strictly utilizing dt = 5s.

        [ ] Task 1.6: Power Subsystem & Failure Contract

            [ ] 1.6.1: Code the SatelliteBus class (State of Charge, Power Budget, Arrhenius degradation).

            [ ] 1.6.2: Define the Failure Contract: The satellite is strictly considered "Dead" (Done = True) if any of the following occur:

                1. Power Failure: Battery SoC $\le 0\%$.

                2. Telemetry Loss: Loss of Line-of-Sight (LoS) communication with any Ground Station for $> 72$ continuous hours.

                3. Re-entry: Altitude drops below $200\text{ km}$.

        [ ] Task 1.7: C-API Export

            [ ] Write extern "C" functions adhering strictly to the Task 0.5 versioned State Contract.

#### Phase 2: Core MAS & Survival Training (Python) 

    Constructing the MARL architecture, scaling strategies, and training loop.

        [ ] Task 2.1: Data Preprocessing & Features

            [ ] 2.1.1: Linearly interpolate space weather data to match the dt = 5s tick rate.

            [ ] 2.1.2: Generate Look-ahead/Look-back vectors.

        [ ] Task 2.2: Observation/Action Spaces & Normalization

            [ ] Define the State Space, apply Data Normalization (Min-Max, Robust Scaling), and cast to FP16.

            [ ] Define the Action Contract:

                1. Thrust (Nav Agent): Continuous 3D Vector $[-1.0, 1.0]$ representing attitude vector and throttle percentage.

                2. Sleep (Bus Agent): Discrete/Binary $[0, 1]$. Output uses an Argmax or Sigmoid threshold ($> 0.5 = 1$).

                3. Payload (Mission Agent): Discrete/Binary $[0, 1]$ (ON/OFF).

        [ ] Task 2.3: Survival Logic & Explicit Reward Weights

            [ ] Implement $R_{survival}$ utilizing explicit weights to balance competing objectives. Example baseline:

                $R_t = w_1(1.0) - w_2(\Delta V_{used}) - w_3(\text{DoD}) - w_4(P_{fatal})$

                Suggested Weights: $w_1 = 1.0$ (Alive bonus), $w_2 = 5.0$ (Fuel penalty), $w_3 = 2.0$ (Battery Depth of Discharge penalty), $w_4 = 1000.0$ (Triggering the Failure Contract).

        [ ] Task 2.4: Scaling Strategy (Multi-Agent Architecture)

            [ ] 2.4.1: Implement a Shared Policy Architecture (e.g., MAPPO) to drastically reduce VRAM usage.

            [ ] 2.4.2: Implement Batch Inference ([num_agents, obs_dim]).

        [ ] Task 2.5: Training Strategy & Loop Execution

            [ ] 2.5.1: Define Episode Length: Based on dt = 5s (e.g., 1 Orbit $\approx 1,176$ steps, 1 Week $\approx 120,960$ steps). Implement early truncation if the Failure Contract is triggered.

            [ ] 2.5.2: Write the Training Loop (Reset $\rightarrow$ Rollout $\rightarrow$ GAE $\rightarrow$ Update Policy $\rightarrow$ Log Metrics).

##### Phase 3: Mission Layer & Coordination (Python)

    The opt-in behavioral layer. Can be toggled and trained independently of Phase 2.

        [ ] Task 3.1: Mission Reward Weights & Constraints

            [ ] Implement $r_{coverage}$ and $P_{fatal\_risk}$ with explicit weights.

                Example: $R_{mission} = +50.0$ (Valid Target Imaged) $- 500.0$ (Payload ON inside SAA boundaries).

        [ ] Task 3.2: Mission Agent & Resiliency Logic

            [ ] Construct the Neural Network for the Mission Agent.

            [ ] Implement Software-Defined Resiliency (Override: If Bus Agent deep sleeps $\rightarrow$ Force Mission Payload to 0).

        [ ] Task 3.3: ONNX Export Pipeline

            [ ] Quantize weights (FP32 $\rightarrow$ FP16) and export models to .onnx.

###### Phase 4: The Controller (C# Execution Bridge)

    The orchestrator handling C++ execution, AI batch inference, and telemetry systems.

        [ ] Task 4.1: ONNX Batch Inference Integration

            [ ] 4.1.1: Initialize the InferenceSession.

            [ ] 4.1.2: Map C# state array into a single 2D ONNX Tensor [10000, state_dim] to execute Batch Inference.

        [ ] Task 4.2: P/Invoke Bridge (C# $\leftrightarrow$ C++)

            [ ] Implement unmanaged memory mapping based on the versioned State/Action Contract.

        [ ] Task 4.3: Mission-Critical WebSocket Server

            [ ] 4.3.1: Set up an Async WebSocket Server.

            [ ] 4.3.2: Define a Binary Packet Schema [Header(Version) | Payload | Checksum].

            [ ] 4.3.3: Implement Backpressure (Drop Stale Frames) to prevent server OOM during client lag.

        [ ] Task 4.4: Telemetry Logging & Episode Replay System

            [ ] 4.4.1: Log State Over Time: Write a high-performance logger (e.g., to Parquet or binary files) capturing the position, battery, and hazard states of all agents at specified intervals.

            [ ] 4.4.2: Dump Agent Decisions: Log the output actions (Thrust, Sleep, Toggle) from the ONNX model paired with their input states for debugging RL behaviors.

            [ ] 4.4.3: Offline Replay Mode: Implement a "Replay Engine" that bypasses the C++ Physics and ONNX models, streaming historical log files directly to the WebGPU frontend for post-simulation analysis.

###### Phase 5: WebGPU Dashboard (React Frontend)

    Visual presentation. High emphasis on Resource Lifecycle management to prevent VRAM leaks.

        [ ] Task 5.1: WebSocket Client & Decoder

            [ ] Write a useWebSocket hook to ingest the Binary stream, decode (checking Header version), and handle Replay Mode controls (Play/Pause/Scrub).

        [ ] Task 5.2: WebGPU Context & Resource Lifecycle (Critical)

            [ ] 5.2.1: Initialize the Device Context.

            [ ] 5.2.2: Lifecycle Manager: Ensure every GPUBuffer and GPUTexture has its .destroy() method called upon component unmount.

        [ ] Task 5.3: Environment Render

            [ ] Write WGSL Shaders to draw the Earth sphere with day/night illumination.

        [ ] Task 5.4: Instanced Rendering (The Swarm)

            [ ] Write a WGSL Shader utilizing @builtin(instance_index) to draw 10,000 satellites in a single Draw Call.

            [ ] Map colors dynamically (Blue = Nominal, Green = Active Payload, Red = Fatal Error).

        [ ] Task 5.5: Trajectory Trails

            [ ] Manage a Circular Buffer for orbital trails using line-strip topology.



