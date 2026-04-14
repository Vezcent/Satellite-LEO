# IMPLEMENTATION PLAN (ACTIONABLE TASKS): S-MAS PROJECT

    [General Scientific Disclaimer: Training vs. Mission-Grade]
    The C++ Physics Engine developed in this project is explicitly designed as a "Training Approximation" environment. The priority is to achieve high-throughput simulation (millions of steps per second) necessary for MARL convergence. While equations are grounded in real physics, the models (Atmosphere, Radiation, Degradation) utilize lightweight approximations and static lookups. This system is optimized for training AI policies and is not a substitute for mission-grade, high-fidelity orbital mechanics software (e.g., AGI STK or NASA GMAT).

## Phase 0: System Setup & Architecture

    Setting up the environment to enable interoperability between C++, C#, Python, and the React frontend.

        [ ] Task 0.1: Initialize the Git Repository and standard directory structure (/backend_cpp, /controller_csharp, /marl_python, /frontend_webgpu).

        [ ] Task 0.2: Set up the Python virtual environment (venv) and install core dependencies (torch, numpy, onnx,...).

        [ ] Task 0.3: Initialize the C++ project with CMake (crucial for building the shared library .dll for Windows or .so for Linux).

        [ ] Task 0.4: Initialize the C# project (Console App or ASP.NET Core) and configure P/Invoke for C++ interoperability.

        [ ] Task 0.5: Initialize the React project using Vite (with TypeScript support).

## Phase 1: Build Engine (C++ Physics Core Backend)

    Developing the ultra-lightweight physics core to parse static files and compute the system State.

        [ ] Task 1.1: Data Parsers & Static Heatmaps

            [ ] 1.1.1: Write C++ CSV parsers for F10.7 and Kp/Ap indices (NOAA).

            [ ] 1.1.2: Write C++ parser for the SPENVIS AP-8 2D Static Heatmap (Proton Flux).

            [Scientific Disclaimer: Static Heatmap] The SAA radiation heatmap is a static 2D grid representing average historical proton flux. It ignores real-time micro-fluctuations, solar particle events (SPEs), and localized space debris. It is sufficient for teaching agents spatial hazard avoidance but lacks the temporal dynamism of actual radiation belts.

        [ ] Task 1.2: Math & Geometry Engine

            [ ] 1.2.1: Implement the Analytical Ray-Box Intersection algorithm to calculate eclipse periods.

            [ ] 1.2.2: Implement Spherical Trigonometry to check Line-of-Sight (LoS) with an Elevation Mask of $5^\circ$.

        [ ] Task 1.3: Dynamics & Orbital Decay Implementation

            [ ] 1.3.1: Embed the open-source NRLMSISE-00 atmospheric model to calculate density ($\rho$) dynamically.

            [ ] 1.3.2: Implement the Aerodynamic Drag equation ($F_D = -0.5 \cdot \rho \cdot A \cdot C_D \cdot v^2$) and J2 perturbations.

            [ ] 1.3.3: Implement an ODE Integrator (e.g., Runge-Kutta 4 or Euler method) to update the satellite's position, velocity, and altitude at each time step based on $F_D$
        
        [ ] Task 1.4: Power & Hardware Degradation Subsystem

            [ ] 1.4.1: Program the SatelliteBus class to manage State of Charge (SoC) and Power Budget (charge vs. discharge).

            [ ] 1.4.2: Implement the Arrhenius Model (thermal cycling degradation) and Radiation Power Fade mathematics.

        [ ] Task 1.5: C-API Export

            [ ] Write extern "C" functions (UpdatePhysics(), GetState()) to expose data to C#.

### Phase 2 & 3: Custom MAS & Mission Logic (Python)

    Constructing the MARL training environment, preprocessing pipelines, and Agent logic.

        [ ] Task 2.1: Data Preprocessing & Look-ahead Features

            [ ] 2.1.1: Write linear interpolation functions to synchronize 3-hour Kp/Ap data to the C++ tick rate.

            [ ] 2.1.2: Generate temporal "Look-ahead" (future) and "Look-back" (historical) feature vectors for solar indices.

            [Scientific Disclaimer: Look-ahead Features] Providing agents with perfect future F10.7/Kp data assumes the satellite receives flawless space weather forecast uplinks from Earth. In reality, these forecasts carry prediction errors and uplink latency, which the simulation abstracts away for training stability.

        [ ] Task 2.2: Observation Spaces & Normalization

            [ ] Define the State Space, apply Data Normalization (Min-Max, Robust Scaling), and cast observations to FP16 precision.

        [ ] Task 2.3: Reward Shaping Implementation

            [ ] Code the Potential-based Survival Reward ($R_{survival}$) based on battery/fuel health.

            [ ] Code the Conditional Mission Reward ($r_{coverage}$) and the severe penalty ($P_{fatal\_risk}$) for payload activation in hazard zones.
        
        [ ] Task 2.4: Agent Neural Networks (custom_algorithm.py)

            [ ] Construct PyTorch architectures for Nav, Bus, and Mission Agents.

            [ ] Implement the Software-Defined Resiliency constraint (e.g., Bus sleep signal forces Mission Payload to 0).

        [ ] Task 2.5: ONNX Export Pipeline

            [ ] Write a script to quantize model weights (FP32 $\rightarrow$ FP16) and export the trained agents to .onnx files.

#### Phase 4A: Export & Controller (C# Execution Bridge)

    Orchestrating the C++ Core, executing AI Models via ONNX, and streaming telemetry.

        [ ] Task 4.1: ONNX Runtime C++ / C# Integration

            [ ] 4.1.1: Install the Microsoft.ML.OnnxRuntime package.

            [ ] 4.1.2: Initialize the InferenceSession to load .onnx models into memory.

            [ ] 4.1.3: Write mapping functions to convert C# float[] state arrays into Tensor<float> inputs for ONNX.

            [ ] 4.1.4: Extract and cast the ONNX output tensors back into actionable C# commands (Thrust, Sleep, Toggle).

        [ ] Task 4.2: P/Invoke Bridge (C# $\leftrightarrow$ C++)

            [ ] Declare [DllImport] and [StructLayout] to map C++ unmanaged memory to C#.

            [ ] Write the Main Execution Loop: C++ Physics Step $\rightarrow$ Get State $\rightarrow$ ONNX Inference $\rightarrow$ Send Action to C++.

        [ ] Task 4.3: High-Performance WebSocket Server

            [ ] 4.3.1: Set up an Async TCP/WebSocket Server using a lightweight library (e.g., WatsonWebsocket or Fleck).

            [ ] 4.3.2: Implement a Binary Serializer. Avoid JSON parsing overhead by packing data into compact byte arrays (e.g., [AgentID (4 bytes) | X (4 bytes) | Y (4 bytes) | Z (4 bytes) | StatusFlag (1 byte)]).

            [ ] 4.3.3: Manage a thread-safe broadcast loop to push binary states to clients at 30-60 Hz.

#### Phase 4B: WebGPU Dashboard (React Frontend)

    Visualizing the massive multi-agent swarm purely as a high-performance "Dumb Renderer".

        [ ] Task 4.4: WebSocket Client & Buffer Management

            [ ] 4.4.1: Write a useWebSocket hook to ingest the Binary stream and decode it into a Float32Array.

            [ ] 4.4.2: Maintain and update the unified State Buffer for GPU memory uploads.

        [ ] Task 4.5: WebGPU Setup & Environment Rendering

            [ ] 4.5.1: Request navigator.gpu.requestAdapter() and initialize the Device Context on the HTML Canvas.

            [ ] 4.5.2: Configure the Pipeline Layout, Camera projection matrices, and OrbitControls.

            [ ] 4.5.3: Write WGSL Vertex/Fragment Shaders to render the textured Earth sphere and day/night illumination.

        [ ] Task 4.6: Instanced Rendering (The Swarm)

            [ ] 4.6.1: Write a WGSL Shader utilizing @builtin(instance_index) to draw a single low-poly satellite model up to 10,000 times in a single draw call.

            [ ] 4.6.2: Bind the State Buffer to dynamically update instance positions and colors (Blue = Nominal, Green = Payload ON, Red = Fatal).

        [ ] Task 4.7: Trajectory Trails

            [ ] 4.7.1: Implement a Circular Buffer on the frontend to track the last 10-20 position vectors of each active agent.

            [ ] 4.7.2: Utilize line-strip topology in WebGPU to render short fading orbital trails behind the agents.

            
