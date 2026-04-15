
# IMPLEMENTATION PIPELINE: MULTI-AGENT SYSTEM FOR SATELLITE LIFETIME OPTIMIZATION (S-MAS)

## 0. CREATE VENV, THIS IS VERY IMPORTANT

### 1. System Overview

The objective is to develop an autonomous satellite simulation and control system utilizing Multi-Agent Reinforcement Learning (MARL). The system integrates high-fidelity physical computations (C++/C/C#) with flexible AI training environments (Python) and real-time 3D visualization (WebGPU).

Core Paradigm: The primary focus is Heterogeneous Subsystem Control for a single satellite, where the "Multi-Agent" aspect refers to internal subsystems (Navigation, Resource, Mission) negotiating survival. Swarm scaling (10,000 agents) is an optional stress-test utilizing Shared Policies and Batch Inference.
Global Physics Clock: The system strictly adheres to a global integration time step of dt = 5.0 seconds to ensure mathematical consistency across all C++ physics solvers and Python reward evaluations.

#### 2. Data Pipeline (Tasks 1 & 2)

Goal: Collect 15-20 years of environmental data (2000–2020) while maintaining a storage footprint of < 50GB.

#### 2.1. Data Reduction & Optimization

Strategy: To maintain a high-fidelity simulation within a $\leq 50 \text{ GB}$ storage constraint for a 15-year dataset, the following multi-layered reduction strategies are applied, specifically localized to the South Atlantic Anomaly (SAA) region ($0^\circ \text{ to } 50^\circ\text{S}$, $90^\circ\text{W} \text{ to } 40^\circ\text{E}$).

#### 2.1.1. Atmospheric Dynamics: Space Weather Driven Models (NRLMSISE-00)

Instead of relying on tropospheric/stratospheric models like ERA5, the simulation utilizes the NRLMSISE-00 empirical atmospheric model, natively integrated into the C++ physics core. This accurately captures thermospheric density variations at LEO altitudes 600km) driven by solar activity.

    Strategy: Eliminating the need to download massive multi-level atmospheric cubes. Density ($\rho$) is dynamically computed at runtime based on the satellite's exact State Vector (position and time) and historical space weather indices.

    Data Inputs: Lightweight CSV timeseries containing F10.7 (Solar Radio Flux) and Ap/Kp indices (Geomagnetic Activity), sourced from NOAA/SWPC.

    Storage Optimization: Reduces atmospheric data footprint from terabytes of GRIB files to less than 10MB of indexed CSV data for the entire 20-year span.

#### 2.1.2. Energy & Thermal Layer: Analytical Ray-Tracing

Given that LEO satellites receive direct solar radiation unattenuated by weather systems, the reliance on ground-based irradiance APIs (like NASA POWER) is removed.

    Strategy: Solar incidence and thermal loading are calculated using an Analytical Ray-Tracing Algorithm (Cylindrical Shadow Model) intersecting the Earth's umbra and penumbra.

    Parameters Computed:
        * Beta Angle: The angle between the orbital plane and the solar vector.
        * Eclipse Duration: Precise calculation of the time spent in the Earth's shadow to simulate battery discharge cycles and thermal stress on the satellite bus.
    
    Data Inputs: Requires only standard Two-Line Elements (TLEs) and high-precision Solar Ephemeris data, computed locally within the simulation engine.

#### 2.1.3. Communication Constraints: Point-to-Point Line-of-Sight (LoS)

Instead of downloading massive SRTM Topographical Elevation maps for large coastal areas, the model focuses strictly on geometric constraints relevant to a 600km orbit, where Earth's curvature dominates signal blockage.

    Strategy: Implementing a Spherical Trigonometry-based LoS calculation combined with an Elevation Mask (typically $5^\circ$ to $10^\circ$).

    Data Inputs: The system only requires the precise GPS Coordinates (Latitude, Longitude, Altitude) of specific Ground Stations.

    Optimization: This approach entirely eliminates the need for SRTM DEM tiles, freeing up significant memory.
    
#### 2.1.4. Space Environment & Radiation: SAA Localized Heatmaps

This layer is the core of SAA (South Atlantic Anomaly) simulation, focusing on the Trapped Proton Flux and Geomagnetic Anomalies that drive hardware failure (Single Event Upsets - SEUs) and solar panel degradation.

    Strategy: The system utilizes pre-computed Static Radiation Heatmaps specifically generated for the 600km altitude shell to act as a spatial hazard map for the agents.

    Radiation Focus: 2D grid lookups (Latitude vs. Longitude) for high-energy integral proton fluxes ($>10 \text{ MeV}, >30 \text{ MeV}, which the MAS Agents query based on their current position to trigger defensive states.

#### 2.2. Preprocessing & Feature Engineering

To ensure the Multi-Agent System (MAS) can effectively process real-time physics data, the preprocessing pipeline focuses on temporal synchronization, state normalization, and strict data contracts:

#### 2.2.1. Multi-Dimensional Risk Mapping & Contracts

SEU Probability Modeling: Mapping the pre-computed Static 2D Proton Flux Heatmaps directly to Single Event Upset (SEU) rates.
Versioning Contract: All State and Action schemas passed between C++, C#, and Python include a `uint8_t version` header to prevent memory corruption and silent bugs during serialization.

#### 2.2.2. Temporal Alignment & Simulation Stepping

Continuous Interpolation: Historical space weather data are linearly interpolated to match the strict `dt = 5.0s` integration steps of the C++ orbital engine.

Lag-Feature Creation: Generating 3-hour and 6-hour "look-back" and "look-ahead" windows for solar indices.

#### 2.2.3. VRAM Optimization & Replay Buffer Scaling

Precision Scaling: Converting all non-critical physics observations from Float64 (C++ native) to Half-Precision (FP16) before loading them into the PyTorch tensors.

#### 2.2.4. Spatial Normalization for MAS

Hazard-Relative Coordinates: Positions are transformed into a Localized Hazard-Relative frame (e.g., angular distance to the SAA centroid, time-to-eclipse).
Robust Feature Scaling: Applying Min-Max scaling for bounded variables (Battery SoC $0-100\%$) and Robust Scaling for unbounded variables ($F_D$).

##### 3. Three-Phase Execution Pipeline

    The simulation employs a Centralized Training, Decentralized Execution (CTDE) framework, structured into a modular

    3-Phase pipeline.

        C#: Manages high-level data logic, MARL batch-inference orchestration, and the WebSocket telemetry server.

        C++/C: Physics core (Shared Library) for high-performance execution.

        WebGPU: High-performance compute and rendering pipeline.

PHASE 1: BUILD ENGINE (Simulation Environment & Physics Backend)

    This phase acts as the "Ground Truth" generator.

##### 3.1.1 Hardware Constraints: PROBA-1 (Autonomous Micro-Satellite Platform)

NORAD Catalog ID: 26957 (Launch Year: 2001).

Basic Specs: 94 kg Launch Mass, Sun-Synchronous Orbit (SSO) at ~600 km. Dimensions: 600 x 600 x 800 mm.

Aerodynamic Area ($A$): ~0.36 m² (Used for Drag calculation $F_D$).

Power Subsystem: Gallium Arsenide (GaAs) Solar Arrays providing ~90W. Lithium-ion battery for Eclipse survival.

Radiation Resiliency: Triple Modular Redundancy (TMR) OBC, ECC/MRAM non-volatile storage to support onboard autonomy.

##### 3.1.2 Core Physics & Failure Contract

A. Atmospheric Drag & Orbital Decay:
    $$F_D = -\frac{1}{2} \rho A C_D v^2$$
    Calculated using the NRLMSISE-00 model and integrated using an RK4 solver locked at `dt = 5.0s`.

B. Power Dynamics & Thermal Degradation:
    $$t_{min} = \max(t_{xmin}, t_{ymin}, t_{zmin}), \quad t_{max} = \min(t_{xmax}, t_{ymax}, t_{zmax})$$
    Arrhenius model calculates the chemical degradation rate:
    $$k = A \exp\left(-\frac{E_a}{R T}\right)$$

C. Failure Contract (Terminal States):
    The simulation strictly defines the satellite as "Dead" (Episode Done = True) under three conditions:
    1. Power Failure: Battery SoC drops to $\le 0\%$.
    2. Telemetry Loss: Loss of LoS with Ground Stations for $> 72$ continuous hours.
    3. Re-entry: Altitude drops below $200 \text{ km}$.

PHASE 2: CUSTOM MAS (User Deployment & Agent Roles)

    Heterogeneous agents coordinate to manage survival subsystems. To support swarm scaling, agents use a Shared Policy Architecture and Batch Inference.

##### 3.2.1 Agent Roles & Action Contracts

Navigation Agent (The Pilot): Computes $\Delta V$ thruster burns.
    Action Contract: Continuous 3D Vector $[-1.0, 1.0]$ for attitude/throttle.

Resource Agent (The Bus Manager): Controls the "Deep Sleep" toggle.
    Action Contract: Discrete/Binary $[0, 1]$ evaluated via an Argmax/Sigmoid threshold ($>0.5$).

##### 3.2.2 Explicit Reward Shaping

To prevent "Reward Hacking", the baseline survival reward utilizes explicit weights to balance competing objectives:

    $$R_t = w_1(1.0) - w_2(\Delta V_{used}) - w_3(\text{DoD}) - w_4(P_{fatal})$$

    Where: $w_1 = 1.0$ (Alive bonus), $w_2 = 5.0$ (Fuel penalty), $w_3 = 2.0$ (Battery Depth of Discharge penalty), and $w_4 = 1000.0$ (Massive penalty for triggering the Failure Contract).

PHASE 3: CUSTOM MISSION (Optional Opportunistic Coverage)

    The mission layer is an opt-in behavioral layer.

##### 3.3.1 Mission Execution Logic

    Mission Agent: Decides when to toggle the optical payload (CHRIS instrument) ON or OFF.
    
    Action Contract: Discrete/Binary $[0, 1]$.

##### 3.3.2 Dynamic Reward Shaping for Phase 3

    The mission reward balances opportunistic data collection against radiation risk:

    $$R_t = R_{survival} + \big( S_{payload} \cdot r_{coverage} \big) - \big( S_{payload} \cdot P_{fatal\_risk} \big)$$

    Explicit Weights: $r_{coverage} = +50.0$ (Valid Target Imaged), $P_{fatal\_risk} = -500.0$ (Payload ON inside SAA boundaries).

##### 3.3.3 Custom Algorithm Interface (custom_algorithm.py)

    This file coordinates Phase 2 and 3 using vectorized operations to support scaling from 1 to 10,000 agents seamlessly.

sample:

Python
import numpy as np
import torch
import torch.nn as nn

class SatelliteMARLController:
    """
    Optimized for Shared Policy and Batch Inference (evaluating thousands of agents simultaneously).
    """
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = {
            "nav": self._build_agent("navigation"),  # Phase 2: Manages Thrust
            "bus": self._build_agent("resource"),    # Phase 2: Manages Deep Sleep
            "mission": self._build_agent("mission")  # Phase 3: Manages Payload
        }

    def get_actions(self, batched_physics_state):
        """
        batched_physics_state: Dict of Tensors from Phase 1. Shape: [batch_size, feature_dim]
        """
        with torch.no_grad():
            # Vectorized inference for N agents simultaneously
            thrust_intent = self.agents["nav"].act(batched_physics_state["orbit"])
            sleep_intent = self.agents["bus"].act(batched_physics_state["power"])
            payload_toggle = self.agents["mission"].act(
                state=batched_physics_state["hazard"], 
                power_context=batched_physics_state["power"]
            )

        # Meta-Coordination (Software-Defined Resiliency) - VECTORIZED
        # Overrides Payload to 0.0 for ANY specific agent where sleep_intent > 0.5
        payload_toggle = torch.where(
            sleep_intent > 0.5, 
            torch.zeros_like(payload_toggle), 
            payload_toggle
        )

        return {
            "thrust_vector": thrust_intent,
            "deep_sleep_flag": sleep_intent,
            "payload_toggle": payload_toggle
        }

    def export_to_onnx(self, path_prefix="satellite_model"):
        """
        Exporting to ONNX with FP16 and Dynamic Axes for scalable Batch Inference.
        """
        for name, agent in self.agents.items():
            dummy_input = agent.get_dummy_input().to(self.device).half()
            torch.onnx.export(
                agent.model.half(), 
                dummy_input, 
                f"{path_prefix}_{name}.onnx",
                opset_version=14,
                do_constant_folding=True,
                input_names=['state_input'],
                output_names=['action_output'],
                dynamic_axes={'state_input': {0: 'batch_size'}, 'action_output': {0: 'batch_size'}}
            )

###### 4. Export & Deployment

This final stage realizes the Decentralized Execution phase.

###### 4.1 Model Export & Batch Inference Integration

C++ Inference Engine: The ONNX Runtime C++ API executes the AI policies. To maximize GPU utilization, C# packs agent states into a single 2D Tensor `[batch_size, state_dim]` for Batch Inference.

###### 4.2. Telemetry, Logging & WebSocket Server

To support Mission-Critical telemetry and Offline RL Debugging:
    Binary Packet Schema: Uses a strict `[Header(Version) | Payload | Checksum]` byte array to eliminate JSON parsing overhead.
    Backpressure & Culling: The server implements a 'Drop Stale Frames' strategy. If the client lags, old frames are dropped to prevent server Out-of-Memory (OOM) crashes.
    Logging & Replay: States and ONNX decisions are dumped to high-performance log files (e.g., Parquet). An Offline Replay Mode allows streaming these logs directly to the frontend bypassing physics calculations.

###### 4.3. WebGPU Dashboard & Resource Lifecycle

Frontend Architecture: A React-based web dashboard utilizing the WebGPU API for Instanced Rendering (up to 10,000 agents in a single draw call).
Resource Lifecycle Management (Critical): To prevent VRAM memory leaks, the architecture guarantees that every `GPUBuffer` and `GPUTexture` invokes its `.destroy()` method upon React component unmounting.
State-Driven Rendering: Colors dynamically reflect the State Contract: Blue (Nominal)
