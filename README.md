# S-MAS — Satellite Multi-Agent System for Lifetime Optimization

> High-fidelity LEO satellite operational twin powered by Multi-Agent Reinforcement Learning (MARL) and rule-based FDIR safety governors.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Clone & Environment Setup](#1-clone--environment-setup)
  - [2. Build the C++ Physics Engine](#2-build-the-c-physics-engine)
  - [3. Train the MARL Agents](#3-train-the-marl-agents)
  - [4. Run the C# Controller](#4-run-the-c-controller)
  - [5. Launch the WebGPU Dashboard](#5-launch-the-webgpu-dashboard)
- [Subsystem Details](#subsystem-details)
  - [Data Pipeline](#data-pipeline)
  - [Physics Engine](#physics-engine)
  - [MARL Framework](#marl-framework)
  - [FDIR Safety Governor](#fdir-safety-governor)
  - [Visualization](#visualization)
- [Testing](#testing)
- [License](#license)

---

## Overview

S-MAS simulates the realistic, noisy lifecycle of the **ESA PROBA-1** micro-satellite at **600 km LEO altitude**. It optimizes satellite longevity and mission efficiency within the harsh South Atlantic Anomaly (SAA) environment by coupling:

- **C++ physics core** — orbital mechanics (RK4 + J2), NRLMSISE-00 atmospheric drag, attitude dynamics, thermal modelling, and stochastic failure injection.
- **Python MARL training** — MAPPO with Centralized Training / Decentralized Execution (CTDE) and potential-based reward shaping.
- **C# orchestration** — real-time FDIR state machine, ONNX inference, and WebSocket telemetry server.
- **WebGPU dashboard** — procedural planet rendering, SAA radiation heatmap, and real-time multi-agent telemetry.

The simulation covers **10+ year degraded-lifetime scenarios** with battery cycling, solar-panel radiation damage, and propellant depletion.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WebGPU Dashboard (React + Vite)             │
│         Procedural Earth · SAA Heatmap · Live Telemetry         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket (JSON + Binary)
┌──────────────────────────▼──────────────────────────────────────┐
│              C# Controller (SmasController)                     │
│   FDIR Governor · ONNX Inference · Telemetry Server             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ P/Invoke (C ABI)
┌──────────────────────────▼──────────────────────────────────────┐
│              C++ Physics Engine (smas_engine.dll)                │
│   Orbital RK4 · Atmosphere · Radiation · Thermal · Attitude     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ctypes (shared memory)
┌──────────────────────────▼──────────────────────────────────────┐
│              Python MARL Training (MAPPO)                       │
│   3 Agents: Navigation · Resource · Mission                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Satellite LEO/
├── backend_cpp/              # C++ physics engine (shared library)
│   ├── CMakeLists.txt        # CMake build configuration (C++14)
│   ├── include/              # Public headers (contracts.h, types.h, …)
│   └── src/                  # Engine source files
│       ├── simulation_engine.cpp
│       ├── orbital_mechanics.cpp
│       ├── atmosphere.cpp
│       ├── satellite_bus.cpp
│       ├── thermal.cpp
│       ├── attitude.cpp
│       ├── stochastic.cpp
│       └── c_api.cpp         # C ABI exports for P/Invoke & ctypes
│
├── controller_csharp/        # C# orchestration & inference
│   ├── SmasController.csproj # .NET 10.0 project
│   ├── Program.cs            # Entry point & test suite
│   ├── AI/                   # ONNX inference engine
│   ├── Governor/             # FDIR state machine
│   ├── Interop/              # P/Invoke bindings to smas_engine
│   ├── Telemetry/            # WebSocket telemetry server
│   └── models/               # Exported ONNX policy files
│
├── marl_python/              # Python MARL training pipeline
│   ├── train.py              # Training entry point
│   ├── mappo.py              # MAPPO algorithm implementation
│   ├── env_wrapper.py        # Gymnasium environment wrapper
│   ├── observation.py        # 42-dim observation builder
│   ├── reward.py             # Potential-based reward shaping
│   ├── config.py             # Hyperparameters & dimensions
│   ├── export_onnx.py        # ONNX export (FP16, dynamic axes)
│   ├── validate_tle.py       # TLE historical validation
│   └── checkpoints/          # Saved model checkpoints
│
├── frontend_webgpu/          # WebGPU real-time dashboard
│   ├── package.json          # Vite + React + TypeScript
│   └── src/                  # UI components & WebGPU renderers
│
├── dataset/                  # Raw environmental datasets
├── preprocessed-data/        # Processed binary data files
├── tests/                    # Integration & regression tests
├── requirements.txt          # Python dependencies
├── Satellite LEO.sln         # Visual Studio solution
└── run_frontend.bat          # Quick-launch script (Windows)
```

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11 (64-bit) |
| **C++ Compiler** | MSVC 14.x+ (Visual Studio 2022/2025/2026) or GCC/Clang with C++14 |
| **CMake** | ≥ 3.16 |
| **Python** | 3.11+ |
| **.NET SDK** | 10.0+ |
| **Node.js** | 20+ (LTS) |
| **GPU** | WebGPU-compatible browser (Chrome 113+, Edge 113+) for the dashboard |

---

## Getting Started

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Vezcent/Satellite-LEO.git
cd Satellite-LEO

# Create a Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### 2. Build the C++ Physics Engine

```bash
cd backend_cpp
cmake -S . -B build
cmake --build build --config Release
```

This produces `smas_engine.dll` (Windows) or `libsmas_engine.so` (Linux) in the `build/` directory.

### 3. Train the MARL Agents

```bash
cd marl_python
python train.py
```

Training outputs are saved to `marl_python/checkpoints/`. After training, export the policies to ONNX:

```bash
python export_onnx.py
```

Exported models are written to `marl_python/onnx_export/` and should be copied to `controller_csharp/models/`.

### 4. Run the C# Controller

```bash
cd controller_csharp
dotnet run --configuration Release
```

The controller loads the ONNX models, initializes the physics engine via P/Invoke, runs the FDIR state machine, and starts a WebSocket telemetry server on `ws://localhost:6969`.

### 5. Launch the WebGPU Dashboard

```bash
cd frontend_webgpu
npm install
npm run dev
```

Open `http://localhost:5173` in a WebGPU-compatible browser. The dashboard connects to the controller's WebSocket feed automatically.

---

## Subsystem Details

### Data Pipeline

Multi-layered environmental data (2000–2020) optimized to < 50 MB at 5.0 s temporal resolution:

| Layer | Source | Description |
|-------|--------|-------------|
| **Atmospheric** | NRLMSISE-00 | Thermospheric density from F10.7 solar flux & Ap/Kp geomagnetic indices |
| **Radiation** | SPENVIS (AP-8 MAX) | Pre-computed 2D SAA heatmaps (10,890 grid points); proton flux > 10 MeV & > 30 MeV |
| **Energy / Thermal** | Analytical | Cylindrical shadow model for beta angles and eclipse durations |
| **Communication** | Geometric LoS | Spherical trigonometry with 5° elevation mask (Redu & Kiruna ground stations) |

### Physics Engine

- **Platform:** PROBA-1 — 94 kg, GaAs solar arrays (~90 W), CHRIS optical payload
- **Orbital Dynamics:** 3D ECI numerical integration (RK4) with J2 perturbations and calibrated multi-year decay curves
- **Attitude Control:** 3-axis quaternion rotational dynamics, reaction-wheel clusters, PD controller with 5× sub-stepping
- **Thermal Model:** Multi-node thermal network with eclipse cycling and radiative/conductive coupling
- **Observation Space:** 42 dimensions — orbit (7), power (4), environment (5), comms (2), FDIR (4), degradation (3), SEU (1), fuel (2), thermal (4), target altitude (1), ADCS (5), lag features (4)
- **Failure Conditions:** Battery depletion (SoC ≤ 0%), telemetry loss (> 72 h), re-entry (< 200 km)

### MARL Framework

**MAPPO** with Centralized Training / Decentralized Execution (CTDE):

| Agent | Role | Action Space |
|-------|------|-------------|
| **Navigation** | Orbit maintenance via 3D ΔV thruster burns | Continuous (3D) |
| **Resource** | Power management & Deep Sleep activation | Discrete |
| **Mission** | CHRIS instrument duty-cycle optimization | Binary (ON/OFF) |

- Independent policy trunks prevent task interference during high-dimensional training
- Potential-based reward shaping penalizes fuel waste, battery DoD, and SAA-induced upsets
- FDIR interventions apply massive negative reward penalties to prevent reward hacking

### FDIR Safety Governor

Four dynamic operational modes based on real-time telemetry:

```
NOMINAL ──→ DEGRADED ──→ SAFE ──→ RECOVERY ──→ NOMINAL
                                     ↑
                              (auto-recovery)
```

The FDIR state is fully exposed within the agent observation space to ensure policy awareness of safety boundaries.

### Visualization

- **WebGPU rendering** — procedural planet with specular ocean glint, atmospheric bloom
- **SAA radiation heatmap** — real-time overlay from SPENVIS data
- **Instanced rendering** — supports up to 1,111 simultaneous satellite agents
- **Binary telemetry** — `StatePacket` (222 bytes, packed struct) decoded and visualized in real time

---

## Testing

```bash
# C++ engine tests
cd backend_cpp/build
./smas_test                          # or .\Release\smas_test.exe on Windows

# Python regression suite
cd tests
python regression_suite.py

# C# integration tests (built into Program.cs)
cd controller_csharp
dotnet run --configuration Release   # runs assertion suite on startup
```

---

## License

This project is developed for academic and research purposes.  
All rights reserved © 2024–2026.
