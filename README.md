# Multi-Agent System for Satellite Lifetime Optimization (S-MAS)

## Project Overview

The S-MAS project focuses on developing an autonomous satellite simulation and control system using Multi-Agent Reinforcement Learning (MARL). Designed to simulate a 15-year extended operation of the VNREDSat-1 (and custom satellites) within the harsh South Atlantic Anomaly (SAA) environment, the system incorporates a hybrid architecture (C++/C/C# and Python) to balance high-performance physics computation with flexible AI training.

## 1. Data Pipeline & Optimization

A highly optimized, sub-50GB data pipeline provides real-time multi-dimensional environmental inputs:

- **Atmospheric Layer (ERA5):** High-altitude pressure/temperature subsets (6-hourly).
- **Energy/Thermal Layer (NASA POWER):** Localized irradiance coordinates for ground stations and nodes.
- **Topographical Layer (SRTM):** 90m and 30m targeted tiles around ground stations.
- **Space Weather (NOAA SWPC / OMNI):** Proton flux and geomagnetic indices (Kp, Dst) to determine
Single Event Upset (SEU) rates.
Data is aligned temporarily, compressed to NetCDF4, and scaled using FP16/quantization techniques to fit within VRAM constraints.

## 2. Satellite Engine & Physics Models

Developed with a C++ physics core and C# UI, the engine models critical components and physical perturbations:

- **Radiation-Hardening & Hardware Specs:** Incorporates TMR, shielding models, error-correcting memory, and smart-discharge battery cycling.
- **Physics Models:** Realistic orbital decay (aerodynamic drag with ERA5 data), J2 oblateness perturbations, and high-fidelity thermal & radiation-induced component degradation over a 15-year horizon.
- **Graphics & Ray Tracing:** Uses an optimized Oriented Bounding Box (OBB) ray tracing model for solar incidence and shadowing, with an option to toggle headless training for faster AI execution.

## 3. Multi-Agent Framework (CTDE)

S-MAS utilizes a Centralized Training, Decentralized Execution (CTDE) strategy, managing conflicting subsystem requirements through four specialized sub-agents coordinating within `SatelliteMARLController`:

1. **Space Agent (The Shield):** Predicts radiation doses and triggers safe-mode shutdowns.
2. **Atmosphere Agent (The Navigator):** Maps drag/density to optimize orbital maneuvers and save fuel.
3. **Ground Agent (The Communicator):** Detects Line-of-Sight transmission windows amidst terrain/noise constraints.
4. **Decision Agent (The Strategist):** Aggregates cross-layer inputs to balance mission operations with long-term hardware survival goals.

The RL agents are trained utilizing potential-based reward shaping to maximize longevity survival and successful mission completions.

## 4. Export & Deployment

The finalized system is built for real-world embedded operations and monitoring:

- **Embedded AI:** Agents evaluate decisions onboard using ONNX Runtime (C++) optimized at FP16.
- **Real-Time Dashboard:** A responsive React and Three.js web application utilizing WebSockets to plot orbital vectors, telemetry, and 3D scenes in the browser directly from the back-end engine.

## Task Implementation Roadmap

The development pipeline is segmented into five core stages:

- [x] **Task 0:** Virtual Environment Initialization.
- [ ] **Task 1 & 2:** Raw Data acquisition, geographic limiting, normalization, and optimization.
- [ ] **Task 3:** Core physics implementation, modeling satellite hardware behavior under degradation, and ray-box math.
- [ ] **Task 4:** Constructing the Multi-Agent Reinforcement Learning framework with specialized observation layers and decision-conflict resolution.
- [ ] **Task 5:** Model export to ONNX runtime integration and live WebSocket web-dashboard rendering.
