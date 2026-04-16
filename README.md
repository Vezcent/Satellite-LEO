# Multi-Agent System for Satellite Lifetime Optimization (S-MAS)

Project Overview

The S-MAS project focuses on developing a **High-Fidelity Satellite Operational Twin** utilizing Multi-Agent Reinforcement Learning (MARL) combined with Rule-Based Safety Governors. The system is specifically engineered to simulate the realistic, noisy lifecycle of the ESA PROBA-1 micro-satellite at a 600km LEO altitude.

The core objective is to optimize satellite longevity and mission efficiency within the harsh South Atlantic Anomaly (SAA) environment. It achieves this by balancing high-fidelity C++ physical computations, stochastic environmental impairments, and decentralized AI decision-making.

## 1. Optimized Data Pipeline

The system utilizes a multi-layered environmental dataset (2000–2020) optimized for a footprint of < 50MB while maintaining a strict 5.0s temporal resolution:

* **Atmospheric Layer (NRLMSISE-00):** Native C++ integration of the empirical thermospheric model, using F10.7 solar flux and Ap/Kp indices from NASA OMNIWeb to compute real-time density ($\rho$).
* **Radiation Hazard Map (SPENVIS):** Pre-computed 2D Static Heatmaps ($10890$ points) based on the AP-8 MAX model, providing integral proton fluxes for >10 MeV and >30 MeV energy levels.
* **Energy & Thermal Layer:** Analytical Ray-Tracing using a Cylindrical Shadow Model to calculate Beta angles and eclipse durations without external API reliance.
* **Communication Constraints:** Geometric Line-of-Sight (LoS) calculations based on Spherical Trigonometry and a $5^\circ$ elevation mask for ESA ground stations (Redu and Kiruna).

## 2. Satellite Engine & Physics Models (Inspired from Unreal Engine)

Developed as a high-performance shared library (C++/C) integrated with a C# orchestration layer, transitioning from pure training to mission-grade simulation:

* **Platform Specs (PROBA-1):** 94 kg launch mass, GaAs solar arrays (~90W), and the CHRIS (Compact High Resolution Imaging Spectrometer) optical payload.
* **Orbital Dynamics:** RK4 integration solver locked at $dt = 5.0s$, modeling aerodynamic drag, orbital decay, and J2 perturbations.
* **Stochastic Realism & Epistemic Uncertainty:** Includes injected Gaussian noise for sensors (Position, Velocity, Power), Actuator non-ideality ($\pm 5\%$ thruster error), random SEU (Single Event Upset) spikes, Action Execution Latency (simulating command queue delays), and continuous Physics Model Drift (e.g., stochastic drift in the drag coefficient $C_D$ over time to prevent AI overfitting to a perfect simulator).
* **Degradation Modeling:** Advanced Arrhenius-based chemical degradation for batteries factoring in charge/discharge cycles, alongside hardware degradation from radiation.
* **Failure Contract:** Episodes terminate upon battery depletion ($SoC \le 0\%$), prolonged telemetry loss ($> 72h$), or re-entry ($< 200km$).

## 3. Hybrid Multi-Agent Framework (CTDE + FDIR)

S-MAS utilizes a Centralized Training, Decentralized Execution (CTDE) strategy governed by a dynamic **Failure Detection, Isolation, and Recovery (FDIR)** state machine. The FDIR operates across 4 dynamic modes (NOMINAL, DEGRADED, SAFE, RECOVERY) based on real-time telemetry.

To prevent the AI from "reward hacking" the safety net (learning to rely on FDIR to save it), **FDIR interventions apply a massive negative reward penalty to the agents, and the current FDIR state is fully exposed within the AI's observation space.**

By default, the system implements the **MAPPO (Multi-Agent Proximal Policy Optimization)** algorithm as a robust baseline. The framework manages conflicting subsystem requirements through specialized sub-agents:

* **Navigation Agent (The Pilot):** Manages continuous 3D $\Delta V$ thruster burns to counteract drag and maintain orbit (subject to actuator noise).
* **Resource Agent (The Bus Manager):** Controls discrete power states, toggling "Deep Sleep" to protect the bus during solar storms or high-flux SAA crossings.
* **Mission Agent (The Payload Manager):** By default, executes a data collection mission. It optimizes the CHRIS instrument duty cycle via Binary states (ON/OFF). Crucially, the ON state actively consumes battery power. Users can later override this with custom mission logic.

The RL agents are trained utilizing potential-based reward shaping to maximize longevity survival and successful mission completions.

Reward Shaping: Explicitly weighted to penalize fuel waste, battery Depth of Discharge (DoD), and SAA-induced fatal upsets while rewarding mission coverage.

## 4. Export & Deployment

* **Embedded Inference:** AI policies are exported to ONNX (FP16) with dynamic axes, enabling high-speed batch inference via the ONNX Runtime C++ API.
* **Telemetry & Comm Delay:** The WebSocket server simulates real-world operations by injecting 1-10s communication delays and random packet losses.
* **WebGPU Dashboard:** A React-based visualization engine utilizing the WebGPU API for instanced rendering of satellite swarms (up to 1111 agents).

## Task Implementation Roadmap

The development pipeline is segmented into five core stages:

[x] **Task 0:** Virtual Environment & Dependency Management.
[x] **Task 1 & 2:** Raw Data Acquisition (OMNIWeb/SPENVIS), cleaning, and SAA Heatmap generation.
[X] **Task 3:** C++ Physics Core, Noise Injection, and Actuator Error modeling.
[ ] **Task 4:** MARL Framework construction (MAPPO) with specialized Reward Shaping.
[ ] **Task 5:** ONNX integration, C# FDIR Governor, and WebGPU real-time dashboard rendering.
[ ] **Task 6:** Validation with real PROBA-1 TLE historical data.
