# Multi-Agent System for Satellite Lifetime Optimization (S-MAS)

Project Overview

The S-MAS project focuses on developing an autonomous satellite simulation and control system utilizing Multi-Agent Reinforcement Learning (MARL). The system is specifically engineered to simulate the lifecycle of the ESA PROBA-1 micro-satellite at a 600km LEO altitude.

The core objective is to optimize satellite longevity and mission efficiency within the harsh South Atlantic Anomaly (SAA) environment by balancing high-fidelity C++ physical computations with decentralized AI decision-making.

## 1. Optimized Data Pipeline

The system utilizes a multi-layered environmental dataset (2000–2020) optimized for a footprint of < 50MB while maintaining a strict 5.0s temporal resolution:

Atmospheric Layer (NRLMSISE-00): Native C++ integration of the empirical thermospheric model, using F10.7 solar flux and Ap/Kp indices from NASA OMNIWeb to compute real-time density ($\rho$).

Radiation Hazard Map (SPENVIS): Pre-computed 2D Static Heatmaps ($10890$ points) based on the AP-8 MAX model, providing integral proton fluxes for >10 MeV and >30 MeV energy levels.

Energy & Thermal Layer: Analytical Ray-Tracing using a Cylindrical Shadow Model to calculate Beta angles and eclipse durations without external API reliance.Communication Constraints: Geometric Line-of-Sight (LoS) calculations based on Spherical Trigonometry and a $5^\circ$ elevation mask for ESA ground stations (Redu and Kiruna).

## 2. Satellite Engine & Physics Models (Inspired from Unreal Engine)

Developed as a high-performance shared library (C++/C) integrated with a C# orchestration layer:

Platform Specs (PROBA-1): 94 kg launch mass, GaAs solar arrays (~90W), and the CHRIS (Compact High Resolution Imaging Spectrometer) optical payload.

Orbital Dynamics: RK4 integration solver locked at $dt = 5.0s$, modeling aerodynamic drag, orbital decay, and J2 perturbations.Failure Contract: Episodes terminate upon battery depletion ($SoC \le 0\%$), prolonged telemetry loss ($> 72h$), or re-entry ($< 200km$).

Degradation Modeling: Arrhenius-based chemical degradation for batteries and SEU (Single Event Upset) probability mapping for onboard electronics.

## 3. Multi-Agent Framework (CTDE)

S-MAS utilizes a Centralized Training, Decentralized Execution (CTDE) strategy, managing conflicting subsystem requirements through four specialized sub-agents coordinating within `SatelliteMARLController`:

Navigation Agent (The Pilot): Manages continuous 3D $\Delta V$ thruster burns to counteract drag and maintain orbit.

Resource Agent (The Bus Manager): Controls discrete power states, toggling "Deep Sleep" to protect the bus during solar storms or high-flux SAA crossings.

Mission Agent (The Payload Manager): Optimizes the CHRIS instrument duty cycle, balancing opportunistic imaging rewards against radiation-induced hardware risk.

The RL agents are trained utilizing potential-based reward shaping to maximize longevity survival and successful mission completions.

Reward Shaping: Explicitly weighted to penalize fuel waste, battery Depth of Discharge (DoD), and SAA-induced fatal upsets while rewarding mission coverage.

## 4. Export & Deployment

Embedded Inference: AI policies are exported to ONNX (FP16) with dynamic axes, enabling high-speed batch inference via the ONNX Runtime C++ API.

WebGPU Dashboard: A React-based visualization engine utilizing the WebGPU API for instanced rendering of satellite swarms (up to 10,000 agents) with real-time binary telemetry streaming.

## Task Implementation Roadmap

The development pipeline is segmented into five core stages:

[v] Task 0: Virtual Environment & Dependency Management.

[v] Task 1 & 2: Raw Data Acquisition (OMNIWeb/SPENVIS), cleaning, and SAA Heatmap generation.

[ ] Task 3: C++ Physics Core implementation (RK4, NRLMSISE-00, and Analytical Shadowing).

[ ] Task 4: MARL Framework construction with specialized Reward Shaping for PROBA-1 subsystems.

[ ] Task 5: ONNX integration and WebGPU real-time dashboard renderi
