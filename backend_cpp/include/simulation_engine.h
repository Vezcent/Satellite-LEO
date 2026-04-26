/*
 * S-MAS: Simulation Engine — the master orchestrator.
 * Integrates all Phase 1 subsystems into a single step() call.
 */
#pragma once
#include "types.h"
#include "constants.h"
#include "contracts.h"
#include "data_loader.h"
#include "atmosphere.h"
#include "geometry.h"
#include "orbital_mechanics.h"
#include "satellite_bus.h"
#include "stochastic.h"
#include <string>

namespace smas {

struct EngineConfig {
    std::string data_dir;           // path to preprocessed-data/
    uint64_t    seed          = 42; // master RNG seed
    double      max_dv_per_step = 0.01; // max thrust delta-v per step (m/s)
    bool        enable_noise  = true;
    bool        enable_drift  = true;
    bool        enable_seu    = true;
    bool        enable_delay  = true;
    double      density_multiplier = 1.0;
};

class SimulationEngine {
public:
    explicit SimulationEngine(const EngineConfig& cfg);

    // ── Lifecycle ─────────────────────────────────────────────────
    bool  init();       // Load data files, return false on failure
    void  reset();      // Reset to initial conditions for new episode
    void  set_time(double time_s); // Jump to a specific time
    void  set_degradation(double capacity_j, double panel_eff); // Age the satellite
    StatePacket step(const ActionPacket& action); // Advance one dt

    // ── Accessors ─────────────────────────────────────────────────
    const StatePacket& current_state() const { return state_; }
    double sim_time()                  const { return orbit_.time; }
    bool   is_done()                   const { return state_.is_done != 0; }

private:
    EngineConfig cfg_;

    // ── Data ──
    SpaceWeatherTable weather_;
    SAAHeatmap        saa_;
    GroundStationList gs_list_;
    TLEParser         tle_;

    // ── Physics ──
    NRLMSISEModel     atmosphere_;
    OrbitalState      orbit_;
    SatelliteBus      bus_;

    // ── Stochastic ──
    SensorNoise       sensor_noise_;
    SEUGenerator      seu_gen_;
    ActuatorModel     actuator_;
    ModelDrift         drift_;

    // ── State tracking ──
    StatePacket       state_;
    SimTime           sim_time_struct_;
    double            time_since_contact_;
    FDIRMode          fdir_mode_;

    // ── Internal helpers ──
    void update_time();
    void update_environment();
    void update_fdir();
    SpaceWeatherRecord get_current_weather() const;
    double compute_local_solar_time(double lon_deg) const;
};

} // namespace smas
