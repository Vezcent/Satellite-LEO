/*
 * S-MAS: Test Driver / Standalone Simulation
 * Runs the engine for N steps to validate all subsystems.
 */
#include "simulation_engine.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════╗\n"
              << "║  S-MAS : Satellite Multi-Agent Simulation Engine  ║\n"
              << "║  Phase 1 — Physics Core Validation                ║\n"
              << "╚════════════════════════════════════════════════════╝\n\n";

    // Default data directory (relative to project root)
    std::string data_dir = "../preprocessed-data";
    if (argc > 1) data_dir = argv[1];

    int total_steps = 17280; // ~1 day at dt=5s (86400/5)
    if (argc > 2) total_steps = std::atoi(argv[2]);

    // ── Create & initialise engine ────────────────────────────────
    smas::EngineConfig cfg;
    cfg.data_dir = data_dir;
    cfg.seed = 42;

    smas::SimulationEngine engine(cfg);

    if (!engine.init()) {
        std::cerr << "ERROR: Failed to initialise engine. Check data paths.\n";
        return 1;
    }

    engine.reset();
    std::cout << "Engine initialised. Running " << total_steps << " steps "
              << "(dt=" << smas::constants::DT << "s, ~"
              << (total_steps * smas::constants::DT / 3600.0) << " hours).\n\n";

    // ── Simulation loop ───────────────────────────────────────────
    smas::ActionPacket action{};
    action.version   = 1;
    action.thrust_x  = 0.0f;
    action.thrust_y  = 0.0f;
    action.thrust_z  = 0.0f;
    action.throttle  = 0.0f;
    action.deep_sleep = 0;
    action.payload_on = 0;

    // Header
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Step   | Time(h) | Alt(km) |  Lat  |  Lon   | SoC(%) | ρ(kg/m³)   | Eclipse | SAA | FDIR | GS\n";
    std::cout << "---------|---------|---------|-------|--------|--------|------------|---------|-----|------|----\n";

    int print_interval = total_steps / 40;  // ~40 lines of output
    if (print_interval < 1) print_interval = 1;

    for (int i = 0; i < total_steps; ++i) {
        smas::StatePacket s = engine.step(action);

        // Print progress at intervals
        if (i % print_interval == 0 || s.is_done) {
            double hours = s.sim_time_s / 3600.0;
            std::cout << std::setw(8) << i << " | "
                      << std::setw(7) << hours << " | "
                      << std::setw(7) << s.altitude_km << " | "
                      << std::setw(5) << s.latitude_deg << " | "
                      << std::setw(6) << s.longitude_deg << " | "
                      << std::setw(6) << (s.battery_soc * 100.0) << " | "
                      << std::scientific << std::setprecision(3) << s.atm_density
                      << std::fixed << std::setprecision(2) << " | "
                      << (s.in_eclipse ? "  YES  " : "  no   ") << " | "
                      << (s.in_saa ? " Y " : " n ") << "  | "
                      << static_cast<int>(s.fdir_mode) << "    | "
                      << static_cast<int>(s.gs_visible) << "\n";
        }

        if (s.is_done) {
            const char* reasons[] = {"ongoing", "BATTERY_DEAD", "TELEMETRY_LOSS", "REENTRY", "SEU_FATAL"};
            std::cout << "\n  *** EPISODE TERMINATED: "
                      << reasons[s.done_reason] << " at step " << i
                      << " (t=" << s.sim_time_s / 3600.0 << "h) ***\n";
            break;
        }
    }

    // ── Summary ───────────────────────────────────────────────────
    const auto& s = engine.current_state();
    std::cout << "\n── Final State ──────────────────────────────────────\n"
              << "  Time:        " << s.sim_time_s / 3600.0 << " hours\n"
              << "  Altitude:    " << s.altitude_km << " km\n"
              << "  Battery SoC: " << s.battery_soc * 100.0 << " %\n"
              << "  Capacity:    " << s.battery_capacity_j / 1000.0 << " kJ\n"
              << "  Panel Eff:   " << s.panel_efficiency * 100.0 << " %\n"
              << "  Drag Cd:     " << s.drag_coeff << "\n"
              << "  Cycles:      " << s.charge_cycles << "\n"
              << "  FDIR Mode:   " << static_cast<int>(s.fdir_mode) << "\n";

    std::cout << "\n  StatePacket size: " << sizeof(smas::StatePacket) << " bytes\n"
              << "  ActionPacket size: " << sizeof(smas::ActionPacket) << " bytes\n";

    return 0;
}
