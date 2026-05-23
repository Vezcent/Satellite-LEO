/*
 * S-MAS: Power Subsystem, Propellant Tracking & Realistic Degradation
 * Task 1.6 + Phase A (fuel tracking, thermal integration)
 */
#pragma once
#include "contracts.h"

namespace smas {

class SatelliteBus {
public:
    SatelliteBus();

    void reset();

    // ── Power update for one time step (dt = 5 s) ─────────────────
    //   in_eclipse     : true if in Earth's shadow
    //   panel_eff      : current solar panel efficiency [0,1]
    //   deep_sleep     : Resource Agent flag
    //   payload_on     : Mission Agent flag
    //   dt             : time step (seconds)
    void update(bool in_eclipse, double panel_eff,
                bool deep_sleep, bool payload_on, double cos_sun_angle, double dt);

    // ── Power Getters ─────────────────────────────────────────────
    double soc()              const { return soc_; }           // [0,1]
    double capacity_j()       const { return capacity_j_; }
    double solar_power_w()    const { return solar_w_; }
    double power_draw_w()     const { return draw_w_; }
    uint32_t charge_cycles()  const { return cycles_; }

    // ── Fuel Getters (Phase A) ────────────────────────────────────
    double fuel_kg()          const { return fuel_kg_; }
    double fuel_fraction()    const;                           // [0,1]
    bool   is_fuel_depleted() const { return fuel_kg_ <= 0.0; }

    // ── Fuel Consumption ──────────────────────────────────────────
    //   dv       : delta-v applied this step (m/s)
    //   mass_kg  : current total spacecraft mass (kg)
    //   Returns actual delta-v achievable (may be less if fuel runs out)
    double consume_fuel(double dv, double mass_kg);

    // ── Battery Degradation ───────────────────────────────────────
    void apply_cycle_degradation();
    void set_degradation(double capacity_j);

    // ── Thermal Integration (Phase A) ─────────────────────────────
    //   Apply temperature-dependent capacity factor
    //   factor in [0.5, 1.0]
    void apply_thermal_factor(double factor);

    //   Subtract heater power from battery energy
    void apply_heater_draw(double heater_w, double dt);

    // ── Failure Contract (Done) ───────────────────────────────────
    DoneReason check_failure(double time_since_contact,
                             double altitude_km,
                             bool   seu_fatal) const;

private:
    double   soc_;           // [0,1]
    double   capacity_j_;    // current max capacity (Joules)
    double   effective_capacity_j_; // after thermal factor
    double   solar_w_;       // instantaneous solar generation
    double   draw_w_;        // instantaneous total draw
    uint32_t cycles_;        // accumulated charge/discharge count
    bool     was_charging_;  // for cycle detection edge
    double   fuel_kg_;       // remaining propellant (kg)
    double   thermal_factor_; // battery temp factor [0.5, 1.0]
};

} // namespace smas
