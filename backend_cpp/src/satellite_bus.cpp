/*
 * S-MAS: Power Subsystem, Propellant Tracking & Realistic Degradation
 * Task 1.6 + Phase A
 */
#include "satellite_bus.h"
#include "types.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace smas {

SatelliteBus::SatelliteBus() { reset(); }

void SatelliteBus::reset() {
    soc_          = 1.0;    // start fully charged
    capacity_j_   = constants::SAT_BATTERY_CAP_J;
    effective_capacity_j_ = capacity_j_;
    solar_w_      = 0.0;
    draw_w_       = 0.0;
    cycles_       = 0;
    was_charging_ = true;   // assume starts charging
    fuel_kg_      = constants::SAT_FUEL_KG;
    thermal_factor_ = 1.0;
}

void SatelliteBus::update(bool in_eclipse, double panel_eff,
                          bool deep_sleep, bool payload_on, double dt) {
    // ── Solar power generation ────────────────────────────────────
    if (in_eclipse) {
        solar_w_ = 0.0;
    } else {
        solar_w_ = constants::SAT_SOLAR_POWER_W * panel_eff;
    }

    // ── Power consumption ─────────────────────────────────────────
    if (deep_sleep) {
        draw_w_ = constants::SAT_SLEEP_POWER_W;
        // Force payload OFF in deep sleep
    } else {
        draw_w_ = constants::SAT_BUS_POWER_W;
        if (payload_on) {
            draw_w_ += constants::SAT_PAYLOAD_POWER_W;
        }
    }

    // ── Net power & SoC update (using effective capacity) ─────────
    double eff_cap = effective_capacity_j_;
    double net_power_w = solar_w_ - draw_w_;
    double energy_delta_j = net_power_w * dt;

    double battery_energy_j = soc_ * eff_cap + energy_delta_j;
    battery_energy_j = smas::compat::clamp(battery_energy_j, 0.0, eff_cap);
    soc_ = (eff_cap > 0.0) ? battery_energy_j / eff_cap : 0.0;

    // ── Charge/discharge cycle detection ──────────────────────────
    bool is_charging = (net_power_w > 0.0);
    if (was_charging_ && !is_charging) {
        // Transition from charging to discharging → one cycle completed
        cycles_++;
        apply_cycle_degradation();
    }
    was_charging_ = is_charging;
}

void SatelliteBus::apply_cycle_degradation() {
    // Arrhenius-based degradation: each cycle reduces max capacity slightly.
    double loss = constants::BATT_CYCLE_DEGRAD *
                  (1.0 + constants::BATT_THERMAL_FACT * static_cast<double>(cycles_));
    capacity_j_ *= (1.0 - loss);
    capacity_j_ = std::max(capacity_j_, 0.0);
    // Update effective capacity with current thermal factor
    effective_capacity_j_ = capacity_j_ * thermal_factor_;
}

void SatelliteBus::set_degradation(double capacity_j) {
    capacity_j_ = smas::compat::clamp(capacity_j, 10000.0, constants::SAT_BATTERY_CAP_J);
    effective_capacity_j_ = capacity_j_ * thermal_factor_;
    // Adjust SoC to preserve current energy if possible
    double current_energy = soc_ * effective_capacity_j_;
    soc_ = smas::compat::clamp(current_energy / effective_capacity_j_, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════
//  Fuel Tracking (Phase A)
// ═══════════════════════════════════════════════════════════════════

double SatelliteBus::fuel_fraction() const {
    return smas::compat::clamp(fuel_kg_ / constants::SAT_FUEL_KG, 0.0, 1.0);
}

double SatelliteBus::consume_fuel(double dv, double mass_kg) {
    if (fuel_kg_ <= 0.0 || dv <= 0.0) return 0.0;

    // Tsiolkovsky rocket equation: Δm = m × (1 - exp(-Δv / (Isp × g0)))
    double exhaust_vel = constants::THRUSTER_ISP_S * constants::G0;
    double mass_fraction = 1.0 - std::exp(-dv / exhaust_vel);
    double fuel_needed = mass_kg * mass_fraction;

    if (fuel_needed > fuel_kg_) {
        // Not enough fuel — compute achievable Δv
        double achievable_mass_frac = fuel_kg_ / mass_kg;
        double achievable_dv = -exhaust_vel * std::log(1.0 - achievable_mass_frac);
        fuel_kg_ = 0.0;
        return achievable_dv;
    }

    fuel_kg_ -= fuel_needed;
    return dv;  // full Δv achieved
}

// ═══════════════════════════════════════════════════════════════════
//  Thermal Integration (Phase A)
// ═══════════════════════════════════════════════════════════════════

void SatelliteBus::apply_thermal_factor(double factor) {
    thermal_factor_ = smas::compat::clamp(factor, 0.5, 1.0);
    effective_capacity_j_ = capacity_j_ * thermal_factor_;
}

void SatelliteBus::apply_heater_draw(double heater_w, double dt) {
    if (heater_w <= 0.0) return;
    double energy_j = heater_w * dt;
    double battery_energy = soc_ * effective_capacity_j_;
    battery_energy -= energy_j;
    battery_energy = std::max(battery_energy, 0.0);
    soc_ = (effective_capacity_j_ > 0.0) ? battery_energy / effective_capacity_j_ : 0.0;
    // Add to draw for telemetry
    draw_w_ += heater_w;
}

// ═══════════════════════════════════════════════════════════════════
//  Failure Contract
// ═══════════════════════════════════════════════════════════════════

DoneReason SatelliteBus::check_failure(double time_since_contact,
                                        double alt_km,
                                        bool   seu_fatal) const {
    // 1. Battery depletion
    if (soc_ <= 0.0)
        return DoneReason::BATTERY_DEAD;

    // 2. Prolonged telemetry loss (> 72 h)
    if (time_since_contact >= constants::TELEMETRY_LOSS_S)
        return DoneReason::TELEMETRY_LOSS;

    // 3. Re-entry
    if (alt_km < constants::REENTRY_ALT_KM)
        return DoneReason::REENTRY;

    // 4. Fatal SEU
    if (seu_fatal)
        return DoneReason::SEU_FATAL;

    // 5. Fuel depleted at dangerously low altitude
    if (fuel_kg_ <= 0.0 && alt_km < 400.0)
        return DoneReason::FUEL_DEPLETED_LOW;

    return DoneReason::ONGOING;
}

} // namespace smas
