/*
 * S-MAS: Power Subsystem & Realistic Degradation — Implementation
 * Task 1.6
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
    solar_w_      = 0.0;
    draw_w_       = 0.0;
    cycles_       = 0;
    was_charging_ = true;   // assume starts charging
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

    // ── Net power & SoC update ────────────────────────────────────
    double net_power_w = solar_w_ - draw_w_;
    double energy_delta_j = net_power_w * dt;

    double battery_energy_j = soc_ * capacity_j_ + energy_delta_j;
    battery_energy_j = smas::compat::clamp(battery_energy_j, 0.0, capacity_j_);
    soc_ = battery_energy_j / capacity_j_;

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
    // loss_fraction = base_rate * (1 + thermal_factor * cycles)
    double loss = constants::BATT_CYCLE_DEGRAD *
                  (1.0 + constants::BATT_THERMAL_FACT * static_cast<double>(cycles_));
    capacity_j_ *= (1.0 - loss);
    capacity_j_ = std::max(capacity_j_, 0.0);
}

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

    return DoneReason::ONGOING;
}

} // namespace smas
