/*
 * S-MAS Phase A: Lumped-Node Thermal Model — Implementation
 *
 * 3-node model: bus, battery, payload
 * Heat sources: solar flux, internal dissipation, Earth albedo
 * Heat sinks: radiative cooling (Stefan-Boltzmann)
 * Inter-node: conductive coupling
 */
#include "thermal.h"
#include "types.h"
#include <cmath>
#include <algorithm>

namespace smas {

// ── Thermal coupling constants ────────────────────────────────────
static constexpr double K_BUS_BATT    = 2.0;   // W/K conduction bus↔battery
static constexpr double K_BUS_PAYLOAD = 1.0;   // W/K conduction bus↔payload
static constexpr double BATT_MASS_KG  = 8.0;   // battery thermal mass
static constexpr double BATT_CP       = 1000.0; // J/(kg·K) Li-ion specific heat
static constexpr double PAYLOAD_MASS  = 15.0;  // kg payload thermal mass
static constexpr double PAYLOAD_CP    = 800.0; // J/(kg·K)
static constexpr double VIEW_FACTOR   = 0.3;   // Earth view factor for albedo

ThermalModel::ThermalModel() { reset(); }

void ThermalModel::reset() {
    // Start at comfortable 20°C
    state_.temp_bus     = 20.0;
    state_.temp_battery = 20.0;
    state_.temp_payload = 20.0;
    state_.heater_on    = false;
}

double ThermalModel::radiative_loss(double temp_k, double emissivity, double area) {
    double t_space = constants::SPACE_TEMP_K;
    return emissivity * constants::STEFAN_BOLTZMANN * area *
           (temp_k * temp_k * temp_k * temp_k -
            t_space * t_space * t_space * t_space);
}

void ThermalModel::update(double penumbra_factor, double solar_power_w,
                           double power_draw_w, double dt) {
    (void)solar_power_w; // Suppress unused parameter warning
    // ── Bus node ──────────────────────────────────────────────────
    double T_bus_k = to_kelvin(state_.temp_bus);

    // Solar input to bus (absorbed by structure, not solar panels)
    double Q_solar_bus = constants::SOLAR_FLUX_W_M2 * constants::SAT_ABSORPTIVITY *
                         constants::SAT_AREA_M2 * penumbra_factor;

    // Earth albedo (always present, reduced in eclipse)
    double Q_albedo = constants::SOLAR_FLUX_W_M2 * constants::EARTH_ALBEDO *
                      VIEW_FACTOR * constants::SAT_AREA_M2 * penumbra_factor;

    // Internal dissipation (electronics waste heat)
    double Q_internal = power_draw_w * 0.6; // ~60% becomes heat

    // Radiative loss
    double Q_rad_bus = radiative_loss(T_bus_k, constants::SAT_EMISSIVITY,
                                       constants::SAT_RADIATOR_AREA);

    // Conduction to battery and payload
    double T_batt_k = to_kelvin(state_.temp_battery);
    double T_pay_k  = to_kelvin(state_.temp_payload);
    double Q_cond_batt = K_BUS_BATT * (T_bus_k - T_batt_k);
    double Q_cond_pay  = K_BUS_PAYLOAD * (T_bus_k - T_pay_k);

    // Bus temperature update: dT = (Q_in - Q_out) / (m * Cp) * dt
    double dT_bus = (Q_solar_bus + Q_albedo + Q_internal -
                     Q_rad_bus - Q_cond_batt - Q_cond_pay) /
                    (constants::SAT_THERMAL_MASS * constants::SAT_SPECIFIC_HEAT) * dt;
    state_.temp_bus += dT_bus;

    // ── Battery node ──────────────────────────────────────────────
    // Heat input: conduction from bus + heater
    double Q_heater = 0.0;
    if (state_.temp_battery < constants::HEATER_ON_TEMP_C) {
        state_.heater_on = true;
        Q_heater = constants::HEATER_POWER_W;
    } else if (state_.temp_battery > constants::HEATER_ON_TEMP_C + 5.0) {
        // Hysteresis: turn off 5°C above threshold
        state_.heater_on = false;
    }

    double Q_rad_batt = radiative_loss(T_batt_k, 0.3, 0.05); // small radiator
    double dT_batt = (Q_cond_batt + Q_heater - Q_rad_batt) /
                     (BATT_MASS_KG * BATT_CP) * dt;
    state_.temp_battery += dT_batt;

    // ── Payload node ──────────────────────────────────────────────
    double Q_rad_pay = radiative_loss(T_pay_k, 0.6, 0.1);
    double dT_pay = (Q_cond_pay - Q_rad_pay) /
                    (PAYLOAD_MASS * PAYLOAD_CP) * dt;
    state_.temp_payload += dT_pay;

    // ── Clamp to physical limits ──────────────────────────────────
    state_.temp_bus     = smas::compat::clamp(state_.temp_bus,     -60.0, 80.0);
    state_.temp_battery = smas::compat::clamp(state_.temp_battery, -40.0, 70.0);
    state_.temp_payload = smas::compat::clamp(state_.temp_payload, -60.0, 80.0);
}

double ThermalModel::battery_temp_factor() const {
    double T = state_.temp_battery;
    if (T >= 0.0 && T <= 40.0) {
        return 1.0;  // optimal range
    } else if (T > 40.0) {
        // Linear degradation above 40°C, min 0.5 at 60°C
        return smas::compat::clamp(1.0 - 0.025 * (T - 40.0), 0.5, 1.0);
    } else {
        // Linear degradation below 0°C, min 0.5 at -20°C
        return smas::compat::clamp(1.0 - 0.025 * (-T), 0.5, 1.0);
    }
}

double ThermalModel::heater_power_w() const {
    return state_.heater_on ? constants::HEATER_POWER_W : 0.0;
}

} // namespace smas
