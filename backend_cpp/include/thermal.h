/*
 * S-MAS Phase A: Lumped-Node Thermal Model
 * Three thermal nodes: bus, battery, payload.
 * Computes temperature evolution based on solar flux, internal dissipation,
 * Earth albedo, and radiative cooling.
 */
#pragma once
#include "constants.h"

namespace smas {

struct ThermalState {
    double temp_bus;       // °C
    double temp_battery;   // °C
    double temp_payload;   // °C
    bool   heater_on;      // battery heater active
};

class ThermalModel {
public:
    ThermalModel();

    void reset();

    // ── Main update ───────────────────────────────────────────────
    //   penumbra_factor: [0, 1] solar illumination fraction
    //   solar_power_w : current solar panel output (W)
    //   power_draw_w  : current bus power draw (W)
    //   dt            : time step (s)
    void update(double penumbra_factor, double solar_power_w,
                double power_draw_w, double dt);

    // ── Getters ───────────────────────────────────────────────────
    double temp_bus()     const { return state_.temp_bus; }
    double temp_battery() const { return state_.temp_battery; }
    double temp_payload() const { return state_.temp_payload; }
    bool   heater_on()    const { return state_.heater_on; }

    // ── Battery temperature factor ────────────────────────────────
    //   Returns [0.5, 1.0] capacity multiplier based on battery temp.
    //   1.0 when 0°C ≤ T ≤ 40°C, degrades outside this range.
    double battery_temp_factor() const;

    // ── Heater power draw ─────────────────────────────────────────
    //   Returns watts consumed by heater (0 if off)
    double heater_power_w() const;

private:
    ThermalState state_;

    // Convert °C to Kelvin
    static double to_kelvin(double celsius) { return celsius + 273.15; }
    static double to_celsius(double kelvin) { return kelvin - 273.15; }

    // Radiative heat loss (Stefan-Boltzmann)
    static double radiative_loss(double temp_k, double emissivity, double area);
};

} // namespace smas
