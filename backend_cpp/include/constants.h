/*
 * S-MAS: Physical and simulation constants.
 * All values are mission-specific to the PROBA-1 platform at 600km LEO.
 */
#pragma once

namespace smas {
namespace constants {

// ── Global Simulation ──────────────────────────────────────────────
constexpr double DT = 5.0;  // Integration time step (seconds). Immutable.

// ── Earth Parameters ───────────────────────────────────────────────
constexpr double EARTH_RADIUS_M   = 6371000.0;
constexpr double EARTH_RADIUS_KM  = 6371.0;
constexpr double EARTH_GM         = 3.986004418e14;  // m³/s²
constexpr double EARTH_J2         = 1.08263e-3;
constexpr double EARTH_J3         = -2.5324859e-6;
constexpr double EARTH_J4         = -1.61962159e-6;
constexpr double EARTH_J5         = -2.2771610e-7;
constexpr double EARTH_J6         = 5.40665e-7;
constexpr double EARTH_ROTATION   = 7.2921159e-5;    // rad/s

// ── PROBA-1 Platform ──────────────────────────────────────────────
constexpr double SAT_MASS_KG         = 94.0;
constexpr double SAT_AREA_M2         = 0.36;        // aerodynamic cross-section
constexpr double SAT_CD_NOMINAL      = 2.2;         // nominal drag coefficient
constexpr double SAT_SOLAR_POWER_W   = 90.0;        // peak GaAs array output
constexpr double SAT_BATTERY_CAP_J   = 360000.0;    // 100 Wh = 360 kJ
constexpr double SAT_PAYLOAD_POWER_W = 25.0;        // CHRIS instrument draw
constexpr double SAT_BUS_POWER_W     = 30.0;        // bus baseline
constexpr double SAT_SLEEP_POWER_W   = 5.0;         // deep-sleep minimum

// ── Propulsion ─────────────────────────────────────────────────────
constexpr double SAT_FUEL_KG         = 5.6;         // PROBA-1 hydrazine (kg)
constexpr double THRUSTER_ISP_S      = 220.0;       // Monoprop specific impulse (s)
constexpr double G0                  = 9.80665;     // Standard gravity (m/s²)
constexpr double FUEL_CRITICAL_KG    = 0.1;         // Emergency threshold (kg)

// ── Orbital ────────────────────────────────────────────────────────
constexpr double NOMINAL_ALT_KM      = 600.0;
constexpr double REENTRY_ALT_KM      = 200.0;

// ── Math ───────────────────────────────────────────────────────────
constexpr double PI        = 3.14159265358979323846;
constexpr double TWO_PI    = 2.0 * PI;
constexpr double DEG2RAD   = PI / 180.0;
constexpr double RAD2DEG   = 180.0 / PI;
constexpr double BOLTZMANN = 1.380649e-23;   // J/K
constexpr double AMU       = 1.66053906660e-27; // kg (atomic mass unit)

// ── Communication ──────────────────────────────────────────────────
constexpr double TELEMETRY_LOSS_S    = 72.0 * 3600.0;  // 72 h
constexpr double ELEVATION_MASK_DEG  = 5.0;

// ── Stochastic / Noise ────────────────────────────────────────────
constexpr double ACTUATOR_ERROR   = 0.05;       // ±5 %
constexpr int    ACT_DELAY_MIN    = 1;           // steps
constexpr int    ACT_DELAY_MAX    = 3;

// ── Battery Degradation (Arrhenius-inspired) ──────────────────────
constexpr double BATT_CYCLE_DEGRAD  = 0.00002;  // capacity loss / cycle (calibrated for 20yr LEO life)
constexpr double BATT_THERMAL_FACT  = 0.00005;   // thermal acceleration (reduced from 0.001 — old value killed battery in 1.7yr)

// ── Thermal Model ─────────────────────────────────────────────────
constexpr double SOLAR_FLUX_W_M2    = 1361.0;   // Solar constant at 1 AU
constexpr double SAT_ABSORPTIVITY   = 0.3;      // Solar absorptivity
constexpr double SAT_EMISSIVITY     = 0.8;      // IR emissivity
constexpr double SAT_RADIATOR_AREA  = 0.25;     // m² radiator area
constexpr double SAT_THERMAL_MASS   = 50.0;     // kg effective thermal mass
constexpr double SAT_SPECIFIC_HEAT  = 900.0;    // J/(kg·K) aluminum
constexpr double HEATER_POWER_W     = 7.0;      // Battery heater draw
constexpr double HEATER_ON_TEMP_C   = -5.0;     // Turn heater on below this
constexpr double STEFAN_BOLTZMANN   = 5.670374419e-8; // W/(m²·K⁴)
constexpr double SPACE_TEMP_K       = 2.7;      // Cosmic background
constexpr double EARTH_ALBEDO       = 0.3;      // Average Earth albedo

// ── SEU Parameters ────────────────────────────────────────────────
constexpr double SEU_BASE_PROB     = 0.001;
constexpr double SEU_SAA_MULT      = 100.0;

// ── Model Drift (Epistemic Uncertainty) ───────────────────────────
constexpr double CD_DRIFT_SIGMA    = 0.001;     // per step σ for Cd random walk
constexpr double PANEL_DRIFT_SIGMA = 0.00003;    // per step σ for panel efficiency (reduced for 20yr realism)

// ── SAA Boundary (coarse bounding box for quick checks) ───────────
constexpr double SAA_LAT_MIN = -50.0;
constexpr double SAA_LAT_MAX =   0.0;
constexpr double SAA_LON_MIN = -90.0;
constexpr double SAA_LON_MAX =  40.0;
constexpr double SAA_FLUX_THRESHOLD = 1.0; // flux > this ⇒ "in SAA"

} // namespace constants
} // namespace smas
