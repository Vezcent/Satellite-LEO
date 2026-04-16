/*
 * S-MAS: Orbital Perturbations & RK4 Integrator
 * Tasks 1.4, 1.5 — Aerodynamic drag, J2 perturbation, and the
 *                   fixed-step RK4 ODE solver at dt = 5.0 s.
 */
#pragma once
#include "types.h"

namespace smas {

// ── Acceleration Models ───────────────────────────────────────────

// Gravitational acceleration (point-mass + J2 oblateness).
// pos_m: satellite ECI position (metres).
Vec3 gravity_j2(const Vec3& pos_m);

// Aerodynamic drag acceleration.
//   pos_m     : ECI position (m)
//   vel_ms    : ECI velocity (m/s)
//   rho       : atmospheric density (kg/m³)
//   cd        : drag coefficient (nominally 2.2, drifts)
//   area_m2   : cross-sectional area (m²)
//   mass_kg   : satellite mass (kg)
Vec3 drag_acceleration(const Vec3& pos_m, const Vec3& vel_ms,
                       double rho, double cd,
                       double area_m2, double mass_kg);

// Thrust acceleration from agent action.
//   thrust_dir : normalised direction [-1,1] per axis (attitude)
//   throttle   : [0,1]
//   max_dv     : maximum delta-v per step (m/s)
//   mass_kg    : satellite mass (kg)
Vec3 thrust_acceleration(const Vec3& thrust_dir, double throttle,
                         double max_dv_per_step, double mass_kg);

// Total acceleration for the RK4 integrator. Wraps gravity, drag, thrust.
struct AccelParams {
    double rho;
    double cd;
    double area_m2;
    double mass_kg;
    Vec3   thrust_accel;  // pre-computed thrust accel (m/s²)
};

Vec3 total_acceleration(const Vec3& pos_m, const Vec3& vel_ms,
                        const AccelParams& params);

// ── Runge-Kutta 4 Integrator ──────────────────────────────────────

struct OrbitalState {
    Vec3   pos;   // metres, ECI
    Vec3   vel;   // m/s, ECI
    double time;  // seconds since epoch
};

// Single RK4 step with dt = constants::DT (5.0 s).
OrbitalState rk4_step(const OrbitalState& state, const AccelParams& params);

// Compute altitude (km) from ECI position.
double altitude_km(const Vec3& pos_m);

// Compute orbital velocity magnitude (m/s).
inline double orbital_speed(const Vec3& vel_ms) { return vel_ms.magnitude(); }

} // namespace smas
