/*
 * S-MAS: Orbital Perturbations & RK4 Integrator — Implementation
 * Tasks 1.4, 1.5
 */
#include "orbital_mechanics.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace smas {

// ═══════════════════════════════════════════════════════════════════
//  Gravitational Acceleration: Point Mass + J2
// ═══════════════════════════════════════════════════════════════════

Vec3 gravity_j2(const Vec3& pos_m) {
    double r = pos_m.magnitude();
    if (r < 1.0) return Vec3(); // safety

    double r2 = r * r;
    double r5 = r2 * r2 * r;
    double mu = constants::EARTH_GM;
    double Re = constants::EARTH_RADIUS_M;
    double J2 = constants::EARTH_J2;
    double z2 = pos_m.z * pos_m.z;

    // Point-mass gravity: -μ/r³ * r_vec
    double factor = -mu / (r2 * r);

    // J2 perturbation terms
    double j2_coeff = 1.5 * J2 * Re * Re / r2;
    double z2_r2 = z2 / r2;

    Vec3 acc;
    acc.x = factor * pos_m.x * (1.0 + j2_coeff * (1.0 - 5.0 * z2_r2));
    acc.y = factor * pos_m.y * (1.0 + j2_coeff * (1.0 - 5.0 * z2_r2));
    acc.z = factor * pos_m.z * (1.0 + j2_coeff * (3.0 - 5.0 * z2_r2));

    return acc;
}

// ═══════════════════════════════════════════════════════════════════
//  Aerodynamic Drag
// ═══════════════════════════════════════════════════════════════════

Vec3 drag_acceleration(const Vec3& pos_m, const Vec3& vel_ms,
                       double rho, double cd,
                       double area_m2, double mass_kg) {
    // F_D = -0.5 * ρ * Cd * A * v² * v̂
    // Account for Earth rotation: velocity relative to atmosphere
    // v_rel = v - ω×r  (simplified: atmosphere co-rotates with Earth)
    Vec3 omega_cross_r = {-constants::EARTH_ROTATION * pos_m.y,
                           constants::EARTH_ROTATION * pos_m.x,
                           0.0};
    Vec3 v_rel = vel_ms - omega_cross_r;
    double v_mag = v_rel.magnitude();
    if (v_mag < 1e-6) return Vec3();

    double accel_mag = -0.5 * rho * cd * area_m2 * v_mag * v_mag / mass_kg;
    return v_rel.normalized() * accel_mag;
}

// ═══════════════════════════════════════════════════════════════════
//  Thrust Acceleration
// ═══════════════════════════════════════════════════════════════════

Vec3 thrust_acceleration(const Vec3& thrust_dir, double throttle,
                         double max_dv_per_step, double mass_kg) {
    // Convert agent action to a physical acceleration
    // thrust_dir components are in [-1, 1] (attitude)
    // throttle is in [0, 1]
    Vec3 dir = thrust_dir.normalized();
    // delta-v this step = max_dv * throttle
    // acceleration = delta_v / dt
    double accel = max_dv_per_step * throttle / constants::DT;
    return dir * accel;
}

// ═══════════════════════════════════════════════════════════════════
//  Total Acceleration
// ═══════════════════════════════════════════════════════════════════

Vec3 total_acceleration(const Vec3& pos_m, const Vec3& vel_ms,
                        const AccelParams& p) {
    Vec3 a_grav  = gravity_j2(pos_m);
    Vec3 a_drag  = drag_acceleration(pos_m, vel_ms, p.rho, p.cd, p.area_m2, p.mass_kg);
    Vec3 a_total = a_grav + a_drag + p.thrust_accel;
    return a_total;
}

// ═══════════════════════════════════════════════════════════════════
//  Runge-Kutta 4 Integrator (dt = 5.0 s)
// ═══════════════════════════════════════════════════════════════════

OrbitalState rk4_step(const OrbitalState& s, const AccelParams& p) {
    const double dt = constants::DT;
    const double h2 = dt * 0.5;
    const double h6 = dt / 6.0;

    // State: [pos, vel]   Derivative: [vel, accel]

    // k1
    Vec3 a1 = total_acceleration(s.pos, s.vel, p);
    Vec3 k1v = s.vel;
    Vec3 k1a = a1;

    // k2
    Vec3 pos2 = s.pos + k1v * h2;
    Vec3 vel2 = s.vel + k1a * h2;
    Vec3 a2 = total_acceleration(pos2, vel2, p);
    Vec3 k2v = vel2;
    Vec3 k2a = a2;

    // k3
    Vec3 pos3 = s.pos + k2v * h2;
    Vec3 vel3 = s.vel + k2a * h2;
    Vec3 a3 = total_acceleration(pos3, vel3, p);
    Vec3 k3v = vel3;
    Vec3 k3a = a3;

    // k4
    Vec3 pos4 = s.pos + k3v * dt;
    Vec3 vel4 = s.vel + k3a * dt;
    Vec3 a4 = total_acceleration(pos4, vel4, p);
    Vec3 k4v = vel4;
    Vec3 k4a = a4;

    // Combine
    OrbitalState next;
    next.pos = s.pos + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * h6;
    next.vel = s.vel + (k1a + 2.0 * k2a + 2.0 * k3a + k4a) * h6;
    next.time = s.time + dt;

    return next;
}

double altitude_km(const Vec3& pos_m) {
    return pos_m.magnitude() / 1000.0 - constants::EARTH_RADIUS_KM;
}

} // namespace smas
