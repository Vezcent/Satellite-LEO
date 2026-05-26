#include "orbital_mechanics.h"
#include "constants.h"
#include "geometry.h"
#include <cmath>
#include <algorithm>

namespace smas {

// ═══════════════════════════════════════════════════════════════════
//  Gravitational Acceleration: Point Mass + J2 - J6 Zonal Harmonics
// ═══════════════════════════════════════════════════════════════════

Vec3 gravity_high_fidelity(const Vec3& pos_m) {
    double r = pos_m.magnitude();
    if (r < 1.0) return Vec3(); // safety

    double r_inv = 1.0 / r;
    double q = pos_m.z * r_inv;
    double q2 = q * q;

    double mu = constants::EARTH_GM;
    double Re = constants::EARTH_RADIUS_M;

    // Point-mass gravity: -μ/r³ * r_vec
    double pm_factor = -mu * r_inv * r_inv * r_inv;
    Vec3 acc = pos_m * pm_factor;

    // Legendre polynomials and their derivatives for n = 2..6
    // n = 2
    double P2 = 0.5 * (3.0 * q2 - 1.0);
    double dP2 = 3.0 * q;

    // n = 3
    double P3 = 0.5 * (5.0 * q * q2 - 3.0 * q);
    double dP3 = 0.5 * (15.0 * q2 - 3.0);

    // n = 4
    double P4 = 0.125 * (35.0 * q2 * q2 - 30.0 * q2 + 3.0);
    double dP4 = 0.5 * (35.0 * q * q2 - 15.0 * q);

    // n = 5
    double P5 = 0.125 * (63.0 * q2 * q2 * q - 70.0 * q2 * q + 15.0 * q);
    double dP5 = 0.125 * (315.0 * q2 * q2 - 210.0 * q2 + 15.0);

    // n = 6
    double P6 = 0.0625 * (231.0 * q2 * q2 * q2 - 315.0 * q2 * q2 + 105.0 * q2 - 5.0);
    double dP6 = 0.125 * (693.0 * q2 * q2 * q - 630.0 * q2 * q + 105.0 * q);

    // Harmonic coefficients J_2 to J_6
    double J[7] = {0.0, 0.0, constants::EARTH_J2, constants::EARTH_J3, constants::EARTH_J4, constants::EARTH_J5, constants::EARTH_J6};
    double P[7] = {0.0, 0.0, P2, P3, P4, P5, P6};
    double dP[7] = {0.0, 0.0, dP2, dP3, dP4, dP5, dP6};

    double sum_xy = 0.0;
    double sum_z = 0.0;

    double Re_r = Re * r_inv;
    double Re_r_n = Re_r * Re_r; // Re^2 / r^2

    for (int n = 2; n <= 6; ++n) {
        // A_n = mu * Re^n * J_n / r^(n+3) = (mu / r^3) * (Re/r)^n * J_n
        double term_coeff = J[n] * Re_r_n;
        sum_xy += term_coeff * ((n + 1) * P[n] + q * dP[n]);
        sum_z  += term_coeff * ((n + 1) * q * P[n] - (1.0 - q2) * dP[n]);
        Re_r_n *= Re_r; // (Re/r)^(n+1)
    }

    double acc_factor_xy = mu * r_inv * r_inv * r_inv;
    acc.x += pos_m.x * acc_factor_xy * sum_xy;
    acc.y += pos_m.y * acc_factor_xy * sum_xy;
    acc.z += r * acc_factor_xy * sum_z;

    return acc;
}

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
//  Third-Body Ephemeris
// ═══════════════════════════════════════════════════════════════════

static Vec3 approximate_moon_position(int year, int doy, double hour_utc) {
    // Days since J2000.0 (2000 Jan 1 12:00 UTC)
    double d = (year - 2000) * 365.25 + (doy - 1) + hour_utc / 24.0 - 0.5;

    // Mean longitude, mean anomaly, and mean distance argument (degrees)
    double L = 218.316 + 13.1763964 * d;
    double M = 134.963 + 13.0649929 * d;
    double F = 93.272 + 13.2293502 * d;

    L = std::fmod(L, 360.0); if (L < 0.0) L += 360.0;
    M = std::fmod(M, 360.0); if (M < 0.0) M += 360.0;
    F = std::fmod(F, 360.0); if (F < 0.0) F += 360.0;

    double M_rad = M * constants::DEG2RAD;
    double F_rad = F * constants::DEG2RAD;

    double lambda = L + 6.289 * std::sin(M_rad);
    double beta = 5.128 * std::sin(F_rad);

    double lambda_rad = lambda * constants::DEG2RAD;
    double beta_rad = beta * constants::DEG2RAD;

    // Distance in meters
    double r_M = 3.844e8 * (1.0 - 0.0549 * std::cos(M_rad));
    double eps_rad = 23.439 * constants::DEG2RAD;

    Vec3 moon;
    moon.x = r_M * (std::cos(beta_rad) * std::cos(lambda_rad));
    moon.y = r_M * (std::cos(eps_rad) * std::cos(beta_rad) * std::sin(lambda_rad) - std::sin(eps_rad) * std::sin(beta_rad));
    moon.z = r_M * (std::sin(eps_rad) * std::cos(beta_rad) * std::sin(lambda_rad) + std::cos(eps_rad) * std::sin(beta_rad));

    return moon;
}

// ═══════════════════════════════════════════════════════════════════
//  Total Acceleration
// ═══════════════════════════════════════════════════════════════════

Vec3 total_acceleration(const Vec3& pos_m, const Vec3& vel_ms,
                        const AccelParams& p) {
    // 1. Zonal gravity (J2-J6 high-fidelity)
    Vec3 a_grav = gravity_high_fidelity(pos_m);

    // 2. Drag
    Vec3 a_drag = drag_acceleration(pos_m, vel_ms, p.rho, p.cd, p.area_m2, p.mass_kg);

    // 3. Third-body perturbations (Sun & Moon)
    // Sun position
    Vec3 sun_dir = approximate_sun_direction(p.year, p.doy, p.hour_utc);
    Vec3 pos_sun = sun_dir * 1.496e11; // 1 AU in meters
    Vec3 d_sun_sat = pos_sun - pos_m;
    double d_sun_sat_mag = d_sun_sat.magnitude();
    double pos_sun_mag = pos_sun.magnitude();

    Vec3 a_sun_3b(0.0, 0.0, 0.0);
    if (d_sun_sat_mag > 1e-6 && pos_sun_mag > 1e-6) {
        double gm_sun = 1.32712440018e20; // m³/s²
        a_sun_3b = d_sun_sat * (gm_sun / (d_sun_sat_mag * d_sun_sat_mag * d_sun_sat_mag)) - 
                   pos_sun * (gm_sun / (pos_sun_mag * pos_sun_mag * pos_sun_mag));
    }

    // Moon position
    Vec3 pos_moon = approximate_moon_position(p.year, p.doy, p.hour_utc);
    Vec3 d_moon_sat = pos_moon - pos_m;
    double d_moon_sat_mag = d_moon_sat.magnitude();
    double pos_moon_mag = pos_moon.magnitude();

    Vec3 a_moon_3b(0.0, 0.0, 0.0);
    if (d_moon_sat_mag > 1e-6 && pos_moon_mag > 1e-6) {
        double gm_moon = 4.9027779e12; // m³/s²
        a_moon_3b = d_moon_sat * (gm_moon / (d_moon_sat_mag * d_moon_sat_mag * d_moon_sat_mag)) - 
                    pos_moon * (gm_moon / (pos_moon_mag * pos_moon_mag * pos_moon_mag));
    }

    Vec3 a_total = a_grav + a_drag + p.thrust_accel + a_sun_3b + a_moon_3b;
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
