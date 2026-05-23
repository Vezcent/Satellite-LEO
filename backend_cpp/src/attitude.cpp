/*
 * S-MAS: Attitude Determination and Control System (ADCS) — Implementation
 * Implements quaternion integration, reaction wheel dynamics, environmental torques,
 * and closed-loop Nadir tracking PD pointing control.
 */
#include "attitude.h"
#include <cmath>
#include <algorithm>

namespace smas {

// Constants
constexpr double MU = 3.986004418e14; // Earth GM (m^3/s^2)

AttitudeModel::AttitudeModel() {
    // Moments of inertia diagonal: typical small satellite (PROBA-1-like)
    inertia_[0] = 10.0; // Ixx
    inertia_[1] = 10.0; // Iyy
    inertia_[2] = 12.0; // Izz

    max_wheel_h_      = 0.2;  // Nms
    max_wheel_torque_ = 0.01; // Nm
    reset();
}

void AttitudeModel::reset() {
    // Start with nominal Z-axis aligned to ECI +Z (nadir-like startup alignment)
    state_.q0 = 1.0;
    state_.q1 = 0.0;
    state_.q2 = 0.0;
    state_.q3 = 0.0;

    state_.wx = 0.0;
    state_.wy = 0.0;
    state_.wz = 0.0;

    state_.wheel_h[0] = 0.0;
    state_.wheel_h[1] = 0.0;
    state_.wheel_h[2] = 0.0;

    state_.sun_angle   = 0.0;
    state_.nadir_error = 0.0;
}

void AttitudeModel::step(double dt, const Vec3& pos_eci, const Vec3& sun_dir_eci, const double* wheel_torque_cmd) {
    // Sub-stepping to handle angular rate dynamics (5 sub-steps of 1 second for dt=5.0s)
    const int N_SUB = 5;
    const double sub_dt = dt / N_SUB;

    for (int i = 0; i < N_SUB; ++i) {
        double control_torque[3] = {0.0, 0.0, 0.0};
        
        if (wheel_torque_cmd) {
            // Apply commanded torques directly from the agent
            control_torque[0] = wheel_torque_cmd[0];
            control_torque[1] = wheel_torque_cmd[1];
            control_torque[2] = wheel_torque_cmd[2];
        } else {
            // Closed-loop Nadir pointing PD control overlay (default)
            pointing_control(pos_eci, control_torque);
        }

        sub_step(sub_dt, pos_eci, sun_dir_eci, control_torque);
    }

    // Update diagnostic angles once per main step
    double r_mag = pos_eci.magnitude();
    if (r_mag > 0.0) {
        // Nadir vector pointing from satellite to center of Earth
        Vec3 nadir_dir_eci(-pos_eci.x / r_mag, -pos_eci.y / r_mag, -pos_eci.z / r_mag);
        
        // Convert to Body frame
        double q_conj[4] = { state_.q0, -state_.q1, -state_.q2, -state_.q3 };
        
        // Convert ECI Nadir to Body
        double nadir_p[4] = { 0.0, nadir_dir_eci.x, nadir_dir_eci.y, nadir_dir_eci.z };
        double temp[4], nadir_body[4];
        q_mult(state_.q0 ? &state_.q0 : nullptr, nadir_p, temp);
        q_mult(temp, q_conj, nadir_body);

        // Nadir error is angle between +Z body axis [0, 0, 1] and Nadir body vector
        double cos_nadir = nadir_body[3]; // Z component
        cos_nadir = std::max(-1.0, std::min(1.0, cos_nadir));
        state_.nadir_error = std::acos(cos_nadir);
    } else {
        state_.nadir_error = 0.0;
    }

    double s_mag = sun_dir_eci.magnitude();
    if (s_mag > 0.0) {
        Vec3 sun_dir_unit(sun_dir_eci.x / s_mag, sun_dir_eci.y / s_mag, sun_dir_eci.z / s_mag);
        double q_conj[4] = { state_.q0, -state_.q1, -state_.q2, -state_.q3 };
        
        double sun_p[4] = { 0.0, sun_dir_unit.x, sun_dir_unit.y, sun_dir_unit.z };
        double temp[4], sun_body[4];
        q_mult(state_.q0 ? &state_.q0 : nullptr, sun_p, temp);
        q_mult(temp, q_conj, sun_body);

        // Sun angle is angle between solar panel normal (-Z body, Zenith-pointing face) and Sun body vector
        // We model a realistic multi-panel configuration: Zenith face (-Z) plus side panel contributions (+/-X, +/-Y)
        double cos_sun = std::max(0.0, -sun_body[3]) + 0.4 * (std::abs(sun_body[1]) + std::abs(sun_body[2]));
        cos_sun = std::max(-1.0, std::min(1.0, cos_sun));
        state_.sun_angle = std::acos(cos_sun);
    } else {
        state_.sun_angle = 0.0;
    }
}

double AttitudeModel::cos_sun_angle() const {
    // Only generate power if the sun is in the positive hemisphere of the panel normal Z
    return std::max(0.0, std::cos(state_.sun_angle));
}

bool AttitudeModel::is_nadir_pointing(double tol_rad) const {
    return state_.nadir_error <= tol_rad;
}

void AttitudeModel::sub_step(double sub_dt, const Vec3& pos_eci, const Vec3& sun_dir_eci, const double* control_torque) {
    // ── 1. Calculate environmental torques ────────────────────────
    Vec3 t_gg = compute_gravity_gradient_torque(pos_eci);
    Vec3 t_srp = compute_srp_torque(sun_dir_eci);
    
    double ext_torque[3] = { t_gg.x + t_srp.x, t_gg.y + t_srp.y, t_gg.z + t_srp.z };

    // ── 2. Update reaction wheels and apply torques ──────────────
    double act_torque[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i) {
        // Clamp command to hardware torque limit
        double cmd = std::max(-max_wheel_torque_, std::min(max_wheel_torque_, control_torque[i]));
        
        // If wheel is saturated, it cannot exert additional torque
        if (state_.wheel_h[i] >= max_wheel_h_ && cmd > 0.0) cmd = 0.0;
        if (state_.wheel_h[i] <= -max_wheel_h_ && cmd < 0.0) cmd = 0.0;
        
        act_torque[i] = cmd;
    }

    update_wheels(sub_dt, act_torque);

    // ── 3. Rotational Euler dynamics propagation ──────────────────
    // I * w_dot = T_ext - T_wheels - w x (I * w + h_wheels)
    double h_tot[3] = {
        inertia_[0] * state_.wx + state_.wheel_h[0],
        inertia_[1] * state_.wy + state_.wheel_h[1],
        inertia_[2] * state_.wz + state_.wheel_h[2]
    };

    // w x h_tot
    double cross[3] = {
        state_.wy * h_tot[2] - state_.wz * h_tot[1],
        state_.wz * h_tot[0] - state_.wx * h_tot[2],
        state_.wx * h_tot[1] - state_.wy * h_tot[0]
    };

    // Calculate angular accelerations
    double wx_dot = (ext_torque[0] - act_torque[0] - cross[0]) / inertia_[0];
    double wy_dot = (ext_torque[1] - act_torque[1] - cross[1]) / inertia_[1];
    double wz_dot = (ext_torque[2] - act_torque[2] - cross[2]) / inertia_[2];

    // Euler integration
    state_.wx += wx_dot * sub_dt;
    state_.wy += wy_dot * sub_dt;
    state_.wz += wz_dot * sub_dt;

    // ── 4. Integrate Quaternion ───────────────────────────────────
    integrate_quaternion(sub_dt);
}

void AttitudeModel::integrate_quaternion(double sub_dt) {
    // Quaternion kinematics: q_dot = 0.5 * q * [0, w]
    double dq0 = 0.5 * (-state_.q1 * state_.wx - state_.q2 * state_.wy - state_.q3 * state_.wz);
    double dq1 = 0.5 * ( state_.q0 * state_.wx - state_.q3 * state_.wy + state_.q2 * state_.wz);
    double dq2 = 0.5 * ( state_.q3 * state_.wx + state_.q0 * state_.wy - state_.q1 * state_.wz);
    double dq3 = 0.5 * (-state_.q2 * state_.wx + state_.q1 * state_.wy + state_.q0 * state_.wz);

    // Integrate
    state_.q0 += dq0 * sub_dt;
    state_.q1 += dq1 * sub_dt;
    state_.q2 += dq2 * sub_dt;
    state_.q3 += dq3 * sub_dt;

    // Renormalise quaternion to maintain rotational accuracy
    double norm = std::sqrt(state_.q0 * state_.q0 + state_.q1 * state_.q1 +
                            state_.q2 * state_.q2 + state_.q3 * state_.q3);
    if (norm > 0.0) {
        state_.q0 /= norm;
        state_.q1 /= norm;
        state_.q2 /= norm;
        state_.q3 /= norm;
    }
}

void AttitudeModel::update_wheels(double sub_dt, const double* torque_cmd) {
    for (int i = 0; i < 3; ++i) {
        // Integrate wheel angular momentum: h_dot = T_cmd
        state_.wheel_h[i] += torque_cmd[i] * sub_dt;
        
        // Clamp to physical saturation limits
        state_.wheel_h[i] = std::max(-max_wheel_h_, std::min(max_wheel_h_, state_.wheel_h[i]));

        // Continuous Active Magnetic Desaturation: 
        // Slowly dump accumulated momentum towards zero (magnetic coils active)
        state_.wheel_h[i] *= (1.0 - 0.00015 * sub_dt);
    }
}

Vec3 AttitudeModel::compute_gravity_gradient_torque(const Vec3& pos_eci) const {
    double r_mag = pos_eci.magnitude();
    if (r_mag < 1000.0) return Vec3(0, 0, 0);

    // Pos unit vector in ECI
    Vec3 r_unit(pos_eci.x / r_mag, pos_eci.y / r_mag, pos_eci.z / r_mag);

    // Transform ECI position unit vector to spacecraft Body frame
    double q_conj[4] = { state_.q0, -state_.q1, -state_.q2, -state_.q3 };
    double r_p[4] = { 0.0, r_unit.x, r_unit.y, r_unit.z };
    double temp[4], r_body[4];
    q_mult(state_.q0 ? &state_.q0 : nullptr, r_p, temp);
    q_mult(temp, q_conj, r_body);

    Vec3 rb(r_body[1], r_body[2], r_body[3]); // unit vector in Body

    // T_gg = (3 * MU / r^3) * (r_body x (I * r_body))
    Vec3 I_r(inertia_[0] * rb.x, inertia_[1] * rb.y, inertia_[2] * rb.z);
    Vec3 cross = rb.cross(I_r);

    double factor = (3.0 * MU) / std::pow(r_mag, 3.0);
    return Vec3(factor * cross.x, factor * cross.y, factor * cross.z);
}

Vec3 AttitudeModel::compute_srp_torque(const Vec3& sun_dir_eci) const {
    // Simplified solar radiation pressure torque. 
    // Small surface area offset (~0.05m offset between CP and CG)
    double s_mag = sun_dir_eci.magnitude();
    if (s_mag <= 0.0) return Vec3(0, 0, 0);

    Vec3 s_unit(sun_dir_eci.x / s_mag, sun_dir_eci.y / s_mag, sun_dir_eci.z / s_mag);
    double q_conj[4] = { state_.q0, -state_.q1, -state_.q2, -state_.q3 };
    double s_p[4] = { 0.0, s_unit.x, s_unit.y, s_unit.z };
    double temp[4], s_body[4];
    q_mult(state_.q0 ? &state_.q0 : nullptr, s_p, temp);
    q_mult(temp, q_conj, s_body);

    Vec3 sb(s_body[1], s_body[2], s_body[3]);

    // Force magnitude: Area (0.25m^2) * Solar pressure (4.5e-6 N/m^2) ~ 1.1e-6 N
    double force_mag = 0.25 * 4.5e-6; 
    
    // Offset vector from CG: +0.05m along X
    Vec3 offset(0.05, 0.0, 0.0);
    Vec3 force(-sb.x * force_mag, -sb.y * force_mag, -sb.z * force_mag);
    
    return offset.cross(force);
}

void AttitudeModel::pointing_control(const Vec3& pos_eci, double* out_wheel_torque) const {
    double r_mag = pos_eci.magnitude();
    if (r_mag <= 0.0) {
        out_wheel_torque[0] = out_wheel_torque[1] = out_wheel_torque[2] = 0.0;
        return;
    }

    // Nadir unit vector in ECI
    Vec3 nadir_dir_eci(-pos_eci.x / r_mag, -pos_eci.y / r_mag, -pos_eci.z / r_mag);

    // Transform ECI Nadir to Body frame
    double q_conj[4] = { state_.q0, -state_.q1, -state_.q2, -state_.q3 };
    double nadir_p[4] = { 0.0, nadir_dir_eci.x, nadir_dir_eci.y, nadir_dir_eci.z };
    double temp[4], nadir_body[4];
    q_mult(state_.q0 ? &state_.q0 : nullptr, nadir_p, temp);
    q_mult(temp, q_conj, nadir_body);

    Vec3 nb(nadir_body[1], nadir_body[2], nadir_body[3]); // Nadir vector in Body frame

    // Pointing error vector: cross product of +Z body normal and Nadir body vector
    // e = [0, 0, 1] x [nb_x, nb_y, nb_z] = [-nb_y, nb_x, 0]
    double ex = -nb.y;
    double ey = nb.x;
    double ez = 0.0; // No yaw error defined with respect to Nadir only

    // Damping/Control gains for PD controller
    constexpr double KP = 0.20; // Proportional Gain
    constexpr double KD = 1.80; // Derivative Damping Gain (wx, wy, wz)

    // wheel_torque = KP * error + KD * omega
    out_wheel_torque[0] = KP * ex + KD * state_.wx;
    out_wheel_torque[1] = KP * ey + KD * state_.wy;
    out_wheel_torque[2] = KP * ez + KD * state_.wz; // damp Z rotational spin
}

void AttitudeModel::q_mult(const double* p, const double* q, double* r) const {
    if (!p || !q || !r) return;
    r[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];
    r[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2];
    r[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1];
    r[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];
}

} // namespace smas
