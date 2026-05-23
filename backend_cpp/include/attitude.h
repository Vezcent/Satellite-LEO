/*
 * S-MAS: Attitude Determination and Control System (ADCS) — Header
 * Models 3-axis quaternion rotational dynamics, reaction wheels, and torques.
 */
#pragma once
#include "types.h"

namespace smas {

struct AttitudeState {
    // Quaternion representing attitude (rotation from ECI to Body frame)
    double q0 = 1.0;
    double q1 = 0.0;
    double q2 = 0.0;
    double q3 = 0.0;

    // Angular velocity in Body frame (rad/s)
    double wx = 0.0;
    double wy = 0.0;
    double wz = 0.0;

    // Reaction wheel angular momentum (Nms)
    double wheel_h[3] = {0.0, 0.0, 0.0};

    // Derived diagnostic metrics
    double sun_angle = 0.0;    // Angle between panel normal (+Z body) and Sun vector (rad)
    double nadir_error = 0.0;  // Pointing error between optical axis (+Z body) and Nadir (rad)
};

class AttitudeModel {
public:
    AttitudeModel();

    void reset();

    // Propagate attitude dynamics using sub-stepping (dt_main = 5s, dt_sub = 1s)
    //   dt               : main time step (seconds, e.g. 5.0)
    //   pos_eci          : satellite position vector in ECI (m)
    //   sun_dir_eci      : unit direction vector pointing to the Sun in ECI
    //   wheel_torque_cmd : external torque command from the agent (3-axis, optional, default = nullptr)
    void step(double dt, const Vec3& pos_eci, const Vec3& sun_dir_eci, const double* wheel_torque_cmd = nullptr);

    // Get current state
    const AttitudeState& state() const { return state_; }

    // Useful getters
    double cos_sun_angle() const;
    bool is_nadir_pointing(double tol_rad = 0.0872665) const; // Default 5.0 degrees in radians

    // Dynamic state
    AttitudeState state_;

private:
    // Rotational equations solver for a single sub-step (sub_dt)
    void sub_step(double sub_dt, const Vec3& pos_eci, const Vec3& sun_dir_eci, const double* control_torque);

    // Dynamic calculations
    void integrate_quaternion(double sub_dt);
    void update_wheels(double sub_dt, const double* torque_cmd);
    Vec3 compute_gravity_gradient_torque(const Vec3& pos_eci) const;
    Vec3 compute_srp_torque(const Vec3& sun_dir_eci) const;

    // Quaternion multiplication utility
    void q_mult(const double* p, const double* q, double* r) const;

    // Closed-loop Nadir pointing PD control
    // Generates reaction wheel torques to align body +Z axis to Earth Center (Nadir)
    void pointing_control(const Vec3& pos_eci, double* out_wheel_torque) const;

    // Spacecraft properties
    double inertia_[3];        // Diagonal principal moments of inertia [Ixx, Iyy, Izz] (kg*m^2)
    double max_wheel_h_;       // Max angular momentum per wheel (Nms, default: 0.2)
    double max_wheel_torque_;  // Max torque per wheel (Nm, default: 0.01)
};

} // namespace smas
