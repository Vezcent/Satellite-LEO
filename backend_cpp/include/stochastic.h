/*
 * S-MAS: Stochastic Realism & Epistemic Uncertainty
 * Tasks 1.8, 1.9, 1.10 — Sensor noise, actuator errors,
 *                         SEU anomalies, action queue, model drift.
 */
#pragma once
#include "types.h"
#include "contracts.h"
#include <random>
#include <deque>

namespace smas {

// ── Sensor Noise Injector (Task 1.8.1) ────────────────────────────
class SensorNoise {
public:
    explicit SensorNoise(uint64_t seed = 42);

    // Inject Gaussian noise into position reading (metres).
    Vec3 noisy_position(const Vec3& true_pos, double sigma_m = 50.0);

    // Inject Gaussian noise into velocity reading (m/s).
    Vec3 noisy_velocity(const Vec3& true_vel, double sigma_ms = 0.5);

    // Inject noise into power SoC reading (fractional).
    double noisy_soc(double true_soc, double sigma = 0.01);

private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};
    double sample() { return gauss_(rng_); }
};

// ── SEU Anomaly Generator (Task 1.8.2) ────────────────────────────
class SEUGenerator {
public:
    explicit SEUGenerator(uint64_t seed = 123);

    // Returns true if a non-fatal SEU spike occurs this step.
    // Probability scales with SAA proton flux.
    bool check_seu(float saa_flux_10mev);

    // Returns true if the SEU is fatal (extremely rare, but possible).
    bool is_fatal(float saa_flux_10mev);

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

// ── Actuator Error & Execution Latency (Task 1.9) ─────────────────
class ActuatorModel {
public:
    explicit ActuatorModel(uint64_t seed = 456);

    // Apply ±5% deviation to a thruster command.
    ActionPacket apply_error(const ActionPacket& cmd);

    // Push action into the execution queue (1-3 step delay).
    void enqueue(const ActionPacket& cmd);

    // Pop the next executable action (or a no-op if queue is empty).
    ActionPacket dequeue();

    void reset();

private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};
    std::uniform_int_distribution<int> delay_dist_;
    std::deque<std::pair<int, ActionPacket>> queue_; // (steps_remaining, action)

    int current_step_ = 0;
};

// ── Epistemic Model Drift (Task 1.10) ─────────────────────────────
class ModelDrift {
public:
    explicit ModelDrift(uint64_t seed = 789);

    void reset();

    // Advance one step: apply random walk to Cd and panel efficiency.
    void step();

    void set_panel_efficiency(double eff);

    double cd()               const { return cd_; }
    double panel_efficiency() const { return panel_eff_; }

private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};

    double cd_;         // current drag coefficient
    double panel_eff_;  // current panel efficiency [0,1]
};

} // namespace smas
