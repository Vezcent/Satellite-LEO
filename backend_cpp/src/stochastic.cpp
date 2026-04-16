/*
 * S-MAS: Stochastic Realism & Epistemic Uncertainty — Implementation
 * Tasks 1.8, 1.9, 1.10
 */
#include "stochastic.h"
#include "types.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace smas {

// ═══════════════════════════════════════════════════════════════════
//  Sensor Noise (Task 1.8.1)
// ═══════════════════════════════════════════════════════════════════

SensorNoise::SensorNoise(uint64_t seed) : rng_(seed) {}

Vec3 SensorNoise::noisy_position(const Vec3& true_pos, double sigma_m) {
    return {true_pos.x + sigma_m * sample(),
            true_pos.y + sigma_m * sample(),
            true_pos.z + sigma_m * sample()};
}

Vec3 SensorNoise::noisy_velocity(const Vec3& true_vel, double sigma_ms) {
    return {true_vel.x + sigma_ms * sample(),
            true_vel.y + sigma_ms * sample(),
            true_vel.z + sigma_ms * sample()};
}

double SensorNoise::noisy_soc(double true_soc, double sigma) {
    return smas::compat::clamp(true_soc + sigma * sample(), 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════
//  SEU Anomaly Generator (Task 1.8.2)
// ═══════════════════════════════════════════════════════════════════

SEUGenerator::SEUGenerator(uint64_t seed) : rng_(seed) {}

bool SEUGenerator::check_seu(float saa_flux_10mev) {
    double prob = constants::SEU_BASE_PROB;
    if (saa_flux_10mev > constants::SAA_FLUX_THRESHOLD) {
        // Scale probability with flux intensity
        prob *= constants::SEU_SAA_MULT * (saa_flux_10mev / 1000.0);
    }
    prob = std::min(prob, 0.5); // cap
    return uniform_(rng_) < prob;
}

bool SEUGenerator::is_fatal(float saa_flux_10mev) {
    // Fatal SEU: very low probability even inside SAA
    double prob = constants::SEU_BASE_PROB * 0.01; // 1% of normal SEU rate
    if (saa_flux_10mev > constants::SAA_FLUX_THRESHOLD) {
        prob *= 10.0 * (saa_flux_10mev / 10000.0);
    }
    prob = std::min(prob, 0.01);
    return uniform_(rng_) < prob;
}

// ═══════════════════════════════════════════════════════════════════
//  Actuator Error & Execution Latency (Task 1.9)
// ═══════════════════════════════════════════════════════════════════

ActuatorModel::ActuatorModel(uint64_t seed)
    : rng_(seed),
      delay_dist_(constants::ACT_DELAY_MIN, constants::ACT_DELAY_MAX) {}

ActionPacket ActuatorModel::apply_error(const ActionPacket& cmd) {
    ActionPacket out = cmd;
    double err = constants::ACTUATOR_ERROR;

    // Apply ±5% deviation to continuous thrust commands
    out.thrust_x *= static_cast<float>(1.0 + err * gauss_(rng_));
    out.thrust_y *= static_cast<float>(1.0 + err * gauss_(rng_));
    out.thrust_z *= static_cast<float>(1.0 + err * gauss_(rng_));
    out.throttle *= static_cast<float>(1.0 + err * gauss_(rng_));

    // Clamp to valid ranges
    out.thrust_x = smas::compat::clamp(out.thrust_x, -1.0f, 1.0f);
    out.thrust_y = smas::compat::clamp(out.thrust_y, -1.0f, 1.0f);
    out.thrust_z = smas::compat::clamp(out.thrust_z, -1.0f, 1.0f);
    out.throttle = smas::compat::clamp(out.throttle, 0.0f, 1.0f);

    return out;
}

void ActuatorModel::enqueue(const ActionPacket& cmd) {
    int delay = delay_dist_(rng_);
    queue_.push_back({delay, cmd});
}

ActionPacket ActuatorModel::dequeue() {
    // Decrement delay counters and return the first ready action
    ActionPacket noop{};
    noop.version = 1;
    noop.thrust_x = 0; noop.thrust_y = 0; noop.thrust_z = 0;
    noop.throttle = 0; noop.deep_sleep = 0; noop.payload_on = 0;

    if (queue_.empty()) return noop;

    // Tick down all delays
    for (auto& item : queue_) {
        item.first--;
    }

    // Execute the first action whose delay has reached 0
    while (!queue_.empty() && queue_.front().first <= 0) {
        ActionPacket ready = queue_.front().second;
        queue_.pop_front();
        return ready;
    }

    return noop; // no action ready yet
}

void ActuatorModel::reset() {
    queue_.clear();
    current_step_ = 0;
}

// ═══════════════════════════════════════════════════════════════════
//  Epistemic Model Drift (Task 1.10)
// ═══════════════════════════════════════════════════════════════════

ModelDrift::ModelDrift(uint64_t seed) : rng_(seed) { reset(); }

void ModelDrift::reset() {
    cd_        = constants::SAT_CD_NOMINAL;
    panel_eff_ = 1.0;
}

void ModelDrift::step() {
    // Random walk: value += N(0, σ)
    cd_ += constants::CD_DRIFT_SIGMA * gauss_(rng_);
    cd_ = smas::compat::clamp(cd_, 1.5, 3.0); // physically plausible range

    panel_eff_ += constants::PANEL_DRIFT_SIGMA * gauss_(rng_);
    // Also apply slow secular degradation from radiation
    panel_eff_ -= 1e-8; // tiny per-step trend (cumulative over 15 years)
    panel_eff_ = smas::compat::clamp(panel_eff_, 0.3, 1.0);
}

} // namespace smas
