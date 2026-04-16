/*
 * S-MAS: Power Subsystem & Realistic Degradation
 * Task 1.6 — Battery SoC, GaAs solar arrays, Arrhenius degradation,
 *            and the Failure Contract terminal conditions.
 */
#pragma once
#include "contracts.h"

namespace smas {

class SatelliteBus {
public:
    SatelliteBus();

    void reset();

    // ── Power update for one time step (dt = 5 s) ─────────────────
    //   in_eclipse     : true if in Earth's shadow
    //   panel_eff      : current solar panel efficiency [0,1]
    //   deep_sleep     : Resource Agent flag
    //   payload_on     : Mission Agent flag
    //   dt             : time step (seconds)
    void update(bool in_eclipse, double panel_eff,
                bool deep_sleep, bool payload_on, double dt);

    // ── Getters ───────────────────────────────────────────────────
    double soc()              const { return soc_; }           // [0,1]
    double capacity_j()       const { return capacity_j_; }
    double solar_power_w()    const { return solar_w_; }
    double power_draw_w()     const { return draw_w_; }
    uint32_t charge_cycles()  const { return cycles_; }

    // ── Battery Degradation ───────────────────────────────────────
    //   Arrhenius-based: capacity loss per charge/discharge cycle.
    //   Called once each time SoC crosses a threshold.
    void apply_cycle_degradation();

    // ── Failure Contract (Done) ───────────────────────────────────
    //   time_since_contact : seconds without ground station contact
    //   altitude_km        : current altitude
    //   seu_fatal          : true if a fatal SEU just fired
    // Returns DoneReason (0 = ongoing).
    DoneReason check_failure(double time_since_contact,
                             double altitude_km,
                             bool   seu_fatal) const;

private:
    double   soc_;           // [0,1]
    double   capacity_j_;    // current max capacity (Joules)
    double   solar_w_;       // instantaneous solar generation
    double   draw_w_;        // instantaneous total draw
    uint32_t cycles_;        // accumulated charge/discharge count
    bool     was_charging_;  // for cycle detection edge
};

} // namespace smas
