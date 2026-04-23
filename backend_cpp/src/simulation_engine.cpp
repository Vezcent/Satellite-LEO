/*
 * S-MAS: Simulation Engine — Implementation
 * The master orchestrator tying all Phase 1 subsystems together.
 */
#include "simulation_engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace smas {

SimulationEngine::SimulationEngine(const EngineConfig& cfg)
    : cfg_(cfg),
      sensor_noise_(cfg.seed),
      seu_gen_(cfg.seed + 1),
      actuator_(cfg.seed + 2),
      drift_(cfg.seed + 3),
      time_since_contact_(0.0),
      fdir_mode_(FDIRMode::NOMINAL) {
    std::memset(&state_, 0, sizeof(state_));
}

bool SimulationEngine::init() {
    std::string base = cfg_.data_dir;
    // Normalise trailing separator
    if (!base.empty() && base.back() != '/' && base.back() != '\\')
        base += '/';

    bool ok = true;
    ok &= weather_.load(base + "space_weather.csv");
    ok &= saa_.load(base + "saa_heatmap_600km.csv");
    ok &= gs_list_.load(base + "ground_stations.json");
    ok &= tle_.load(base + "initial_state.txt");

    if (!ok) {
        std::cerr << "[Engine] Data loading failed.\n";
    } else {
        std::cout << "[Engine] All data loaded successfully.\n";
    }
    return ok;
}

void SimulationEngine::reset() {
    // Orbital state from TLE
    const auto& init = tle_.state();
    orbit_.pos  = init.position_m;
    orbit_.vel  = init.velocity_ms;
    orbit_.time = 0.0;

    // Subsystems
    bus_.reset();
    actuator_.reset();
    drift_.reset();

    // Time
    sim_time_struct_.year = init.elements.epoch_year;
    sim_time_struct_.doy  = static_cast<int>(init.elements.epoch_day);
    sim_time_struct_.hour = static_cast<int>((init.elements.epoch_day -
                             sim_time_struct_.doy) * 24.0);
    sim_time_struct_.total_seconds = 0.0;

    // Tracking
    time_since_contact_ = 0.0;
    fdir_mode_ = FDIRMode::NOMINAL;

    // State
    std::memset(&state_, 0, sizeof(state_));
    state_.version = 1;
    state_.fdir_mode = static_cast<uint8_t>(FDIRMode::NOMINAL);
    state_.panel_efficiency = 1.0;
    state_.drag_coeff = constants::SAT_CD_NOMINAL;
}

void SimulationEngine::update_time() {
    sim_time_struct_.total_seconds = orbit_.time;

    // Convert total seconds to (year, doy, hour)
    double epoch_day_frac = tle_.state().elements.epoch_day;
    int    epoch_year     = tle_.state().elements.epoch_year;
    int    epoch_doy      = static_cast<int>(epoch_day_frac);
    double epoch_hour     = (epoch_day_frac - epoch_doy) * 24.0;

    double total_hours = orbit_.time / 3600.0 + epoch_hour;
    int    total_days  = static_cast<int>(total_hours / 24.0);
    double rem_hours   = total_hours - total_days * 24.0;

    int doy = epoch_doy + total_days;
    int year = epoch_year;

    // Roll over years (simplified: 365 days/year, ignore leap)
    while (doy > 365) {
        doy -= 365;
        year++;
    }

    sim_time_struct_.year = year;
    sim_time_struct_.doy  = doy;
    sim_time_struct_.hour = static_cast<int>(rem_hours);
}

SpaceWeatherRecord SimulationEngine::get_current_weather() const {
    return weather_.lookup(sim_time_struct_.year,
                           sim_time_struct_.doy,
                           sim_time_struct_.hour);
}

double SimulationEngine::compute_local_solar_time(double lon_deg) const {
    double utc_hour = sim_time_struct_.hour +
                      (sim_time_struct_.total_seconds -
                       static_cast<int>(sim_time_struct_.total_seconds / 3600.0) * 3600.0) / 3600.0;
    double lst = utc_hour + lon_deg / 15.0;
    while (lst < 0.0)  lst += 24.0;
    while (lst >= 24.0) lst -= 24.0;
    return lst;
}

void SimulationEngine::update_fdir() {
    double soc = bus_.soc();

    switch (fdir_mode_) {
        case FDIRMode::NOMINAL:
            if (soc < 0.10)      fdir_mode_ = FDIRMode::SAFE;
            else if (soc < 0.20) fdir_mode_ = FDIRMode::DEGRADED;
            break;

        case FDIRMode::DEGRADED:
            if (soc < 0.10)      fdir_mode_ = FDIRMode::SAFE;
            else if (soc >= 0.25) fdir_mode_ = FDIRMode::RECOVERY;
            break;

        case FDIRMode::SAFE:
            if (soc >= 0.15)     fdir_mode_ = FDIRMode::RECOVERY;
            break;

        case FDIRMode::RECOVERY:
            if (soc >= 0.30)     fdir_mode_ = FDIRMode::NOMINAL;
            else if (soc < 0.10) fdir_mode_ = FDIRMode::SAFE;
            break;
    }
}

StatePacket SimulationEngine::step(const ActionPacket& raw_action) {
    // ── 1. FDIR override ──────────────────────────────────────────
    ActionPacket action = raw_action;

    // FDIR Governor: restrict/override AI actions based on mode
    update_fdir();

    switch (fdir_mode_) {
        case FDIRMode::DEGRADED:
            action.payload_on = 0; // force payload OFF
            action.throttle = std::min(action.throttle, 0.3f); // cap thrust
            break;
        case FDIRMode::SAFE:
            action.payload_on = 0;
            action.deep_sleep = 1; // force deep sleep
            action.throttle   = 0; // disable thrusters
            action.thrust_x = action.thrust_y = action.thrust_z = 0;
            break;
        case FDIRMode::RECOVERY:
            action.payload_on = 0; // keep payload off during recovery
            break;
        default:
            break;
    }

    // Meta-coordination: if deep sleep, force payload off
    if (action.deep_sleep) action.payload_on = 0;

    // ── 2. Actuator pipeline (delay + error) ──────────────────────
    ActionPacket exec_action = action;
    if (cfg_.enable_delay) {
        actuator_.enqueue(action);
        exec_action = actuator_.dequeue();
    }
    if (cfg_.enable_noise) {
        exec_action = actuator_.apply_error(exec_action);
    }

    // ── 3. Epistemic drift ────────────────────────────────────────
    if (cfg_.enable_drift) {
        drift_.step();
    }

    // ── 4. Environment queries ────────────────────────────────────
    update_time();
    auto weather = get_current_weather();

    double gmst = compute_gmst(sim_time_struct_.year,
                                sim_time_struct_.doy,
                                sim_time_struct_.hour);

    GeoCoord geo = eci_to_geodetic(orbit_.pos, gmst);

    double lst = compute_local_solar_time(geo.longitude_deg);

    // Atmospheric density
    double rho = atmosphere_.density(geo.altitude_km, geo.latitude_deg,
                                      lst, weather.f107, weather.f107,
                                      weather.ap) * cfg_.density_multiplier;

    // SAA flux
    SAAFluxPoint flux = saa_.lookup(geo.latitude_deg, geo.longitude_deg);

    // Eclipse
    Vec3 sun = approximate_sun_direction(sim_time_struct_.year,
                                          sim_time_struct_.doy,
                                          sim_time_struct_.hour);
    bool eclipse = is_in_eclipse(orbit_.pos, sun);

    // Ground station visibility
    uint8_t gs_mask = visible_stations_mask(orbit_.pos,
                                             gs_list_.stations(), gmst);

    // ── 5. Orbital integration ────────────────────────────────────
    AccelParams ap;
    ap.rho     = rho;
    ap.cd      = cfg_.enable_drift ? drift_.cd() : constants::SAT_CD_NOMINAL;
    ap.area_m2 = constants::SAT_AREA_M2;
    ap.mass_kg = constants::SAT_MASS_KG;

    // Build thrust acceleration
    Vec3 thrust_dir(exec_action.thrust_x, exec_action.thrust_y, exec_action.thrust_z);
    ap.thrust_accel = thrust_acceleration(thrust_dir, exec_action.throttle,
                                           cfg_.max_dv_per_step, ap.mass_kg);

    orbit_ = rk4_step(orbit_, ap);

    // ── 6. Power subsystem ────────────────────────────────────────
    double panel_eff = cfg_.enable_drift ? drift_.panel_efficiency() : 1.0;
    bus_.update(eclipse, panel_eff,
                exec_action.deep_sleep != 0,
                exec_action.payload_on != 0,
                constants::DT);

    // ── 7. Communication tracking ─────────────────────────────────
    if (gs_mask != 0) {
        time_since_contact_ = 0.0;
    } else {
        time_since_contact_ += constants::DT;
    }

    // ── 8. SEU check ──────────────────────────────────────────────
    bool seu_spike = false;
    bool seu_fatal = false;
    if (cfg_.enable_seu) {
        seu_spike = seu_gen_.check_seu(flux.flux_10mev);
        if (seu_spike) {
            seu_fatal = seu_gen_.is_fatal(flux.flux_10mev);
        }
    }

    // ── 9. Failure contract check ─────────────────────────────────
    double alt = altitude_km(orbit_.pos);
    DoneReason done = bus_.check_failure(time_since_contact_, alt, seu_fatal);

    // ── 10. Build state packet ────────────────────────────────────
    state_.version = 1;
    state_.sim_time_s = orbit_.time;
    state_.year = sim_time_struct_.year;
    state_.doy  = sim_time_struct_.doy;
    state_.hour = sim_time_struct_.hour;

    // Orbital — optionally inject sensor noise
    if (cfg_.enable_noise) {
        Vec3 np = sensor_noise_.noisy_position(orbit_.pos);
        Vec3 nv = sensor_noise_.noisy_velocity(orbit_.vel);
        state_.pos_x = np.x; state_.pos_y = np.y; state_.pos_z = np.z;
        state_.vel_x = nv.x; state_.vel_y = nv.y; state_.vel_z = nv.z;
    } else {
        state_.pos_x = orbit_.pos.x; state_.pos_y = orbit_.pos.y; state_.pos_z = orbit_.pos.z;
        state_.vel_x = orbit_.vel.x; state_.vel_y = orbit_.vel.y; state_.vel_z = orbit_.vel.z;
    }
    state_.altitude_km   = alt;
    state_.latitude_deg  = geo.latitude_deg;
    state_.longitude_deg = geo.longitude_deg;

    // Power
    state_.battery_soc       = cfg_.enable_noise ? sensor_noise_.noisy_soc(bus_.soc()) : bus_.soc();
    state_.battery_capacity_j = bus_.capacity_j();
    state_.solar_power_w     = bus_.solar_power_w();
    state_.power_draw_w      = bus_.power_draw_w();

    // Environment
    state_.atm_density     = rho;
    state_.drag_force_n    = std::abs(drag_acceleration(orbit_.pos, orbit_.vel,
                                                         rho, ap.cd,
                                                         ap.area_m2, ap.mass_kg).magnitude() * ap.mass_kg);
    state_.saa_flux_10mev  = flux.flux_10mev;
    state_.saa_flux_30mev  = flux.flux_30mev;
    state_.in_eclipse      = eclipse ? 1 : 0;
    state_.in_saa          = (flux.flux_10mev > constants::SAA_FLUX_THRESHOLD) ? 1 : 0;

    // Communication
    state_.gs_visible          = gs_mask;
    state_.time_since_contact_s = time_since_contact_;

    // FDIR
    state_.fdir_mode = static_cast<uint8_t>(fdir_mode_);

    // Degradation
    state_.panel_efficiency = panel_eff;
    state_.drag_coeff       = ap.cd;
    state_.charge_cycles    = bus_.charge_cycles();

    // Terminal
    state_.is_done     = (done != DoneReason::ONGOING) ? 1 : 0;
    state_.done_reason = static_cast<uint8_t>(done);

    // SEU
    state_.seu_active = seu_spike ? 1 : 0;

    return state_;
}

} // namespace smas
