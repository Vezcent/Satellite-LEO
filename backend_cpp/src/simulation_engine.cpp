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
    ok &= debris_catalog_.load(base + "debris_catalog.json");

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
    thermal_.reset();
    attitude_.reset();

    // Time
    sim_time_struct_.year = init.elements.epoch_year;
    sim_time_struct_.doy  = static_cast<int>(init.elements.epoch_day);
    sim_time_struct_.hour = static_cast<int>((init.elements.epoch_day -
                             sim_time_struct_.doy) * 24.0);
    sim_time_struct_.total_seconds = 0.0;

    // Tracking
    time_since_contact_ = 0.0;
    fdir_mode_ = FDIRMode::NOMINAL;
    data_buffer_mb_ = 0.0;

    // State
    std::memset(&state_, 0, sizeof(state_));
    state_.version = 1;
    state_.fdir_mode = static_cast<uint8_t>(FDIRMode::NOMINAL);
    state_.panel_efficiency = 1.0;
    state_.drag_coeff = constants::SAT_CD_NOMINAL;
}

void SimulationEngine::set_time(double time_s) {
    orbit_.time = time_s;
    update_time();
}

void SimulationEngine::set_degradation(double capacity_j, double panel_eff) {
    bus_.set_degradation(capacity_j);
    drift_.set_panel_efficiency(panel_eff);
}

void SimulationEngine::set_environment(double seu_multiplier,
                                       double noise_multiplier,
                                       double drift_rate_multiplier,
                                       double density_multiplier) {
    seu_multiplier_   = seu_multiplier;
    noise_multiplier_ = noise_multiplier;
    drift_multiplier_ = drift_rate_multiplier;
    cfg_.density_multiplier = density_multiplier;
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
        drift_.step(drift_multiplier_);
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
                                      lst, weather.f107, weather.f107a,
                                      weather.ap, weather.dst) * cfg_.density_multiplier;

    // SAA flux
    SAAFluxPoint flux = saa_.lookup(geo.latitude_deg, geo.longitude_deg);

    // Eclipse
    Vec3 sun = approximate_sun_direction(sim_time_struct_.year,
                                          sim_time_struct_.doy,
                                          sim_time_struct_.hour);
    double penumbra_factor = get_penumbra_factor(orbit_.pos, sun);
    bool eclipse = (penumbra_factor < 0.5);

    // Ground station visibility (RF link budget SNR check)
    double max_snr_db = -999.0;
    uint8_t gs_mask = 0;
    for (size_t i = 0; i < gs_list_.stations().size() && i < 8; ++i) {
        double snr_db = -999.0;
        double range_m = 0.0;
        bool visible = is_visible_link(orbit_.pos, gs_list_.stations()[i], gmst, snr_db, range_m);
        if (visible) {
            gs_mask |= (1u << i);
        }
        if (snr_db > max_snr_db) {
            max_snr_db = snr_db;
        }
    }

    // ── 5. Orbital integration ────────────────────────────────────────
    // Dynamic mass: account for fuel consumption
    double current_mass = constants::SAT_MASS_KG -
                          (constants::SAT_FUEL_KG - bus_.fuel_kg());

    AccelParams ap;
    ap.rho     = rho;
    ap.cd      = cfg_.enable_drift ? drift_.cd() : constants::SAT_CD_NOMINAL;
    ap.area_m2 = constants::SAT_AREA_M2;
    ap.mass_kg = current_mass;
    ap.year    = sim_time_struct_.year;
    ap.doy     = sim_time_struct_.doy;
    ap.hour_utc = sim_time_struct_.hour + 
                  (sim_time_struct_.total_seconds -
                   static_cast<int>(sim_time_struct_.total_seconds / 3600.0) * 3600.0) / 3600.0;

    // Build thrust acceleration (disabled if fuel depleted)
    Vec3 thrust_accel_vec(0.0, 0.0, 0.0);
    if (!bus_.is_fuel_depleted()) {
        Vec3 thrust_dir(exec_action.thrust_x, exec_action.thrust_y, exec_action.thrust_z);
        thrust_accel_vec = thrust_acceleration(thrust_dir, exec_action.throttle,
                                               cfg_.max_dv_per_step, current_mass);

        // Consume fuel (Tsiolkovsky)
        double dv = thrust_accel_vec.magnitude() * constants::DT;
        if (dv > 0.0) {
            double actual_dv = bus_.consume_fuel(dv, current_mass);
            if (actual_dv < dv && actual_dv > 0.0) {
                // Scale thrust to achievable dv
                double scale = actual_dv / dv;
                thrust_accel_vec = Vec3(thrust_accel_vec.x * scale,
                                        thrust_accel_vec.y * scale,
                                        thrust_accel_vec.z * scale);
            } else if (actual_dv <= 0.0) {
                thrust_accel_vec = Vec3(0.0, 0.0, 0.0);
            }
        }
    }
    ap.thrust_accel = thrust_accel_vec;

    orbit_ = rk4_step(orbit_, ap);

    // ── 5.5. Attitude dynamics (Phase A ADCS) ─────────────────────
    attitude_.step(constants::DT, orbit_.pos, sun, nullptr);

    // Pointing error constraint: CHRIS camera cannot image if pointing error > 5 degrees
    if (exec_action.payload_on != 0 && !attitude_.is_nadir_pointing(5.0 * constants::DEG2RAD)) {
        exec_action.payload_on = 0;
    }

    // ── 6. Power subsystem ────────────────────────────────────────
    double panel_eff = cfg_.enable_drift ? drift_.panel_efficiency() : 1.0;
    double cos_sun = attitude_.cos_sun_angle();
    bus_.update(penumbra_factor, panel_eff,
                exec_action.deep_sleep != 0,
                exec_action.payload_on != 0,
                cos_sun,
                constants::DT);

    // ── 6.5 Thermal subsystem (Phase A) ──────────────────────────
    thermal_.update(penumbra_factor, bus_.solar_power_w(),
                    bus_.power_draw_w(), constants::DT);
    bus_.apply_thermal_factor(thermal_.battery_temp_factor());
    double heater_w = thermal_.heater_power_w();
    if (heater_w > 0.0) {
        bus_.apply_heater_draw(heater_w, constants::DT);
    }

    // ── 7. Communication tracking ─────────────────────────────────
    // Camera data generation (10 Mbps = 1.25 MB/s)
    if (exec_action.payload_on != 0) {
        double data_generated = 1.25 * constants::DT;
        data_buffer_mb_ += data_generated;
        if (data_buffer_mb_ > 256.0) {
            data_buffer_mb_ = 256.0;
        }
    }

    // Downlink (1 Mbps = 0.125 MB/s)
    if (gs_mask != 0) {
        time_since_contact_ = 0.0;
        double data_downlinked = 0.125 * constants::DT;
        if (data_buffer_mb_ > 0.0) {
            data_buffer_mb_ -= std::min(data_buffer_mb_, data_downlinked);
        }
    } else {
        time_since_contact_ += constants::DT;
    }

    // ── 8. SEU check ──────────────────────────────────────────────
    bool seu_spike = false;
    bool seu_fatal = false;
    if (cfg_.enable_seu) {
        seu_spike = seu_gen_.check_seu(flux.flux_10mev, seu_multiplier_);
        if (seu_spike) {
            seu_fatal = seu_gen_.is_fatal(flux.flux_10mev);
        }
    }
    // Ground inject override (one-shot from Developer Testbed)
    if (raw_action.inject_seu) {
        seu_spike = true;
        // Ground-injected SEU should also have a chance to be fatal,
        // boosted by the current seu_multiplier for stress testing.
        if (!seu_fatal) {
            seu_fatal = seu_gen_.is_fatal(flux.flux_10mev);
        }
    }

    // ── 9. Failure contract check ─────────────────────────────────
    double alt = altitude_km(orbit_.pos);
    DoneReason done = bus_.check_failure(time_since_contact_, alt, seu_fatal);

    // ── 10. Build state packet ────────────────────────────────────
    state_.version = 4;
    state_.sim_time_s = orbit_.time;
    state_.year = sim_time_struct_.year;
    state_.doy  = sim_time_struct_.doy;
    state_.hour = sim_time_struct_.hour;

    // Orbital — optionally inject sensor noise
    if (cfg_.enable_noise) {
        Vec3 np = sensor_noise_.noisy_position(orbit_.pos, 50.0, noise_multiplier_);
        Vec3 nv = sensor_noise_.noisy_velocity(orbit_.vel, 0.5, noise_multiplier_);
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
    state_.battery_soc       = cfg_.enable_noise ? sensor_noise_.noisy_soc(bus_.soc(), 0.01, noise_multiplier_) : bus_.soc();
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

    // Fuel (Phase A)
    state_.fuel_fraction = static_cast<float>(bus_.fuel_fraction());
    state_.fuel_depleted = bus_.is_fuel_depleted() ? 1 : 0;

    // Thermal (Phase A)
    state_.temp_bus     = static_cast<float>(thermal_.temp_bus());
    state_.temp_battery = static_cast<float>(thermal_.temp_battery());
    state_.temp_payload = static_cast<float>(thermal_.temp_payload());
    state_.heater_on    = thermal_.heater_on() ? 1 : 0;

    // ADCS (Phase A ADCS)
    state_.sun_angle        = static_cast<float>(attitude_.state().sun_angle);
    state_.nadir_error      = static_cast<float>(attitude_.state().nadir_error);
    state_.wheel_momentum_x = static_cast<float>(attitude_.state().wheel_h[0]);
    state_.wheel_momentum_y = static_cast<float>(attitude_.state().wheel_h[1]);
    state_.wheel_momentum_z = static_cast<float>(attitude_.state().wheel_h[2]);

    // Comms & Data (Phase B Step 7)
    state_.data_buffer_mb = static_cast<float>(data_buffer_mb_);
    state_.snr_db         = static_cast<float>(max_snr_db);

    // ── Constellation & Debris (Task 8.4) ──
    double min_dist = 99999999.0;
    double time_to_tca = 9999.0;

    for (const auto& dobj : debris_catalog_.debris()) {
        OrbitalElements oe;
        oe.semi_major_axis_m = dobj.semi_major_axis_m;
        oe.eccentricity = dobj.eccentricity;
        oe.inclination_rad = dobj.inclination_rad;
        oe.raan_rad = dobj.raan_rad;
        oe.arg_perigee_rad = dobj.arg_perigee_rad;
        double n = std::sqrt(constants::EARTH_GM / (oe.semi_major_axis_m * oe.semi_major_axis_m * oe.semi_major_axis_m));
        oe.mean_anomaly_rad = std::fmod(dobj.mean_anomaly_rad + n * sim_time_struct_.total_seconds, constants::TWO_PI);

        Vec3 deb_pos, deb_vel;
        TLEParser::elements_to_eci(oe, deb_pos, deb_vel);

        double dist = (orbit_.pos - deb_pos).magnitude();
        if (dist < min_dist) {
            min_dist = dist;

            // Approximate TCA using relative position and velocity
            double r_dot_v = (orbit_.pos - deb_pos).dot(orbit_.vel - deb_vel);
            double v_rel_sq = (orbit_.vel - deb_vel).magnitude_sq();
            if (v_rel_sq > 1e-3) {
                double t = -r_dot_v / v_rel_sq;
                if (t > 0) {
                    time_to_tca = t;
                }
            }
        }
    }

    double risk = 0.0;
    if (min_dist < 5000.0) {
        risk = (5000.0 - min_dist) / 4000.0;
        if (risk > 1.0) risk = 1.0;
        if (risk < 0.0) risk = 0.0;
    }

    state_.conjunction_risk = static_cast<float>(risk);
    state_.time_to_tca_s    = static_cast<float>(time_to_tca);

    return state_;
}

} // namespace smas
