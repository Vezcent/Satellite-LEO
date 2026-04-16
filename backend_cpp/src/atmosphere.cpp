/*
 * S-MAS: NRLMSISE-00 Simplified Atmospheric Density Model — Implementation
 * Task 1.3
 *
 * Physical basis:
 *   1. Exospheric temperature T_inf driven by F10.7 and Ap
 *   2. Bates-Walker temperature profile T(h)
 *   3. Diffusive equilibrium density profile
 *   4. Diurnal and latitudinal modulation
 *
 * Accuracy target: captures ρ order-of-magnitude and F10.7/Ap trends at 600km.
 * For true mission-critical work, embed the full NRLMSISE-00 C source.
 */
#include "atmosphere.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace smas {

// Reference altitude (km) — lower boundary of the thermosphere
static constexpr double H_REF    = 120.0;
// Reference temperature at h_ref (K)
static constexpr double T_REF    = 355.0;
// Reference density at h_ref (kg/m³)
static constexpr double RHO_REF  = 2.0e-8;
// Mean molecular mass at 600km — mostly atomic oxygen (amu)
static constexpr double M_AVG_AMU = 16.0;
// Shape parameter for Bates-Walker profile (1/km)
static constexpr double SIGMA    = 0.02;

double NRLMSISEModel::exospheric_temp(double f107, double f107a, double ap) const {
    // Empirical fit to NRLMSISE-00 exospheric temperature behaviour.
    // T_inf ≈ 500 + 3.5*F10.7_avg + 1.5*(F10.7 - F10.7_avg) + 1.5*Ap
    double T_inf = 500.0
                 + 3.5 * f107a
                 + 1.5 * (f107 - f107a)
                 + 1.5 * ap;
    return std::max(T_inf, 600.0); // physical floor
}

double NRLMSISEModel::temperature_at_altitude(double altitude_km, double T_inf) const {
    // Bates-Walker temperature profile:
    //   T(h) = T_inf - (T_inf - T_120) * exp(-σ * (h - 120))
    if (altitude_km <= H_REF) return T_REF;
    double dh = altitude_km - H_REF;
    return T_inf - (T_inf - T_REF) * std::exp(-SIGMA * dh);
}

double NRLMSISEModel::scale_height(double altitude_km, double T) const {
    // H = k*T / (m*g)
    // g at altitude: g = g0 * (Re / (Re + h))²
    double Re = constants::EARTH_RADIUS_KM;
    double g = 9.80665 * (Re / (Re + altitude_km)) * (Re / (Re + altitude_km));
    double m = M_AVG_AMU * constants::AMU; // kg per particle
    return (constants::BOLTZMANN * T) / (m * g) / 1000.0; // convert m → km
}

double NRLMSISEModel::density(double altitude_km,
                               double lat_deg,
                               double lst_hours,
                               double f107,
                               double f107a,
                               double ap) const {
    if (altitude_km < H_REF) {
        // Below thermosphere — return a high density (not our domain)
        return 1.0e-5;
    }

    // 1. Exospheric temperature
    double T_inf = exospheric_temp(f107, f107a, ap);

    // 2. Temperature at altitude
    double T_h = temperature_at_altitude(altitude_km, T_inf);

    // 3. Scale height at this temperature
    double H = scale_height(altitude_km, T_h);

    // 4. Base density profile (exponential decay from reference)
    //    Integrate through layers for better accuracy:
    //    ρ(h) = ρ_ref * (T_ref / T(h))^(1+α) * exp(-integral dh'/H(h'))
    //    Simplified: single-layer exponential
    double dh = altitude_km - H_REF;
    double rho = RHO_REF * (T_REF / T_h) * std::exp(-dh / H);

    // 5. Diurnal variation: density bulge on the dayside
    //    Peak near 14:00 LST, minimum near 04:00 LST
    //    Modulation factor: 1 ± 0.4 (roughly)
    double lst_rad = (lst_hours - 14.0) * constants::PI / 12.0;
    double diurnal = 1.0 + 0.4 * std::cos(lst_rad);
    rho *= diurnal;

    // 6. Latitudinal variation (higher density near equator in thermosphere)
    double lat_rad = lat_deg * constants::DEG2RAD;
    double lat_factor = 1.0 + 0.1 * std::cos(2.0 * lat_rad);
    rho *= lat_factor;

    // 7. Floor to prevent negative/zero density
    return std::max(rho, 1.0e-20);
}

} // namespace smas
