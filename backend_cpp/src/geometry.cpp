/*
 * S-MAS: Math & Geometry Engine — Implementation
 * Task 1.2 — Eclipse (Cylindrical Shadow), LoS (Spherical Trig),
 *            coordinate conversions, solar ephemeris.
 */
#include "geometry.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace smas {

// ═══════════════════════════════════════════════════════════════════
//  Coordinate Conversions
// ═══════════════════════════════════════════════════════════════════

double compute_gmst(int year, int doy, int hour, double sec) {
    // Julian date of J2000.0 = 2451545.0
    // Approximate Julian Date from year/doy/hour
    int y = year;
    // Julian day number for Jan 1 of year
    int a = (14 - 1) / 12;
    int yy = y + 4800 - a;
    int mm = 1 + 12 * a - 3;
    double JD0 = 1 + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045;
    double JD = JD0 + (doy - 1) + (hour + sec / 3600.0) / 24.0;

    double T = (JD - 2451545.0) / 36525.0;    // Julian centuries since J2000
    // GMST in degrees (IAU 1982 formula, simplified)
    double gmst_deg = 280.46061837
                    + 360.98564736629 * (JD - 2451545.0)
                    + 0.000387933 * T * T;
    gmst_deg = std::fmod(gmst_deg, 360.0);
    if (gmst_deg < 0) gmst_deg += 360.0;
    return gmst_deg * constants::DEG2RAD;
}

GeoCoord eci_to_geodetic(const Vec3& pos_eci_m, double gmst_rad) {
    GeoCoord geo;
    double r = pos_eci_m.magnitude();
    geo.altitude_km = r / 1000.0 - constants::EARTH_RADIUS_KM;

    // Geographic latitude (geocentric, sufficient for spherical Earth model)
    geo.latitude_deg = std::asin(pos_eci_m.z / r) * constants::RAD2DEG;

    // Geographic longitude = atan2(y,x) - GMST
    double lon_rad = std::atan2(pos_eci_m.y, pos_eci_m.x) - gmst_rad;
    // Normalise to [-π, π]
    while (lon_rad >  constants::PI) lon_rad -= constants::TWO_PI;
    while (lon_rad < -constants::PI) lon_rad += constants::TWO_PI;
    geo.longitude_deg = lon_rad * constants::RAD2DEG;

    return geo;
}

// ═══════════════════════════════════════════════════════════════════
//  Solar Ephemeris (low-precision, ~1° accuracy)
// ═══════════════════════════════════════════════════════════════════

Vec3 approximate_sun_direction(int year, int doy, double hour_utc) {
    // Approximate ecliptic longitude of the Sun
    // n = days since J2000.0 (2000 Jan 1 12:00 TT)
    double n = (year - 2000) * 365.25 + (doy - 1) + hour_utc / 24.0 - 0.5;

    // Mean longitude (degrees)
    double L = std::fmod(280.460 + 0.9856474 * n, 360.0);
    // Mean anomaly (degrees)
    double g = std::fmod(357.528 + 0.9856003 * n, 360.0);
    double g_rad = g * constants::DEG2RAD;

    // Ecliptic longitude (degrees)
    double lambda = L + 1.915 * std::sin(g_rad) + 0.020 * std::sin(2.0 * g_rad);
    double lambda_rad = lambda * constants::DEG2RAD;

    // Obliquity of ecliptic
    double eps = 23.439 - 0.0000004 * n;
    double eps_rad = eps * constants::DEG2RAD;

    // Sun direction in ECI (geocentric equatorial)
    Vec3 sun;
    sun.x = std::cos(lambda_rad);
    sun.y = std::cos(eps_rad) * std::sin(lambda_rad);
    sun.z = std::sin(eps_rad) * std::sin(lambda_rad);

    return sun.normalized();
}

// ═══════════════════════════════════════════════════════════════════
//  Eclipse Detection (Conical Shadow Model with Penumbra & Oblate Earth)
// ═══════════════════════════════════════════════════════════════════

double get_penumbra_factor(const Vec3& sat_pos_m, const Vec3& sun_dir) {
    // Distance along the shadow axis (away from Sun)
    double s = -sat_pos_m.dot(sun_dir);
    if (s <= 0.0) {
        // Satellite is on the sunlit side of the Earth
        return 1.0;
    }

    // Projection of satellite onto the terminator plane (perpendicular to sun_dir)
    Vec3 proj = sat_pos_m + sun_dir * s;
    double d = proj.magnitude();

    // Earth parameters (WGS84)
    double R_eq = 6378137.0;       // equatorial radius (m)
    double f = 1.0 / 298.257223563; // flattening

    double sin2_phi_p = 0.0;
    if (d > 1e-6) {
        double sin_phi_p = proj.z / d;
        sin2_phi_p = sin_phi_p * sin_phi_p;
    }
    double R_eff = R_eq * (1.0 - f * sin2_phi_p);

    // Sun parameters
    double R_sun = 6.9634e8;        // Sun radius (m)
    double d_sun = 1.496e11;        // Earth-Sun distance (1 AU)

    // Conical shadow radii at distance s
    double R_u = R_eff - s * (R_sun - R_eff) / d_sun;
    double R_p = R_eff + s * (R_sun + R_eff) / d_sun;

    if (d <= R_u) {
        return 0.0; // Umbra (full shadow)
    }
    if (d >= R_p) {
        return 1.0; // Fully illuminated
    }

    // Penumbra (partial shadow) - linear interpolation
    double factor = (d - R_u) / (R_p - R_u);
    return smas::compat::clamp(factor, 0.0, 1.0);
}

bool is_in_eclipse(const Vec3& sat_pos_m, const Vec3& sun_dir) {
    double factor = get_penumbra_factor(sat_pos_m, sun_dir);
    return factor < 0.5;
}

// ═══════════════════════════════════════════════════════════════════
//  Line-of-Sight & RF Link to Ground Stations
// ═══════════════════════════════════════════════════════════════════

// Convert ground station lat/lon/alt to ECEF position
static Vec3 gs_to_ecef(const GroundStation& gs) {
    double lat_r = gs.latitude_deg * constants::DEG2RAD;
    double lon_r = gs.longitude_deg * constants::DEG2RAD;
    double R = constants::EARTH_RADIUS_M + gs.altitude_m;
    return {R * std::cos(lat_r) * std::cos(lon_r),
            R * std::cos(lat_r) * std::sin(lon_r),
            R * std::sin(lat_r)};
}

// ECEF to ECI (rotate by GMST)
static Vec3 ecef_to_eci(const Vec3& ecef, double gmst_rad) {
    double c = std::cos(gmst_rad), s = std::sin(gmst_rad);
    return {ecef.x * c - ecef.y * s,
            ecef.x * s + ecef.y * c,
            ecef.z};
}

double elevation_angle(const Vec3& sat_pos_m,
                       const GroundStation& gs,
                       double gmst_rad) {
    Vec3 gs_ecef = gs_to_ecef(gs);
    Vec3 gs_eci  = ecef_to_eci(gs_ecef, gmst_rad);

    Vec3 diff = sat_pos_m - gs_eci;
    double range = diff.magnitude();
    if (range < 1.0) return 90.0; // coincident

    // Up vector at ground station (radial, normalised)
    Vec3 up = gs_eci.normalized();

    // Elevation = asin(dot(diff_unit, up))
    double sinElev = diff.dot(up) / range;
    return std::asin(smas::compat::clamp(sinElev, -1.0, 1.0)) * constants::RAD2DEG;
}

bool is_visible_link(const Vec3& sat_pos_m,
                     const GroundStation& gs,
                     double gmst_rad,
                     double& out_snr_db,
                     double& out_slant_range_m) {
    double elev = elevation_angle(sat_pos_m, gs, gmst_rad);
    if (elev < gs.min_elevation_deg) {
        out_snr_db = -999.0;
        out_slant_range_m = (sat_pos_m - ecef_to_eci(gs_to_ecef(gs), gmst_rad)).magnitude();
        return false;
    }

    // Slant range
    Vec3 gs_ecef = gs_to_ecef(gs);
    Vec3 gs_eci  = ecef_to_eci(gs_ecef, gmst_rad);
    Vec3 diff = sat_pos_m - gs_eci;
    double range = diff.magnitude();
    out_slant_range_m = range;

    // Antenna off-nadir angle (assuming satellite is nadir-pointing)
    // Sat vector: sat_pos_m
    // Vector from satellite to ground station: -diff
    // cos(off_nadir) = (sat_pos_m . diff) / (sat_pos_m.magnitude() * range)
    double sat_norm = sat_pos_m.magnitude();
    double cos_off_nadir = (range > 0.0 && sat_norm > 0.0) ? sat_pos_m.dot(diff) / (sat_norm * range) : 1.0;
    cos_off_nadir = smas::compat::clamp(cos_off_nadir, -1.0, 1.0);
    double off_nadir_angle = std::acos(cos_off_nadir);

    // Antenna gain (0 dBi peak, cosine roll-off)
    double G_t = 1.0; // 0 dBi = 1.0 linear
    if (off_nadir_angle < constants::PI / 2.0) {
        G_t = std::cos(off_nadir_angle);
    } else {
        G_t = 0.0;
    }

    // Path loss (FSPL)
    double freq = 2.2e9; // 2.2 GHz S-band
    double c = 299792458.0;
    double lambda = c / freq;
    double path_loss = std::pow((4.0 * constants::PI * range) / lambda, 2.0);

    // Rx Gain: 30 dBi = 1000.0 linear
    double G_r = 1000.0;

    // Tx Power: 2 W
    double P_t = 2.0;

    // Received Power
    double P_r = (path_loss > 0.0) ? (P_t * G_t * G_r) / path_loss : 0.0;

    // Noise power: N = k T B
    // T = 290 K, B = 1 MHz
    double N = constants::BOLTZMANN * 290.0 * 1.0e6;

    double snr = (N > 0.0) ? P_r / N : 0.0;
    out_snr_db = (snr > 0.0) ? 10.0 * std::log10(snr) : -999.0;

    return out_snr_db >= 10.0;
}

bool is_visible(const Vec3& sat_pos_m,
                const GroundStation& gs,
                double gmst_rad) {
    double snr_db, range_m;
    return is_visible_link(sat_pos_m, gs, gmst_rad, snr_db, range_m);
}

uint8_t visible_stations_mask(const Vec3& sat_pos_m,
                              const std::vector<GroundStation>& stations,
                              double gmst_rad) {
    uint8_t mask = 0;
    for (size_t i = 0; i < stations.size() && i < 8; ++i) {
        if (is_visible(sat_pos_m, stations[i], gmst_rad))
            mask |= (1u << i);
    }
    return mask;
}

// ═══════════════════════════════════════════════════════════════════
//  Beta Angle
// ═══════════════════════════════════════════════════════════════════

double beta_angle(const Vec3& pos_m, const Vec3& vel_ms, const Vec3& sun_dir) {
    // Orbital angular momentum (normal to orbital plane)
    Vec3 h = pos_m.cross(vel_ms).normalized();
    // Beta = 90° - angle(h, sun) = asin(dot(h, sun))
    double sinBeta = h.dot(sun_dir);
    return std::asin(smas::compat::clamp(sinBeta, -1.0, 1.0)) * constants::RAD2DEG;
}

} // namespace smas
