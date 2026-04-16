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
//  Eclipse Detection (Cylindrical Shadow Model)
// ═══════════════════════════════════════════════════════════════════

bool is_in_eclipse(const Vec3& sat_pos_m, const Vec3& sun_dir) {
    // Cylindrical shadow model:
    // The satellite is in shadow if:
    //   1. It is on the anti-sun side of Earth (dot(pos, sun) < 0)
    //   2. Its distance from the Earth-Sun line < R_Earth

    double dot = sat_pos_m.dot(sun_dir);
    if (dot > 0.0) return false; // sunlit side

    // Project satellite position onto plane perpendicular to sun vector
    Vec3 proj = sat_pos_m - sun_dir * dot;
    double perp_dist = proj.magnitude();

    return perp_dist < constants::EARTH_RADIUS_M;
}

// ═══════════════════════════════════════════════════════════════════
//  Line-of-Sight to Ground Stations
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

bool is_visible(const Vec3& sat_pos_m,
                const GroundStation& gs,
                double gmst_rad) {
    double elev = elevation_angle(sat_pos_m, gs, gmst_rad);
    return elev >= gs.min_elevation_deg;
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
