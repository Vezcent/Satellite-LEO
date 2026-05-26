/*
 * S-MAS: Math & Geometry Engine
 * Task 1.2 — Eclipse detection (Cylindrical Shadow Model) and
 *            Ground-Station Line-of-Sight (Spherical Trigonometry).
 */
#pragma once
#include "types.h"
#include "data_loader.h"
#include <vector>

namespace smas {

// ── Coordinate Conversions ────────────────────────────────────────

// ECI position → geodetic (lat_deg, lon_deg, alt_km).
// Uses a simplified spherical Earth model (sufficient for LEO drag work).
GeoCoord eci_to_geodetic(const Vec3& pos_eci_m, double gmst_rad);

// Compute Greenwich Mean Sidereal Time from simulation time.
// Approximate formula based on Julian centuries from J2000.
double compute_gmst(int year, int doy, int hour, double sec = 0.0);

// ── Eclipse Calculation (Conical Shadow Model with Penumbra) ──────

// Compute the penumbra factor (0.0 = full shadow, 1.0 = full sun)
//   sat_pos_m : satellite ECI position (m)
//   sun_dir   : unit vector pointing from Earth centre toward the Sun
double get_penumbra_factor(const Vec3& sat_pos_m, const Vec3& sun_dir);

// Returns true if the satellite is in Earth's shadow (factor < 0.5).
bool is_in_eclipse(const Vec3& sat_pos_m, const Vec3& sun_dir);

// Approximate Sun direction (unit vector, ECI) for given time.
// Uses low-precision solar ephemeris (~1° accuracy, sufficient here).
Vec3 approximate_sun_direction(int year, int doy, double hour_utc);

// ── Line-of-Sight & RF Link to Ground Stations ────────────────────

// Returns the elevation angle (degrees) of the satellite as seen
// from a ground station.  Negative ⇒ below horizon.
double elevation_angle(const Vec3& sat_pos_m,
                       const GroundStation& gs,
                       double gmst_rad);

// Check if satellite is visible from a ground station (elevation > mask) and RF link is locked.
bool is_visible(const Vec3& sat_pos_m,
                const GroundStation& gs,
                double gmst_rad);

// Detailed Link Budget calculation
bool is_visible_link(const Vec3& sat_pos_m,
                     const GroundStation& gs,
                     double gmst_rad,
                     double& out_snr_db,
                     double& out_slant_range_m);

// Returns a bitmask of visible stations (bit 0 = first station, etc.)
uint8_t visible_stations_mask(const Vec3& sat_pos_m,
                              const std::vector<GroundStation>& stations,
                              double gmst_rad);

// ── Beta Angle ────────────────────────────────────────────────────

// Compute Beta angle: angle between orbital plane and Sun vector.
double beta_angle(const Vec3& pos_m, const Vec3& vel_ms, const Vec3& sun_dir);

} // namespace smas
