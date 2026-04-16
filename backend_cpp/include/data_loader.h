/*
 * S-MAS: Data Parsers & Memory Loaders
 * Task 1.1 — Loads all preprocessed data into memory-efficient structures.
 */
#pragma once
#include "types.h"
#include <string>
#include <vector>
#include <map>
#include <array>

namespace smas {

// ── Space Weather Record ──────────────────────────────────────────
struct SpaceWeatherRecord {
    int32_t year, doy, hour;
    double  kp;       // normalised Kp index
    double  dst;      // Dst index (nT)
    double  ap;       // Ap index
    double  f107;     // F10.7 solar radio flux (sfu)
};

// ── Space Weather Table (time-indexed) ────────────────────────────
class SpaceWeatherTable {
public:
    bool load(const std::string& csv_path);

    // Lookup by exact (year, doy, hour) — returns nearest record.
    const SpaceWeatherRecord& lookup(int year, int doy, int hour) const;

    // Linear interpolation at fractional time (hours since epoch 2000-001-00).
    SpaceWeatherRecord interpolate(double hours_since_epoch) const;

    size_t size() const { return records_.size(); }

private:
    std::vector<SpaceWeatherRecord> records_;
    // key = hours_since_epoch (integer); value = index into records_
    std::map<int64_t, size_t> index_;

    int64_t to_key(int year, int doy, int hour) const;
};

// ── SAA Heatmap Grid ──────────────────────────────────────────────
struct SAAFluxPoint {
    float flux_10mev;
    float flux_30mev;
};

class SAAHeatmap {
public:
    bool load(const std::string& csv_path);

    // Fast 2-D lookup with bilinear interpolation.
    SAAFluxPoint lookup(double latitude_deg, double longitude_deg) const;

    size_t point_count() const { return data_.size(); }

private:
    // Grid parameters (auto-detected from data)
    double lat_min_, lat_max_, lat_step_;
    double lon_min_, lon_max_, lon_step_;
    int    n_lat_, n_lon_;
    std::vector<SAAFluxPoint> data_;  // row-major [lat][lon]

    int grid_index(int lat_idx, int lon_idx) const;
};

// ── Ground Station ────────────────────────────────────────────────
struct GroundStation {
    std::string id;
    std::string name;
    std::string country;
    double latitude_deg;
    double longitude_deg;
    double altitude_m;
    double min_elevation_deg;
    std::string role;
};

class GroundStationList {
public:
    bool load(const std::string& json_path);
    const std::vector<GroundStation>& stations() const { return stations_; }

private:
    std::vector<GroundStation> stations_;
    // Minimal JSON string-value extractor (not general-purpose).
    static std::string extract_string(const std::string& json, const std::string& key);
    static double      extract_number(const std::string& json, const std::string& key);
};

// ── TLE / Initial State ───────────────────────────────────────────
struct OrbitalElements {
    double semi_major_axis_m;
    double eccentricity;
    double inclination_rad;
    double raan_rad;         // Right Ascension of Ascending Node
    double arg_perigee_rad;  // Argument of Perigee
    double mean_anomaly_rad;
    double mean_motion_rad_s;
    int    epoch_year;
    double epoch_day;        // fractional day of year
};

struct InitialState {
    Vec3 position_m;   // ECI (m)
    Vec3 velocity_ms;  // ECI (m/s)
    OrbitalElements elements;
};

class TLEParser {
public:
    bool load(const std::string& tle_path);
    const InitialState& state() const { return state_; }

private:
    InitialState state_;

    // Kepler equation solver (Newton-Raphson)
    static double solve_kepler(double M, double e, int max_iter = 30);
    // Convert orbital elements → ECI state vector
    static void elements_to_eci(const OrbitalElements& oe, Vec3& pos, Vec3& vel);
};

} // namespace smas
