/*
 * S-MAS: Data Parsers & Memory Loaders — Implementation
 * Task 1.1
 */
#include "data_loader.h"
#include "constants.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace smas {

// ═══════════════════════════════════════════════════════════════════
//  Helper: trim whitespace
// ═══════════════════════════════════════════════════════════════════
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ═══════════════════════════════════════════════════════════════════
//  SpaceWeatherTable
// ═══════════════════════════════════════════════════════════════════

int64_t SpaceWeatherTable::to_key(int year, int doy, int hour) const {
    return static_cast<int64_t>(year - 2000) * 366 * 24 +
           static_cast<int64_t>(doy - 1) * 24 +
           static_cast<int64_t>(hour);
}

bool SpaceWeatherTable::load(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open()) {
        std::cerr << "[SpaceWeather] Failed to open: " << csv_path << "\n";
        return false;
    }

    std::string line;
    // Skip header
    std::getline(f, line);

    records_.clear();
    index_.clear();
    records_.reserve(160000);

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        SpaceWeatherRecord rec{};

        // Year,DOY,Hour,Kp,Dst,Ap,F10.7
        std::getline(ss, token, ','); rec.year = std::stoi(trim(token));
        std::getline(ss, token, ','); rec.doy  = std::stoi(trim(token));
        std::getline(ss, token, ','); rec.hour = std::stoi(trim(token));
        std::getline(ss, token, ','); rec.kp   = std::stod(trim(token));
        std::getline(ss, token, ','); rec.dst  = std::stod(trim(token));
        std::getline(ss, token, ','); rec.ap   = std::stod(trim(token));
        std::getline(ss, token, ','); rec.f107 = std::stod(trim(token));

        int64_t key = to_key(rec.year, rec.doy, rec.hour);
        index_[key] = records_.size();
        records_.push_back(rec);
    }

    std::cout << "[SpaceWeather] Loaded " << records_.size() << " records.\n";
    return !records_.empty();
}

const SpaceWeatherRecord& SpaceWeatherTable::lookup(int year, int doy, int hour) const {
    int64_t key = to_key(year, doy, hour);
    auto it = index_.find(key);
    if (it != index_.end()) return records_[it->second];

    // Nearest neighbour fallback
    auto lb = index_.lower_bound(key);
    if (lb == index_.end()) return records_.back();
    if (lb == index_.begin()) return records_[lb->second];
    auto prev = std::prev(lb);
    return (key - prev->first < lb->first - key)
               ? records_[prev->second]
               : records_[lb->second];
}

SpaceWeatherRecord SpaceWeatherTable::interpolate(double hours_since_epoch) const {
    int64_t key = static_cast<int64_t>(hours_since_epoch);
    double  frac = hours_since_epoch - key;

    auto ub = index_.upper_bound(key);
    if (ub == index_.begin() || ub == index_.end()) {
        return (ub == index_.begin()) ? records_.front() : records_.back();
    }
    auto lb = std::prev(ub);

    const auto& a = records_[lb->second];
    const auto& b = records_[ub->second];

    SpaceWeatherRecord r{};
    r.year = a.year;
    r.doy  = a.doy;
    r.hour = a.hour;
    r.kp   = a.kp   + frac * (b.kp   - a.kp);
    r.dst  = a.dst   + frac * (b.dst  - a.dst);
    r.ap   = a.ap   + frac * (b.ap   - a.ap);
    r.f107 = a.f107 + frac * (b.f107 - a.f107);
    return r;
}

// ═══════════════════════════════════════════════════════════════════
//  SAAHeatmap
// ═══════════════════════════════════════════════════════════════════

int SAAHeatmap::grid_index(int lat_idx, int lon_idx) const {
    return lat_idx * n_lon_ + lon_idx;
}

bool SAAHeatmap::load(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open()) {
        std::cerr << "[SAAHeatmap] Failed to open: " << csv_path << "\n";
        return false;
    }

    std::string line;
    std::getline(f, line); // header: Latitude,Longitude,Flux_10MeV,Flux_30MeV

    struct RawPt { double lat, lon; float f10, f30; };
    std::vector<RawPt> raw;
    raw.reserve(11000);

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string t;
        RawPt pt;
        std::getline(ss, t, ','); pt.lat = std::stod(trim(t));
        std::getline(ss, t, ','); pt.lon = std::stod(trim(t));
        std::getline(ss, t, ','); pt.f10 = std::stof(trim(t));
        std::getline(ss, t, ','); pt.f30 = std::stof(trim(t));
        raw.push_back(pt);
    }

    if (raw.empty()) return false;

    // Detect grid parameters
    lat_min_ = raw.front().lat;
    lon_min_ = raw.front().lon;
    lat_max_ = raw.front().lat;
    lon_max_ = raw.front().lon;

    // Find unique latitudes to determine step
    std::vector<double> unique_lats;
    unique_lats.push_back(raw[0].lat);
    for (size_t i = 1; i < raw.size(); ++i) {
        if (raw[i].lat != unique_lats.back()) {
            unique_lats.push_back(raw[i].lat);
        }
        lat_min_ = std::min(lat_min_, raw[i].lat);
        lat_max_ = std::max(lat_max_, raw[i].lat);
        lon_min_ = std::min(lon_min_, raw[i].lon);
        lon_max_ = std::max(lon_max_, raw[i].lon);
    }

    n_lat_ = static_cast<int>(unique_lats.size());
    lat_step_ = (n_lat_ > 1) ? (unique_lats[1] - unique_lats[0]) : 1.0;
    n_lon_ = static_cast<int>(raw.size()) / n_lat_;
    lon_step_ = (n_lon_ > 1) ? (lon_max_ - lon_min_) / (n_lon_ - 1) : 1.0;

    // Store into flat grid
    data_.resize(n_lat_ * n_lon_);
    for (size_t i = 0; i < raw.size(); ++i) {
        int li = static_cast<int>(std::round((raw[i].lat - lat_min_) / lat_step_));
        int lo = static_cast<int>(std::round((raw[i].lon - lon_min_) / lon_step_));
        li = smas::compat::clamp(li, 0, n_lat_ - 1);
        lo = smas::compat::clamp(lo, 0, n_lon_ - 1);
        data_[grid_index(li, lo)] = {raw[i].f10, raw[i].f30};
    }

    std::cout << "[SAAHeatmap] Grid " << n_lat_ << "×" << n_lon_
              << " (" << raw.size() << " points).\n";
    return true;
}

SAAFluxPoint SAAHeatmap::lookup(double lat, double lon) const {
    // Bilinear interpolation
    double fi = (lat - lat_min_) / lat_step_;
    double fj = (lon - lon_min_) / lon_step_;

    int i0 = smas::compat::clamp(static_cast<int>(std::floor(fi)), 0, n_lat_ - 2);
    int j0 = smas::compat::clamp(static_cast<int>(std::floor(fj)), 0, n_lon_ - 2);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    double di = fi - i0;
    double dj = fj - j0;

    auto lerp = [](float a, float b, double t) { return static_cast<float>(a + t * (b - a)); };

    const auto& p00 = data_[grid_index(i0, j0)];
    const auto& p01 = data_[grid_index(i0, j1)];
    const auto& p10 = data_[grid_index(i1, j0)];
    const auto& p11 = data_[grid_index(i1, j1)];

    SAAFluxPoint result;
    result.flux_10mev = lerp(lerp(p00.flux_10mev, p01.flux_10mev, dj),
                             lerp(p10.flux_10mev, p11.flux_10mev, dj), di);
    result.flux_30mev = lerp(lerp(p00.flux_30mev, p01.flux_30mev, dj),
                             lerp(p10.flux_30mev, p11.flux_30mev, dj), di);
    return result;
}

// ═══════════════════════════════════════════════════════════════════
//  GroundStationList — minimal JSON parser
// ═══════════════════════════════════════════════════════════════════

std::string GroundStationList::extract_string(const std::string& json,
                                               const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    auto q1 = json.find('"', pos + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return json.substr(q1 + 1, q2 - q1 - 1);
}

double GroundStationList::extract_number(const std::string& json,
                                          const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0.0;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return 0.0;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    size_t end = pos;
    while (end < json.size() &&
           (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+'))
        ++end;
    return std::stod(json.substr(pos, end - pos));
}

bool GroundStationList::load(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "[GroundStations] Failed to open: " << json_path << "\n";
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    stations_.clear();

    // Find each { ... } block inside "ground_stations" array
    size_t search_pos = 0;
    while (true) {
        auto brace_open = content.find('{', search_pos);
        if (brace_open == std::string::npos) break;
        // Skip the outer object brace
        if (search_pos == 0 && content.find("ground_stations") > brace_open) {
            search_pos = brace_open + 1;
            continue;
        }

        auto brace_close = content.find('}', brace_open);
        if (brace_close == std::string::npos) break;

        std::string block = content.substr(brace_open, brace_close - brace_open + 1);

        GroundStation gs;
        gs.id              = extract_string(block, "id");
        gs.name            = extract_string(block, "name");
        gs.country         = extract_string(block, "country");
        gs.latitude_deg    = extract_number(block, "latitude_deg");
        gs.longitude_deg   = extract_number(block, "longitude_deg");
        gs.altitude_m      = extract_number(block, "altitude_m");
        gs.min_elevation_deg = extract_number(block, "min_elevation_mask_deg");
        gs.role            = extract_string(block, "role");

        if (!gs.id.empty()) {
            stations_.push_back(gs);
        }

        search_pos = brace_close + 1;
    }

    std::cout << "[GroundStations] Loaded " << stations_.size() << " stations.\n";
    return !stations_.empty();
}

// ═══════════════════════════════════════════════════════════════════
//  TLE Parser
// ═══════════════════════════════════════════════════════════════════

double TLEParser::solve_kepler(double M, double e, int max_iter) {
    // Newton-Raphson iteration for Kepler's equation: M = E - e*sin(E)
    double E = M; // initial guess
    for (int i = 0; i < max_iter; ++i) {
        double dE = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= dE;
        if (std::fabs(dE) < 1e-12) break;
    }
    return E;
}

void TLEParser::elements_to_eci(const OrbitalElements& oe, Vec3& pos, Vec3& vel) {
    double a = oe.semi_major_axis_m;
    double e = oe.eccentricity;
    double i = oe.inclination_rad;
    double O = oe.raan_rad;
    double w = oe.arg_perigee_rad;
    double M = oe.mean_anomaly_rad;
    double mu = constants::EARTH_GM;

    // Solve Kepler's equation
    double E = solve_kepler(M, e);

    // True anomaly
    double sinv = std::sqrt(1.0 - e * e) * std::sin(E) / (1.0 - e * std::cos(E));
    double cosv = (std::cos(E) - e) / (1.0 - e * std::cos(E));
    double v = std::atan2(sinv, cosv);

    // Distance
    double r = a * (1.0 - e * std::cos(E));

    // Position in orbital plane
    double px = r * std::cos(v);
    double py = r * std::sin(v);

    // Velocity in orbital plane
    double h = std::sqrt(mu * a * (1.0 - e * e)); // specific angular momentum
    double vx_orb = -mu / h * std::sin(v);
    double vy_orb = mu / h * (e + std::cos(v));

    // Rotation matrices R3(-O) * R1(-i) * R3(-w)
    double cosO = std::cos(O), sinO = std::sin(O);
    double cosi = std::cos(i), sini = std::sin(i);
    double cosw = std::cos(w), sinw = std::sin(w);

    // ECI position
    pos.x = (cosO * cosw - sinO * sinw * cosi) * px +
            (-cosO * sinw - sinO * cosw * cosi) * py;
    pos.y = (sinO * cosw + cosO * sinw * cosi) * px +
            (-sinO * sinw + cosO * cosw * cosi) * py;
    pos.z = (sinw * sini) * px + (cosw * sini) * py;

    // ECI velocity
    vel.x = (cosO * cosw - sinO * sinw * cosi) * vx_orb +
            (-cosO * sinw - sinO * cosw * cosi) * vy_orb;
    vel.y = (sinO * cosw + cosO * sinw * cosi) * vx_orb +
            (-sinO * sinw + cosO * cosw * cosi) * vy_orb;
    vel.z = (sinw * sini) * vx_orb + (cosw * sini) * vy_orb;
}

bool TLEParser::load(const std::string& tle_path) {
    std::ifstream f(tle_path);
    if (!f.is_open()) {
        std::cerr << "[TLE] Failed to open: " << tle_path << "\n";
        return false;
    }

    std::string line1, line2;
    std::getline(f, line1);
    std::getline(f, line2);
    line1 = trim(line1);
    line2 = trim(line2);

    if (line1.empty() || line2.empty()) {
        std::cerr << "[TLE] Invalid TLE format.\n";
        return false;
    }

    OrbitalElements oe{};

    // ── Parse Line 1 ──
    // Cols 19-20: Epoch year (2-digit)
    std::string epoch_yr_str = line1.substr(18, 2);
    int epoch_yr_2d = std::stoi(epoch_yr_str);
    oe.epoch_year = (epoch_yr_2d < 57) ? 2000 + epoch_yr_2d : 1900 + epoch_yr_2d;

    // Cols 21-32: Epoch day (fractional)
    std::string epoch_day_str = line1.substr(20, 12);
    oe.epoch_day = std::stod(trim(epoch_day_str));

    // ── Parse Line 2 ──
    // Cols 9-16: Inclination (deg)
    oe.inclination_rad = std::stod(trim(line2.substr(8, 8))) * constants::DEG2RAD;

    // Cols 18-25: RAAN (deg)
    oe.raan_rad = std::stod(trim(line2.substr(17, 8))) * constants::DEG2RAD;

    // Cols 27-33: Eccentricity (implied leading decimal)
    std::string ecc_str = "0." + trim(line2.substr(26, 7));
    oe.eccentricity = std::stod(ecc_str);

    // Cols 35-42: Arg of Perigee (deg)
    oe.arg_perigee_rad = std::stod(trim(line2.substr(34, 8))) * constants::DEG2RAD;

    // Cols 44-51: Mean Anomaly (deg)
    oe.mean_anomaly_rad = std::stod(trim(line2.substr(43, 8))) * constants::DEG2RAD;

    // Cols 53-63: Mean Motion (rev/day)
    double mean_motion_rev_day = std::stod(trim(line2.substr(52, 11)));
    oe.mean_motion_rad_s = mean_motion_rev_day * constants::TWO_PI / 86400.0;

    // Semi-major axis from mean motion: n = sqrt(μ/a³)  ⇒  a = (μ/n²)^(1/3)
    double n = oe.mean_motion_rad_s;
    oe.semi_major_axis_m = std::cbrt(constants::EARTH_GM / (n * n));

    state_.elements = oe;

    // Convert to ECI state vector
    elements_to_eci(oe, state_.position_m, state_.velocity_ms);

    double alt = state_.position_m.magnitude() / 1000.0 - constants::EARTH_RADIUS_KM;
    std::cout << "[TLE] PROBA-1 (NORAD 26957) loaded.\n"
              << "      Epoch: " << oe.epoch_year << " day " << oe.epoch_day << "\n"
              << "      a = " << oe.semi_major_axis_m / 1000.0 << " km, "
              << "e = " << oe.eccentricity << ", "
              << "i = " << oe.inclination_rad * constants::RAD2DEG << "°\n"
              << "      Alt ≈ " << alt << " km\n";

    return true;
}

} // namespace smas
