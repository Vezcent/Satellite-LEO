/*
 * S-MAS: Multi-Agent System for Satellite Lifetime Optimization
 * Core mathematical types and vector operations.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

// ── Portable clamp (for compilers lacking full C++17 <algorithm>) ──
namespace smas {
namespace compat {
template<typename T>
inline T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
} // namespace compat
} // namespace smas

namespace smas {

struct Vec3 {
    double x, y, z;

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s)      const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s)      const { double inv = 1.0 / s; return {x * inv, y * inv, z * inv}; }
    Vec3 operator-()              const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s)      { x *= s; y *= s; z *= s; return *this; }

    double magnitude()    const { return std::sqrt(x * x + y * y + z * z); }
    double magnitude_sq() const { return x * x + y * y + z * z; }

    Vec3 normalized() const {
        double m = magnitude();
        return (m > 1e-15) ? (*this / m) : Vec3();
    }

    double dot(const Vec3& o)  const { return x * o.x + y * o.y + z * o.z; }
    Vec3   cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// Geodetic coordinates
struct GeoCoord {
    double latitude_deg;
    double longitude_deg;
    double altitude_km;
};

// Simulation time representation
struct SimTime {
    int32_t year;
    int32_t doy;    // day of year [1..366]
    int32_t hour;   // [0..23]
    double  total_seconds; // total elapsed simulation seconds

    // Convert to fractional hours since Year 2000, DOY 1, Hour 0
    double to_hours_since_epoch() const {
        // Approximate: doesn't account for leap years precisely, but
        // sufficient for index lookups in the hourly weather table.
        int dy = year - 2000;
        return dy * 8766.0 + (doy - 1) * 24.0 + hour;
    }
};

} // namespace smas
