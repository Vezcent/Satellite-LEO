/*
 * S-MAS: Versioned State & Action Memory-Layout Contracts.
 *
 * These packed structs define the binary interface between
 * C++ (physics), C# (controller / P/Invoke), and Python (MARL).
 * Rule: NEVER change a field without incrementing `version`.
 */
#pragma once
#include <cstdint>

namespace smas {

#pragma pack(push, 1)

// ── State Packet ──────────────────────────────────────────────────
// Direction: C++ → C# / Python
struct StatePacket {
    uint8_t version = 1;

    // ── Time ──
    double  sim_time_s;           // total elapsed seconds
    int32_t year;
    int32_t doy;
    int32_t hour;

    // ── Orbital state (ECI, metres & m/s) ──
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
    double altitude_km;
    double latitude_deg;
    double longitude_deg;

    // ── Power ──
    double battery_soc;           // [0,1]
    double battery_capacity_j;    // current max (degrades)
    double solar_power_w;
    double power_draw_w;

    // ── Environment ──
    double  atm_density;          // kg/m³
    double  drag_force_n;
    float   saa_flux_10mev;
    float   saa_flux_30mev;
    uint8_t in_eclipse;           // 0 / 1
    uint8_t in_saa;               // 0 / 1

    // ── Communication ──
    uint8_t gs_visible;           // bitmask (bit0 = Redu, bit1 = Kiruna)
    double  time_since_contact_s;

    // ── FDIR ──
    uint8_t fdir_mode;            // 0 NOM / 1 DEG / 2 SAFE / 3 REC

    // ── Degradation ──
    double   panel_efficiency;    // [0,1]
    double   drag_coeff;          // current Cd
    uint32_t charge_cycles;

    // ── Terminal ──
    uint8_t is_done;              // 0 / 1
    uint8_t done_reason;          // 0 ongoing, 1 batt, 2 telem, 3 reentry, 4 seu

    // ── SEU ──
    uint8_t seu_active;           // 0 / 1
};

// ── Action Packet ─────────────────────────────────────────────────
// Direction: C# / Python → C++
struct ActionPacket {
    uint8_t version = 1;

    // Navigation Agent
    float thrust_x;       // attitude [-1,1]
    float thrust_y;
    float thrust_z;
    float throttle;       // [0,1]

    // Resource Agent
    uint8_t deep_sleep;   // 0 / 1

    // Mission Agent
    uint8_t payload_on;   // 0 / 1
};

#pragma pack(pop)

// ── FDIR mode enum ────────────────────────────────────────────────
enum class FDIRMode : uint8_t {
    NOMINAL  = 0,
    DEGRADED = 1,
    SAFE     = 2,
    RECOVERY = 3
};

// ── Done reasons ──────────────────────────────────────────────────
enum class DoneReason : uint8_t {
    ONGOING       = 0,
    BATTERY_DEAD  = 1,
    TELEMETRY_LOSS = 2,
    REENTRY       = 3,
    SEU_FATAL     = 4
};

} // namespace smas
