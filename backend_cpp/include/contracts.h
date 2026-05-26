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
    uint8_t version = 4;

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

    // ── Fuel (Phase A) ──
    float   fuel_fraction;        // [0,1] remaining propellant
    uint8_t fuel_depleted;        // 0 / 1

    // ── Thermal (Phase A) ──
    float   temp_bus;             // °C
    float   temp_battery;         // °C
    float   temp_payload;         // °C
    uint8_t heater_on;            // 0 / 1

    // ── ADCS (Phase A ADCS) ──
    float   sun_angle;            // rad, [0, π]
    float   nadir_error;          // rad, [0, π]
    float   wheel_momentum_x;     // Nms
    float   wheel_momentum_y;     // Nms
    float   wheel_momentum_z;     // Nms
    
    // ── Comms & Data (Phase B Step 7) ──
    float   data_buffer_mb;       // [0, 256.0] MB (onboard flash storage)
    float   snr_db;               // Strongest ground station link SNR (dB), or -999.0 when LOS

    // ── Constellation & Debris (Task 8.4) ──
    float   conjunction_risk;     // Normalized risk metric [0, 1] based on closest debris distance
    float   time_to_tca_s;        // Time to Time of Closest Approach (TCA) in seconds
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

    // Ground Command (Developer Testbed)
    uint8_t inject_seu;   // 1 = force SEU spike this step (one-shot)
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
    ONGOING            = 0,
    BATTERY_DEAD       = 1,
    TELEMETRY_LOSS     = 2,
    REENTRY            = 3,
    SEU_FATAL          = 4,
    FUEL_DEPLETED_LOW  = 5   // out of fuel + altitude < 400 km
};

} // namespace smas
