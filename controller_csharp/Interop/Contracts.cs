/*
 * S-MAS Phase 4 — Interop/Contracts.cs
 * 
 * C# mirrors of the packed C++ structs from contracts.h.
 * MUST stay byte-identical to smas::StatePacket (184B) and smas::ActionPacket (19B).
 *
 * Layout verified against backend_cpp/include/contracts.h version 1.
 */
using System.Runtime.InteropServices;

namespace SmasController.Interop;

// ═══════════════════════════════════════════════════════════════════
//  FDIR Mode Enum
// ═══════════════════════════════════════════════════════════════════

public enum FdirMode : byte
{
    Nominal  = 0,
    Degraded = 1,
    Safe     = 2,
    Recovery = 3
}

// ═══════════════════════════════════════════════════════════════════
//  Done Reason Enum
// ═══════════════════════════════════════════════════════════════════

public enum DoneReason : byte
{
    Ongoing       = 0,
    BatteryDead   = 1,
    TelemetryLoss = 2,
    Reentry       = 3,
    SeuFatal      = 4
}

// ═══════════════════════════════════════════════════════════════════
//  StatePacket — C++ → C# (184 bytes, version 1)
// ═══════════════════════════════════════════════════════════════════

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct StatePacket
{
    public byte Version;              // offset 0

    // ── Time ──
    public double SimTimeS;           // 1
    public int    Year;               // 9
    public int    Doy;                // 13
    public int    Hour;               // 17

    // ── Orbital state (ECI, metres & m/s) ──
    public double PosX;               // 21
    public double PosY;               // 29
    public double PosZ;               // 37
    public double VelX;               // 45
    public double VelY;               // 53
    public double VelZ;               // 61
    public double AltitudeKm;         // 69
    public double LatitudeDeg;        // 77
    public double LongitudeDeg;       // 85

    // ── Power ──
    public double BatterySoc;         // 93
    public double BatteryCapacityJ;   // 101
    public double SolarPowerW;        // 109
    public double PowerDrawW;         // 117

    // ── Environment ──
    public double AtmDensity;         // 125
    public double DragForceN;         // 133
    public float  SaaFlux10Mev;       // 141
    public float  SaaFlux30Mev;       // 145
    public byte   InEclipse;          // 149
    public byte   InSaa;              // 150

    // ── Communication ──
    public byte   GsVisible;          // 151
    public double TimeSinceContactS;  // 152

    // ── FDIR ──
    public byte   FdirMode;           // 160

    // ── Degradation ──
    public double PanelEfficiency;    // 161
    public double DragCoeff;          // 169
    public uint   ChargeCycles;       // 177

    // ── Terminal ──
    public byte   IsDone;             // 181
    public byte   DoneReasonVal;      // 182

    // ── SEU ──
    public byte   SeuActive;          // 183

    // Total: 184 bytes

    /// <summary>Typed accessor for the FDIR mode.</summary>
    public readonly FdirMode FdirModeEnum => (FdirMode)FdirMode;

    /// <summary>Typed accessor for the done reason.</summary>
    public readonly DoneReason DoneReasonEnum => (DoneReason)DoneReasonVal;
}

// ═══════════════════════════════════════════════════════════════════
//  ActionPacket — C# → C++ (19 bytes, version 1)
// ═══════════════════════════════════════════════════════════════════

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct ActionPacket
{
    public byte  Version;     // 0

    // Navigation Agent
    public float ThrustX;     // 1
    public float ThrustY;     // 5
    public float ThrustZ;     // 9
    public float Throttle;    // 13

    // Resource Agent
    public byte  DeepSleep;   // 17

    // Mission Agent
    public byte  PayloadOn;   // 18

    // Total: 19 bytes

    /// <summary>Create a zeroed action with correct version.</summary>
    public static ActionPacket CreateNoOp()
    {
        return new ActionPacket { Version = 1 };
    }
}
