/*
 * S-MAS Phase 4 — AI/ObservationBuilder.cs
 *
 * Exact 1:1 C# port of marl_python/observation.py.
 * Converts a raw StatePacket into a normalised 29-dim float[] vector
 * suitable for ONNX model input.
 *
 * Layout (deterministic, matching Python):
 *   [0-5]   orbit:  alt, lat, lon, |v|, vx/|v|, vy/|v|
 *   [6-9]   power:  soc, cap_frac, solar_w, draw_w
 *   [10-14] env:    log_rho, log_flux10, log_flux30, eclipse, saa
 *   [15-16] comm:   gs_any, t_contact_norm
 *   [17-20] fdir:   one_hot(mode, 4)
 *   [21-23] degrad: panel_eff, cd_norm, cycles_norm
 *   [24]    seu:    seu_active
 *   [25-28] lag:    kp_3h, f107_3h, kp_6h, f107_6h (zeros for now)
 *   Total = 29
 */
using SmasController.Interop;

namespace SmasController.AI;

/// <summary>
/// Builds a normalised 29-dim observation vector from a StatePacket.
/// All normalisation logic exactly mirrors observation.py.
/// </summary>
public sealed class ObservationBuilder
{
    public const int ObsDim = 29;

    /// <summary>
    /// Build the observation vector from a raw StatePacket.
    /// </summary>
    public float[] Build(in StatePacket s)
    {
        var obs = new float[ObsDim];
        int i = 0;

        // ── 1. Orbit features (6) ─────────────────────────────────
        obs[i++] = MinMax(s.AltitudeKm, 200.0, 700.0);
        obs[i++] = MinMax(s.LatitudeDeg, -90.0, 90.0);
        obs[i++] = MinMax(s.LongitudeDeg, -180.0, 180.0);

        double vMag = Math.Sqrt(s.VelX * s.VelX + s.VelY * s.VelY + s.VelZ * s.VelZ);
        obs[i++] = MinMax(vMag, 7000.0, 8000.0);

        if (vMag > 0)
        {
            obs[i++] = (float)(s.VelX / vMag);
            obs[i++] = (float)(s.VelY / vMag);
        }
        else
        {
            obs[i++] = 0f;
            obs[i++] = 0f;
        }

        // ── 2. Power features (4) ─────────────────────────────────
        obs[i++] = (float)s.BatterySoc;  // already [0,1]
        obs[i++] = MinMax(s.BatteryCapacityJ, 0.0, 360000.0);
        obs[i++] = MinMax(s.SolarPowerW, 0.0, 100.0);
        obs[i++] = MinMax(s.PowerDrawW, 0.0, 60.0);

        // ── 3. Environment features (5) ───────────────────────────
        obs[i++] = Robust(LogSafe(s.AtmDensity), median: -10.0, iqr: 1.0);
        obs[i++] = MinMax(LogSafe(Math.Max(s.SaaFlux10Mev, 0f) + 1.0), 0.0, 5.0);
        obs[i++] = MinMax(LogSafe(Math.Max(s.SaaFlux30Mev, 0f) + 1.0), 0.0, 5.0);
        obs[i++] = s.InEclipse;
        obs[i++] = s.InSaa;

        // ── 4. Communication features (2) ─────────────────────────
        obs[i++] = s.GsVisible > 0 ? 1f : 0f;
        obs[i++] = MinMax(s.TimeSinceContactS, 0.0, 72.0 * 3600.0);

        // ── 5. FDIR one-hot (4) ───────────────────────────────────
        for (int m = 0; m < 4; m++)
            obs[i++] = s.FdirMode == m ? 1f : 0f;

        // ── 6. Degradation features (3) ───────────────────────────
        obs[i++] = (float)s.PanelEfficiency;  // [0,1]
        obs[i++] = MinMax(s.DragCoeff, 1.5, 3.0);
        obs[i++] = MinMax(s.ChargeCycles, 0.0, 50000.0);

        // ── 7. SEU (1) ────────────────────────────────────────────
        obs[i++] = s.SeuActive;

        // ── 8. Lag features (4) — placeholder zeros ───────────────
        obs[i++] = 0f;
        obs[i++] = 0f;
        obs[i++] = 0f;
        obs[i++] = 0f;

        return obs;
    }

    // ── Normalisation helpers (matching Python exactly) ────────────

    /// <summary>MinMax scale value from [lo, hi] to [0, 1], clipped.</summary>
    private static float MinMax(double val, double lo, double hi)
    {
        if (hi <= lo) return 0f;
        return (float)Math.Clamp((val - lo) / (hi - lo), 0.0, 1.0);
    }

    /// <summary>Robust scaling: (val - median) / IQR, clipped to ±5.</summary>
    private static float Robust(double val, double median, double iqr, double clipRange = 5.0)
    {
        if (iqr <= 0) return 0f;
        double scaled = (val - median) / iqr;
        return (float)Math.Clamp(scaled, -clipRange, clipRange);
    }

    /// <summary>log10 of a non-negative value, clamped to a floor.</summary>
    private static double LogSafe(double val, double floor = 1e-20)
    {
        return Math.Log10(Math.Max(val, floor));
    }
}
