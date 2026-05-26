/*
 * S-MAS Phase 4 — Telemetry/TelemetryLogger.cs
 *
 * High-performance CSV logger for simulation telemetry.
 * Writes one row per simulation step containing key state + action data
 * for post-simulation analysis and offline replay.
 *
 * Output: controller_csharp/logs/session_{timestamp}.csv
 */
using SmasController.Interop;

namespace SmasController.Telemetry;

/// <summary>
/// CSV-based telemetry logger writing one row per simulation step.
/// </summary>
public sealed class TelemetryLogger : IDisposable
{
    private readonly StreamWriter _writer;
    private bool _disposed;

    public string FilePath { get; }

    /// <summary>
    /// Create a telemetry logger.
    /// </summary>
    /// <param name="logDir">Directory for log files.</param>
    /// <param name="satId">Optional Satellite ID to append to the filename.</param>
    public TelemetryLogger(string logDir, int satId = 0)
    {
        Directory.CreateDirectory(logDir);
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        FilePath = Path.Combine(logDir, $"session_{timestamp}_sat{satId}.csv");
        _writer = new StreamWriter(FilePath, append: false);
        WriteHeader();
        Console.WriteLine($"  Telemetry logging to: {FilePath}");
    }

    private void WriteHeader()
    {
        _writer.WriteLine(
            "step,sim_time_s,altitude_km,latitude_deg,longitude_deg," +
            "battery_soc,solar_power_w,power_draw_w," +
            "in_eclipse,in_saa,fdir_mode,seu_active," +
            "gs_visible,panel_eff,drag_coeff," +
            "thrust_x,thrust_y,thrust_z,throttle,deep_sleep,payload_on," +
            "fdir_overridden,is_done,done_reason," +
            "atm_density,battery_capacity_j,charge_cycles," +
            "manual_override,seu_mult,noise_mult,drift_mult,density_mult," +
            "fuel_fraction,temp_bus,temp_battery,temp_payload,heater_on," +
            "sun_angle,nadir_error,wheel_momentum_x,wheel_momentum_y,wheel_momentum_z");
    }

    /// <summary>Log a single simulation step.</summary>
    public void LogStep(int step, in StatePacket state, in ActionPacket action, bool fdirOverridden,
                        bool manualOverride = false,
                        double seuMult = 1.0, double noiseMult = 1.0,
                        double driftMult = 1.0, double densityMult = 0.01)
    {
        _writer.Write(step);
        _writer.Write(','); _writer.Write(state.SimTimeS);
        _writer.Write(','); _writer.Write(state.AltitudeKm);
        _writer.Write(','); _writer.Write(state.LatitudeDeg);
        _writer.Write(','); _writer.Write(state.LongitudeDeg);
        _writer.Write(','); _writer.Write(state.BatterySoc);
        _writer.Write(','); _writer.Write(state.SolarPowerW);
        _writer.Write(','); _writer.Write(state.PowerDrawW);
        _writer.Write(','); _writer.Write(state.InEclipse);
        _writer.Write(','); _writer.Write(state.InSaa);
        _writer.Write(','); _writer.Write(state.FdirMode);
        _writer.Write(','); _writer.Write(state.SeuActive);
        _writer.Write(','); _writer.Write(state.GsVisible);
        _writer.Write(','); _writer.Write(state.PanelEfficiency);
        _writer.Write(','); _writer.Write(state.DragCoeff);
        _writer.Write(','); _writer.Write(action.ThrustX);
        _writer.Write(','); _writer.Write(action.ThrustY);
        _writer.Write(','); _writer.Write(action.ThrustZ);
        _writer.Write(','); _writer.Write(action.Throttle);
        _writer.Write(','); _writer.Write(action.DeepSleep);
        _writer.Write(','); _writer.Write(action.PayloadOn);
        _writer.Write(','); _writer.Write(fdirOverridden ? 1 : 0);
        _writer.Write(','); _writer.Write(state.IsDone);
        _writer.Write(','); _writer.Write(state.DoneReasonVal);
        // New columns for degradation analysis & environment config
        _writer.Write(','); _writer.Write(state.AtmDensity);
        _writer.Write(','); _writer.Write(state.BatteryCapacityJ);
        _writer.Write(','); _writer.Write(state.ChargeCycles);
        _writer.Write(','); _writer.Write(manualOverride ? 1 : 0);
        _writer.Write(','); _writer.Write(seuMult);
        _writer.Write(','); _writer.Write(noiseMult);
        _writer.Write(','); _writer.Write(driftMult);
        _writer.Write(','); _writer.Write(densityMult);
        // Phase A Fuel and Thermal fields
        _writer.Write(','); _writer.Write(state.FuelFraction.ToString("F4"));
        _writer.Write(','); _writer.Write(state.TempBus.ToString("F2"));
        _writer.Write(','); _writer.Write(state.TempBattery.ToString("F2"));
        _writer.Write(','); _writer.Write(state.TempPayload.ToString("F2"));
        _writer.Write(','); _writer.Write(state.HeaterOn);
        // Phase A ADCS fields
        _writer.Write(','); _writer.Write(state.SunAngle.ToString("F4"));
        _writer.Write(','); _writer.Write(state.NadirError.ToString("F4"));
        _writer.Write(','); _writer.Write(state.WheelMomentumX.ToString("F4"));
        _writer.Write(','); _writer.Write(state.WheelMomentumY.ToString("F4"));
        _writer.Write(','); _writer.Write(state.WheelMomentumZ.ToString("F4"));
        _writer.WriteLine();
    }

    /// <summary>Flush buffered data to disk.</summary>
    public void Flush() => _writer.Flush();

    public void Dispose()
    {
        if (!_disposed)
        {
            _writer.Flush();
            _writer.Dispose();
            _disposed = true;
        }
    }
}
