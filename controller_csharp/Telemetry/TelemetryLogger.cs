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
    public TelemetryLogger(string logDir)
    {
        Directory.CreateDirectory(logDir);
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        FilePath = Path.Combine(logDir, $"session_{timestamp}.csv");
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
            "fdir_overridden,is_done,done_reason");
    }

    /// <summary>Log a single simulation step.</summary>
    public void LogStep(int step, in StatePacket state, in ActionPacket action, bool fdirOverridden)
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
