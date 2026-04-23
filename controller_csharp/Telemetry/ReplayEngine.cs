/*
 * S-MAS Phase 4 — Telemetry/ReplayEngine.cs
 *
 * Offline replay system that reads a telemetry log CSV file
 * and streams frames via WebSocket at configurable playback speed.
 * Bypasses the C++ physics engine entirely for post-mission analysis.
 *
 * Usage:
 *   dotnet run -- --replay logs/session_20260423_103000.csv --speed 10.0
 */
namespace SmasController.Telemetry;

/// <summary>
/// Offline replay engine that streams historical telemetry logs
/// through the WebSocket server at configurable speed.
/// </summary>
public sealed class ReplayEngine
{
    private readonly WebSocketServer _wsServer;

    /// <summary>
    /// Create a replay engine.
    /// </summary>
    /// <param name="wsServer">WebSocket server for broadcasting.</param>
    public ReplayEngine(WebSocketServer wsServer)
    {
        _wsServer = wsServer;
    }

    /// <summary>
    /// Replay a telemetry log file through the WebSocket server.
    /// </summary>
    /// <param name="logPath">Path to the CSV log file.</param>
    /// <param name="speedMultiplier">Playback speed (1.0 = realtime at dt=5s, 10.0 = 10x).</param>
    /// <param name="ct">Cancellation token for stopping playback.</param>
    public async Task PlayAsync(string logPath, double speedMultiplier = 1.0,
                                CancellationToken ct = default)
    {
        if (!File.Exists(logPath))
            throw new FileNotFoundException($"Replay log not found: {logPath}");

        Console.WriteLine($"  Replay mode: {logPath} at {speedMultiplier}x speed");
        Console.WriteLine("  Waiting for WebSocket client...");

        // Wait until at least one client connects
        while (_wsServer.ConnectedClients == 0 && !ct.IsCancellationRequested)
            await Task.Delay(500, ct);

        using var reader = new StreamReader(logPath);
        string? header = await reader.ReadLineAsync();  // skip CSV header
        if (header == null)
        {
            Console.WriteLine("  [Replay] Empty log file.");
            return;
        }

        // Time between frames based on dt=5.0s and speed multiplier
        int frameDelayMs = (int)(5000.0 / Math.Max(speedMultiplier, 0.01));
        uint seq = 0;

        Console.WriteLine($"  [Replay] Frame interval: {frameDelayMs}ms ({speedMultiplier}x)");

        while (!ct.IsCancellationRequested)
        {
            string? line = await reader.ReadLineAsync();
            if (line == null)
            {
                Console.WriteLine("  [Replay] End of log file reached.");
                break;
            }

            // Parse CSV row and build a minimal binary packet
            byte[] packet = BuildReplayPacket(seq++, line);
            _wsServer.EnqueueFrame(packet);
            await _wsServer.FlushAsync(ct);
            await Task.Delay(frameDelayMs, ct);
        }

        Console.WriteLine($"  [Replay] Complete. {seq} frames streamed.");
    }

    /// <summary>
    /// Build a binary replay packet from a CSV row.
    /// Uses the same packet format as live telemetry for frontend compatibility.
    /// </summary>
    private static byte[] BuildReplayPacket(uint seq, string csvLine)
    {
        var parts = csvLine.Split(',');

        // Parse CSV fields in order matching TelemetryLogger header
        using var ms = new MemoryStream(128);
        using var bw = new BinaryWriter(ms);

        // Write version + seq + placeholder length
        bw.Write(TelemetryPacket.PacketVersion);
        bw.Write(seq);

        // Payload start
        int payloadStart = (int)ms.Position;
        bw.Write((uint)0);  // payload length placeholder

        // Write fields — if parsing fails, use defaults
        bw.Write(ParseDouble(parts, 1));   // sim_time_s
        bw.Write(ParseDouble(parts, 2));   // altitude_km
        bw.Write(ParseDouble(parts, 3));   // latitude_deg
        bw.Write(ParseDouble(parts, 4));   // longitude_deg
        bw.Write(ParseDouble(parts, 5));   // battery_soc
        bw.Write(ParseDouble(parts, 6));   // solar_power_w
        bw.Write(ParseDouble(parts, 7));   // power_draw_w
        bw.Write(ParseByte(parts, 8));     // in_eclipse
        bw.Write(ParseByte(parts, 9));     // in_saa
        bw.Write(ParseByte(parts, 10));    // fdir_mode
        bw.Write(ParseByte(parts, 11));    // seu_active
        bw.Write(ParseByte(parts, 12));    // gs_visible
        bw.Write(ParseDouble(parts, 13));  // panel_eff
        bw.Write(ParseDouble(parts, 14));  // drag_coeff
        bw.Write(ParseByte(parts, 22));    // is_done
        bw.Write(ParseByte(parts, 23));    // done_reason
        // Actions
        bw.Write(ParseFloat(parts, 15));   // thrust_x
        bw.Write(ParseFloat(parts, 16));   // thrust_y
        bw.Write(ParseFloat(parts, 17));   // thrust_z
        bw.Write(ParseFloat(parts, 18));   // throttle
        bw.Write(ParseByte(parts, 19));    // deep_sleep
        bw.Write(ParseByte(parts, 20));    // payload_on
        bw.Write(ParseByte(parts, 21));    // fdir_overridden

        bw.Flush();

        // Patch payload length
        byte[] data = ms.ToArray();
        int payloadLen = data.Length - payloadStart - 4;
        BitConverter.GetBytes((uint)payloadLen).CopyTo(data, payloadStart);

        return data;
    }

    private static double ParseDouble(string[] parts, int idx) =>
        idx < parts.Length && double.TryParse(parts[idx], out var v) ? v : 0.0;

    private static float ParseFloat(string[] parts, int idx) =>
        idx < parts.Length && float.TryParse(parts[idx], out var v) ? v : 0f;

    private static byte ParseByte(string[] parts, int idx) =>
        idx < parts.Length && byte.TryParse(parts[idx], out var v) ? v : (byte)0;
}
