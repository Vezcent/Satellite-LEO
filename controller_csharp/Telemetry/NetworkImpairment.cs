/*
 * S-MAS Phase 4 — Telemetry/NetworkImpairment.cs
 *
 * Simulates ground-station communication delays and packet drops
 * to realistically model LEO satellite downlink conditions.
 *
 * Config:
 *   - Random delay: uniform [minDelayMs, maxDelayMs]
 *   - Packet drop probability: [0.0, 1.0]
 */
namespace SmasController.Telemetry;

/// <summary>
/// Network impairment simulator for realistic ground-station comms.
/// </summary>
public sealed class NetworkImpairment
{
    private readonly int _minDelayMs;
    private readonly int _maxDelayMs;
    private readonly double _dropProbability;
    private readonly Random _rng;

    // Counters for diagnostics
    public int TotalFrames { get; private set; }
    public int DroppedFrames { get; private set; }
    public int DelayedFrames { get; private set; }

    /// <summary>
    /// Create a network impairment simulator.
    /// </summary>
    /// <param name="minDelayMs">Minimum communication delay in ms (default 1000).</param>
    /// <param name="maxDelayMs">Maximum communication delay in ms (default 10000).</param>
    /// <param name="dropProbability">Probability of dropping a packet (default 0.02 = 2%).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public NetworkImpairment(int minDelayMs = 1000, int maxDelayMs = 10000,
                             double dropProbability = 0.02, int seed = 42)
    {
        _minDelayMs = minDelayMs;
        _maxDelayMs = maxDelayMs;
        _dropProbability = Math.Clamp(dropProbability, 0.0, 1.0);
        _rng = new Random(seed);
    }

    /// <summary>
    /// Determine whether this frame should be dropped.
    /// </summary>
    public bool ShouldDrop()
    {
        TotalFrames++;
        if (_rng.NextDouble() < _dropProbability)
        {
            DroppedFrames++;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Get a random delay duration for this frame.
    /// </summary>
    public TimeSpan GetDelay()
    {
        int delayMs = _rng.Next(_minDelayMs, _maxDelayMs + 1);
        if (delayMs > 0) DelayedFrames++;
        return TimeSpan.FromMilliseconds(delayMs);
    }

    /// <summary>Print diagnostic summary.</summary>
    public void PrintSummary()
    {
        Console.WriteLine($"  Network Impairment: {TotalFrames} frames, " +
                          $"{DroppedFrames} dropped ({100.0 * DroppedFrames / Math.Max(1, TotalFrames):F1}%), " +
                          $"{DelayedFrames} delayed");
    }
}
