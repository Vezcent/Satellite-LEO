/*
 * S-MAS Phase 4 — AI/InferenceEngine.cs
 *
 * ONNX Runtime session manager for the 3 agent heads.
 * Loads smas_nav.onnx, smas_bus.onnx, smas_mission.onnx
 * and provides inference methods returning typed action outputs.
 *
 * Input:  float[30] normalised observation vector
 * Output: NavigationAction (mu[4]), bus decision (0/1), mission decision (0/1)
 */
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmasController.AI;

/// <summary>
/// Navigation agent inference result.
/// mu[4] = [thrust_x, thrust_y, thrust_z, throttle].
/// At inference time we use the mean (deterministic policy).
/// </summary>
public readonly struct NavigationAction
{
    public float ThrustX { get; init; }
    public float ThrustY { get; init; }
    public float ThrustZ { get; init; }
    public float Throttle { get; init; }
}

/// <summary>
/// Combined inference result from all 3 agent heads.
/// </summary>
public readonly struct AgentActions
{
    public NavigationAction Nav { get; init; }
    public byte DeepSleep { get; init; }
    public byte PayloadOn { get; init; }
}

/// <summary>
/// Manages ONNX Runtime sessions for all 3 agent heads.
/// Designed for single-agent inference with batch support ready.
/// </summary>
public sealed class InferenceEngine : IDisposable
{
    private readonly InferenceSession _navSession;
    private readonly InferenceSession _busSession;
    private readonly InferenceSession _missionSession;
    private bool _disposed;

    /// <summary>
    /// Initialise ONNX sessions from model files.
    /// </summary>
    /// <param name="modelDir">Directory containing smas_nav.onnx, smas_bus.onnx, smas_mission.onnx</param>
    public InferenceEngine(string modelDir)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        string navPath     = Path.Combine(modelDir, "smas_nav.onnx");
        string busPath     = Path.Combine(modelDir, "smas_bus.onnx");
        string missionPath = Path.Combine(modelDir, "smas_mission.onnx");

        if (!File.Exists(navPath))
            throw new FileNotFoundException($"Nav model not found: {navPath}");
        if (!File.Exists(busPath))
            throw new FileNotFoundException($"Bus model not found: {busPath}");
        if (!File.Exists(missionPath))
            throw new FileNotFoundException($"Mission model not found: {missionPath}");

        _navSession     = new InferenceSession(navPath, opts);
        _busSession     = new InferenceSession(busPath, opts);
        _missionSession = new InferenceSession(missionPath, opts);

        Console.WriteLine($"  ONNX sessions loaded from: {modelDir}");
        Console.WriteLine($"    Nav model:     {Path.GetFileName(navPath)}");
        Console.WriteLine($"    Bus model:     {Path.GetFileName(busPath)}");
        Console.WriteLine($"    Mission model: {Path.GetFileName(missionPath)}");
    }

    /// <summary>
    /// Run inference on all 3 agent heads for a single observation.
    /// </summary>
    /// <param name="obs">Normalised 30-dim observation vector.</param>
    /// <returns>Combined actions from all agents.</returns>
    public AgentActions Infer(float[] obs)
    {
        // Create 2D tensor [1, 29] — single agent, batch_size = 1
        var tensor = new DenseTensor<float>(obs, [1, obs.Length]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("obs_input", tensor)
        };

        // ── Navigation Head ──────────────────────────────────────
        // Output: action[4] — tanh-squashed, deterministic policy
        using var navResults = _navSession.Run(inputs);
        var actionTensor = navResults.First().AsTensor<float>();

        // action[0..2] are thrust in [-1, 1] (already tanh-squashed)
        // action[3] is throttle: tanh gives [-1,1], map to [0,1]
        var nav = new NavigationAction
        {
            ThrustX  = actionTensor[0, 0],
            ThrustY  = actionTensor[0, 1],
            ThrustZ  = actionTensor[0, 2],
            Throttle = (actionTensor[0, 3] + 1f) / 2f,  // [-1,1] → [0,1]
        };

        // ── Resource Head ────────────────────────────────────────
        // Output: logit[1] — sigmoid > 0.5 → deep_sleep
        using var busResults = _busSession.Run(inputs);
        var busLogit = busResults.First().AsTensor<float>();
        byte deepSleep = Sigmoid(busLogit[0]) > 0.5f ? (byte)1 : (byte)0;

        // ── Mission Head ─────────────────────────────────────────
        // Output: logit[1] — sigmoid > 0.5 → payload_on
        using var missionResults = _missionSession.Run(inputs);
        var missionLogit = missionResults.First().AsTensor<float>();
        byte payloadOn = Sigmoid(missionLogit[0]) > 0.5f ? (byte)1 : (byte)0;

        return new AgentActions
        {
            Nav       = nav,
            DeepSleep = deepSleep,
            PayloadOn = payloadOn,
        };
    }

    /// <summary>Standard sigmoid activation function.</summary>
    private static float Sigmoid(float x)
    {
        return 1f / (1f + MathF.Exp(-x));
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _navSession.Dispose();
            _busSession.Dispose();
            _missionSession.Dispose();
            _disposed = true;
        }
    }
}
