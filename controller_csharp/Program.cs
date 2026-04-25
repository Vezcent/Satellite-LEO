/*
 * S-MAS Phase 4 — Program.cs
 *
 * Main orchestrator for the Controller & Operations Simulation.
 * Wires together: C++ Physics Engine (P/Invoke) → Observation Builder →
 * ONNX Inference (3 agents) → FDIR Governor → WebSocket Telemetry.
 *
 * Usage:
 *   dotnet run                                              # defaults
 *   dotnet run -- --data-dir "../preprocessed-data"         # custom data
 *   dotnet run -- --steps 17280                             # 1 day
 *   dotnet run -- --skip 100                                # fast-forward 100x
 *   dotnet run -- --replay logs/session_XYZ.csv --speed 10  # replay mode
 *   dotnet run -- --test                                    # integration test
 *
 * CLI Arguments:
 *   --data-dir   Path to preprocessed-data (default: ../preprocessed-data)
 *   --model-dir  Path to ONNX models (default: models)
 *   --steps      Max simulation steps (default: 17280 = ~1 day)
 *   --seed       Random seed (default: 42)
 *   --port       WebSocket port (default: 8765)
 *   --skip       Fast-forward: simulate N steps per WS frame (default: 1)
 *   --no-ws      Disable WebSocket server
 *   --replay     Path to CSV log for replay mode
 *   --speed      Replay speed multiplier (default: 1.0)
 *   --test       Run integration tests then exit
 */
using System.Runtime.InteropServices;
using SmasController.AI;
using SmasController.Governor;
using SmasController.Interop;
using SmasController.Telemetry;

namespace SmasController;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("══════════════════════════════════════════════════════════════");
        Console.WriteLine("  S-MAS Controller — Phase 4: Operations Simulation");
        Console.WriteLine("══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        // ── Parse CLI arguments ──────────────────────────────────
        var config = ParseArgs(args);

        if (config.RunTest)
            return RunIntegrationTests(config);

        if (config.ReplayPath != null)
            return await RunReplayMode(config);

        return await RunSimulation(config);
    }

    // ═════════════════════════════════════════════════════════════
    //  SIMULATION MODE
    // ═════════════════════════════════════════════════════════════

    private static async Task<int> RunSimulation(Config config)
    {
        Console.WriteLine("  Mode: LIVE SIMULATION");
        Console.WriteLine($"  Data dir:  {config.DataDir}");
        Console.WriteLine($"  Model dir: {config.ModelDir}");
        Console.WriteLine($"  Steps:     {config.MaxSteps}");
        Console.WriteLine($"  Skip:      {config.Skip}x (broadcast every {config.Skip} steps)");
        Console.WriteLine($"  Seed:      {config.Seed}");
        Console.WriteLine($"  WebSocket: {(config.NoWebSocket ? "DISABLED" : $"ws://localhost:{config.Port}/")}");
        Console.WriteLine();

        // ── 1. Create + Init Physics Engine ──────────────────────
        using var engine = new PhysicsEngine(config.DataDir, config.Seed);
        engine.ValidateAbi();
        engine.Init();
        Console.WriteLine("  C++ physics engine initialised ✓");

        // ── 2. Load ONNX Models ──────────────────────────────────
        using var inference = new InferenceEngine(config.ModelDir);
        Console.WriteLine();

        // ── 3. Create Components ─────────────────────────────────
        var obsBuilder = new ObservationBuilder();
        var governor = new FdirGovernor();
        var logDir = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".", "logs");
        using var logger = new TelemetryLogger(logDir);

        // ── 4. WebSocket Server (optional) ───────────────────────
        WebSocketServer? wsServer = null;
        NetworkImpairment? impairment = null;
        if (!config.NoWebSocket)
        {
            impairment = new NetworkImpairment(
                minDelayMs: 100, maxDelayMs: 500,   // reduced for sim speed
                dropProbability: 0.02, seed: (int)config.Seed);
            wsServer = new WebSocketServer(config.Port, impairment);
            wsServer.Start();
        }

        // ── 5. Reset Engine ──────────────────────────────────────
        engine.Reset();
        Console.WriteLine();
        Console.WriteLine("  Starting simulation loop...");
        Console.WriteLine("  ────────────────────────────────────────────────────────");

        var state = new StatePacket();
        var action = ActionPacket.CreateNoOp();

        // Initial step to get state
        engine.Step(ref action, ref state);

        int step = 0;
        int fdirOverrides = 0;
        int payloadOnSteps = 0;
        int eclipseSteps = 0;
        int saaSteps = 0;

        // ── 6. Main Simulation Loop ──────────────────────────────
        while (step < config.MaxSteps && state.IsDone == 0)
        {
            // a. Build normalised observation
            float[] obs = obsBuilder.Build(in state);

            // b. Run ONNX inference
            AgentActions aiActions = inference.Infer(obs);

            // c. Apply FDIR Governor overrides
            action = governor.Apply(aiActions, in state, out bool overridden);
            if (overridden) fdirOverrides++;

            // d. Step the engine
            engine.Step(ref action, ref state);
            step++;

            // e. Track metrics
            if (action.PayloadOn == 1) payloadOnSteps++;
            if (state.InEclipse == 1) eclipseSteps++;
            if (state.InSaa == 1) saaSteps++;

            // f. Log + broadcast only on skip boundary (fast-forward)
            bool isBroadcastStep = (step % config.Skip == 0) || state.IsDone == 1;

            if (isBroadcastStep)
            {
                // Log telemetry
                logger.LogStep(step, in state, in action, overridden);

                // Broadcast via WebSocket
                if (wsServer != null)
                {
                    byte[] packet = TelemetryPacket.Serialise(
                        (uint)step, in state, in action, overridden);
                    wsServer.EnqueueFrame(packet);
                    await wsServer.FlushAsync();
                }
            }

            // h. Periodic console output
            int consoleInterval = Math.Max(1000, config.Skip * 100);
            if (step % consoleInterval == 0 || state.IsDone == 1)
            {
                double simHours = state.SimTimeS / 3600.0;
                Console.WriteLine(
                    $"  Step {step,6} | {simHours,6:F1}h | Alt={state.AltitudeKm,7:F1}km " +
                    $"| SoC={state.BatterySoc * 100,5:F1}% | FDIR={FdirGovernor.ModeLabel(state.FdirMode)} " +
                    $"| Eclipse={state.InEclipse} | SAA={state.InSaa} | Payload={action.PayloadOn}");
            }
        }

        // ── 7. Episode Summary ───────────────────────────────────
        logger.Flush();
        Console.WriteLine();
        Console.WriteLine("  ════════════════════════════════════════════════════════");
        Console.WriteLine("  EPISODE SUMMARY");
        Console.WriteLine("  ════════════════════════════════════════════════════════");
        Console.WriteLine($"  Total steps:      {step}");
        Console.WriteLine($"  Simulation time:  {state.SimTimeS / 3600.0:F1} hours ({state.SimTimeS / 86400.0:F2} days)");
        Console.WriteLine($"  Final altitude:   {state.AltitudeKm:F2} km");
        Console.WriteLine($"  Final SoC:        {state.BatterySoc * 100:F2}%");
        Console.WriteLine($"  Final FDIR mode:  {FdirGovernor.ModeLabel(state.FdirMode)}");
        Console.WriteLine($"  Panel efficiency: {state.PanelEfficiency * 100:F2}%");
        Console.WriteLine($"  Drag coefficient: {state.DragCoeff:F4}");
        Console.WriteLine($"  Charge cycles:    {state.ChargeCycles}");
        Console.WriteLine($"  Episode done:     {(state.IsDone == 1 ? "YES" : "NO")}");
        if (state.IsDone == 1)
            Console.WriteLine($"  Done reason:      {state.DoneReasonEnum}");
        Console.WriteLine();
        Console.WriteLine($"  FDIR overrides:   {fdirOverrides} ({100.0 * fdirOverrides / Math.Max(1, step):F1}%)");
        Console.WriteLine($"  Payload ON steps: {payloadOnSteps} ({100.0 * payloadOnSteps / Math.Max(1, step):F1}%)");
        Console.WriteLine($"  Eclipse steps:    {eclipseSteps} ({100.0 * eclipseSteps / Math.Max(1, step):F1}%)");
        Console.WriteLine($"  SAA steps:        {saaSteps} ({100.0 * saaSteps / Math.Max(1, step):F1}%)");
        Console.WriteLine($"  Telemetry log:    {logger.FilePath}");

        if (impairment != null)
            impairment.PrintSummary();

        Console.WriteLine("  ════════════════════════════════════════════════════════");

        // ── 8. Cleanup ───────────────────────────────────────────
        wsServer?.Dispose();
        return 0;
    }

    // ═════════════════════════════════════════════════════════════
    //  REPLAY MODE
    // ═════════════════════════════════════════════════════════════

    private static async Task<int> RunReplayMode(Config config)
    {
        Console.WriteLine("  Mode: OFFLINE REPLAY");
        Console.WriteLine($"  Log file:  {config.ReplayPath}");
        Console.WriteLine($"  Speed:     {config.ReplaySpeed}x");
        Console.WriteLine();

        using var wsServer = new WebSocketServer(config.Port);
        wsServer.Start();

        var replay = new ReplayEngine(wsServer);
        using var cts = new CancellationTokenSource();

        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        try
        {
            await replay.PlayAsync(config.ReplayPath!, config.ReplaySpeed, cts.Token);
        }
        catch (OperationCanceledException) { /* graceful shutdown */ }

        return 0;
    }

    // ═════════════════════════════════════════════════════════════
    //  INTEGRATION TESTS
    // ═════════════════════════════════════════════════════════════

    private static int RunIntegrationTests(Config config)
    {
        Console.WriteLine("  Mode: INTEGRATION TEST");
        Console.WriteLine();

        int passed = 0;
        int total = 0;

        // ── Test 1: ABI Check ────────────────────────────────────
        total++;
        try
        {
            int csState = Marshal.SizeOf<StatePacket>();
            int csAction = Marshal.SizeOf<ActionPacket>();
            Assert(csState == 184, $"StatePacket size: expected 184, got {csState}");
            Assert(csAction == 19, $"ActionPacket size: expected 19, got {csAction}");
            Pass(1, "ABI struct sizes", $"State={csState}B, Action={csAction}B");
            passed++;
        }
        catch (Exception ex) { Fail(1, "ABI struct sizes", ex.Message); }

        // ── Test 2: DLL Load ─────────────────────────────────────
        total++;
        PhysicsEngine? engine = null;
        try
        {
            engine = new PhysicsEngine(config.DataDir, config.Seed);
            engine.ValidateAbi();
            engine.Init();
            Pass(2, "DLL load + ABI validation", "smas_engine.dll loaded");
            passed++;
        }
        catch (Exception ex) { Fail(2, "DLL load + ABI validation", ex.Message); }

        // ── Test 3: Reset + Initial State ────────────────────────
        total++;
        var state = new StatePacket();
        try
        {
            engine!.Reset();
            var action = ActionPacket.CreateNoOp();
            engine.Step(ref action, ref state);
            Assert(state.AltitudeKm > 500 && state.AltitudeKm < 650,
                   $"Alt={state.AltitudeKm:F1} not in [500,650]");
            Assert(state.BatterySoc > 0.9, $"SoC={state.BatterySoc:F2} unexpectedly low");
            Pass(3, "Reset + initial state", $"Alt={state.AltitudeKm:F1}km, SoC={state.BatterySoc * 100:F1}%");
            passed++;
        }
        catch (Exception ex) { Fail(3, "Reset + initial state", ex.Message); }

        // ── Test 4: Observation Builder ──────────────────────────
        total++;
        float[] obs;
        try
        {
            var builder = new ObservationBuilder();
            obs = builder.Build(in state);
            Assert(obs.Length == 30, $"Obs dim: expected 30, got {obs.Length}");
            // Check no NaN/Inf
            for (int i = 0; i < obs.Length; i++)
                Assert(!float.IsNaN(obs[i]) && !float.IsInfinity(obs[i]),
                       $"Obs[{i}] is NaN or Inf");
            Pass(4, "Observation builder", $"dim={obs.Length}, range=[{obs.Min():F3}, {obs.Max():F3}]");
            passed++;
        }
        catch (Exception ex) { Fail(4, "Observation builder", ex.Message); obs = new float[30]; }

        // ── Test 5: ONNX Session Load ────────────────────────────
        total++;
        InferenceEngine? inference = null;
        try
        {
            inference = new InferenceEngine(config.ModelDir);
            Pass(5, "ONNX session load", "3 sessions loaded");
            passed++;
        }
        catch (Exception ex) { Fail(5, "ONNX session load", ex.Message); }

        // ── Test 6: ONNX Inference ───────────────────────────────
        total++;
        try
        {
            var actions = inference!.Infer(obs);
            Assert(actions.Nav.ThrustX >= -1f && actions.Nav.ThrustX <= 1f,
                   $"Nav ThrustX={actions.Nav.ThrustX} out of range");
            Assert(actions.Nav.Throttle >= 0f && actions.Nav.Throttle <= 1f,
                   $"Nav Throttle={actions.Nav.Throttle} out of range");
            Assert(actions.DeepSleep <= 1, $"DeepSleep={actions.DeepSleep} invalid");
            Assert(actions.PayloadOn <= 1, $"PayloadOn={actions.PayloadOn} invalid");
            Pass(6, "ONNX inference", $"nav=[{actions.Nav.ThrustX:F3},{actions.Nav.ThrustY:F3}," +
                 $"{actions.Nav.ThrustZ:F3},{actions.Nav.Throttle:F3}], bus={actions.DeepSleep}, mission={actions.PayloadOn}");
            passed++;
        }
        catch (Exception ex) { Fail(6, "ONNX inference", ex.Message); }

        // ── Test 7: FDIR Governor (SAFE mode override) ───────────
        total++;
        try
        {
            var gov = new FdirGovernor();
            // Simulate SAFE mode
            var safeState = state;
            safeState.FdirMode = 2;  // SAFE
            var testActions = new AgentActions
            {
                Nav = new NavigationAction { ThrustX = 1f, ThrustY = 1f, ThrustZ = 1f, Throttle = 1f },
                DeepSleep = 0,
                PayloadOn = 1
            };
            var result = gov.Apply(testActions, in safeState, out bool overridden);
            Assert(overridden, "SAFE mode should override");
            Assert(result.ThrustX == 0f && result.Throttle == 0f, "Thrust should be zeroed in SAFE");
            Assert(result.DeepSleep == 1, "DeepSleep should be forced ON in SAFE");
            Assert(result.PayloadOn == 0, "Payload should be OFF in SAFE");
            Pass(7, "FDIR Governor (SAFE)", "All actions overridden correctly");
            passed++;
        }
        catch (Exception ex) { Fail(7, "FDIR Governor (SAFE)", ex.Message); }

        // ── Test 8: Meta-Coordination ────────────────────────────
        total++;
        try
        {
            var gov = new FdirGovernor();
            var nominalState = state;
            nominalState.FdirMode = 0;  // NOMINAL
            var testActions = new AgentActions
            {
                Nav = new NavigationAction(),
                DeepSleep = 1,    // bus wants sleep
                PayloadOn = 1     // mission wants payload
            };
            var result = gov.Apply(testActions, in nominalState, out bool overridden);
            Assert(result.PayloadOn == 0, "Meta-coord: payload should be OFF when deep_sleep=1");
            Assert(overridden, "Meta-coord should flag override");
            Pass(8, "Meta-coordination", "deep_sleep=1 → payload forced OFF");
            passed++;
        }
        catch (Exception ex) { Fail(8, "Meta-coordination", ex.Message); }

        // ── Test 9: 50-Step Simulation Loop ──────────────────────
        total++;
        try
        {
            engine!.Reset();
            var builder = new ObservationBuilder();
            var gov = new FdirGovernor();
            var action = ActionPacket.CreateNoOp();
            engine.Step(ref action, ref state);

            for (int i = 0; i < 50; i++)
            {
                obs = builder.Build(in state);
                var ai = inference!.Infer(obs);
                action = gov.Apply(ai, in state, out _);
                engine.Step(ref action, ref state);
            }
            Assert(state.AltitudeKm > 400, $"Alt={state.AltitudeKm:F1} too low after 50 steps");
            Assert(state.BatterySoc > 0.5, $"SoC={state.BatterySoc:F2} too low after 50 steps");
            Pass(9, "50-step simulation", $"Alt={state.AltitudeKm:F1}km, SoC={state.BatterySoc * 100:F1}%");
            passed++;
        }
        catch (Exception ex) { Fail(9, "50-step simulation", ex.Message); }

        // ── Test 10: Telemetry Logger ────────────────────────────
        total++;
        try
        {
            var logDir = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".", "logs_test");
            string logFilePath;
            {
                using var testLogger = new TelemetryLogger(logDir);
                logFilePath = testLogger.FilePath;
                var action = ActionPacket.CreateNoOp();
                testLogger.LogStep(1, in state, in action, false);
                testLogger.Flush();
            } // logger disposed here — file handle released
            Assert(File.Exists(logFilePath), "Log file not created");
            var lines = File.ReadAllLines(logFilePath);
            Assert(lines.Length == 2, $"Expected 2 lines (header+data), got {lines.Length}");
            Pass(10, "Telemetry logger", $"Written to {Path.GetFileName(logFilePath)}");
            passed++;
        }
        catch (Exception ex) { Fail(10, "Telemetry logger", ex.Message); }

        // ── Summary ──────────────────────────────────────────────
        Console.WriteLine();
        Console.WriteLine($"  ══════════════════════════════════════════════════════");
        Console.WriteLine($"  Results: {passed}/{total} PASSED");
        Console.WriteLine($"  ══════════════════════════════════════════════════════");

        inference?.Dispose();
        engine?.Dispose();

        return passed == total ? 0 : 1;
    }

    // ═════════════════════════════════════════════════════════════
    //  CLI PARSING
    // ═════════════════════════════════════════════════════════════

    private record Config
    {
        public string DataDir { get; init; } = Path.GetFullPath(
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "preprocessed-data"));
        public string ModelDir { get; init; } = Path.GetFullPath(
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "controller_csharp", "models"));
        public int MaxSteps { get; init; } = 17_280;
        public int Skip { get; init; } = 1;
        public ulong Seed { get; init; } = 42;
        public int Port { get; init; } = 8765;
        public bool NoWebSocket { get; init; } = false;
        public string? ReplayPath { get; init; } = null;
        public double ReplaySpeed { get; init; } = 1.0;
        public bool RunTest { get; init; } = false;
    }

    private static Config ParseArgs(string[] args)
    {
        var config = new Config();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--data-dir" when i + 1 < args.Length:
                    config = config with { DataDir = Path.GetFullPath(args[++i]) };
                    break;
                case "--model-dir" when i + 1 < args.Length:
                    config = config with { ModelDir = Path.GetFullPath(args[++i]) };
                    break;
                case "--steps" when i + 1 < args.Length:
                    config = config with { MaxSteps = int.Parse(args[++i]) };
                    break;
                case "--skip" when i + 1 < args.Length:
                    config = config with { Skip = Math.Max(1, int.Parse(args[++i])) };
                    break;
                case "--seed" when i + 1 < args.Length:
                    config = config with { Seed = ulong.Parse(args[++i]) };
                    break;
                case "--port" when i + 1 < args.Length:
                    config = config with { Port = int.Parse(args[++i]) };
                    break;
                case "--no-ws":
                    config = config with { NoWebSocket = true };
                    break;
                case "--replay" when i + 1 < args.Length:
                    config = config with { ReplayPath = args[++i] };
                    break;
                case "--speed" when i + 1 < args.Length:
                    config = config with { ReplaySpeed = double.Parse(args[++i]) };
                    break;
                case "--test":
                    config = config with { RunTest = true };
                    break;
            }
        }
        return config;
    }

    // ═════════════════════════════════════════════════════════════
    //  TEST HELPERS
    // ═════════════════════════════════════════════════════════════

    private static void Assert(bool condition, string msg)
    {
        if (!condition) throw new Exception(msg);
    }

    private static void Pass(int num, string name, string detail)
    {
        Console.WriteLine($"  [{num,2}] ✓ {name,-35} {detail}");
    }

    private static void Fail(int num, string name, string detail)
    {
        Console.WriteLine($"  [{num,2}] ✗ {name,-35} {detail}");
    }
}
