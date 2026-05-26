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
using System.Text.Json;
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
        Console.WriteLine("  Mode: LIVE CONSTELLATION SIMULATION");
        Console.WriteLine($"  Data dir:  {config.DataDir}");
        Console.WriteLine($"  Model dir: {config.ModelDir}");
        Console.WriteLine($"  Steps:     {config.MaxSteps}");
        Console.WriteLine($"  Skip:      {config.Skip}x (broadcast every {config.Skip} steps)");
        Console.WriteLine($"  Seed:      {config.Seed}");
        Console.WriteLine($"  WebSocket: {(config.NoWebSocket ? "DISABLED" : $"ws://localhost:{config.Port}/")}");
        Console.WriteLine();

        const int NUM_SATS = 4;

        // ── 1. Create + Init Physics Engines ─────────────────────
        var engines = new PhysicsEngine[NUM_SATS];
        var governors = new FdirGovernor[NUM_SATS];
        var obsBuilders = new ObservationBuilder[NUM_SATS];
        var logDir = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".", "logs");
        var loggers = new TelemetryLogger[NUM_SATS];
        var states = new StatePacket[NUM_SATS];
        var actions = new ActionPacket[NUM_SATS];

        int[] seeds = { 42, 43, 44, 45 };
        double[] timeOffsets = { 0.0, 21600.0, 43200.0, 64800.0 }; // Phasing offsets (6h apart in orbit)

        for (int i = 0; i < NUM_SATS; i++)
        {
            engines[i] = new PhysicsEngine(config.DataDir, (ulong)seeds[i]);
            engines[i].ValidateAbi();
            engines[i].Init();
            engines[i].Reset();
            engines[i].SetTime(timeOffsets[i]);

            governors[i] = new FdirGovernor();
            obsBuilders[i] = new ObservationBuilder();
            loggers[i] = new TelemetryLogger(logDir, i);
            states[i] = new StatePacket();
            actions[i] = ActionPacket.CreateNoOp();
        }
        Console.WriteLine($"  {NUM_SATS} C++ physics engine instances initialised ✓");

        // ── 2. Load ONNX Models ──────────────────────────────────
        using var inference = new InferenceEngine(config.ModelDir);
        Console.WriteLine();

        // ── 4. WebSocket Server (optional) ───────────────────────
        WebSocketServer? wsServer = null;
        NetworkImpairment? impairment = null;
        var groundCmd = new GroundCommandState();
        if (!config.NoWebSocket)
        {
            impairment = new NetworkImpairment(
                minDelayMs: 100, maxDelayMs: 500,   // reduced for sim speed
                dropProbability: 0.02, seed: (int)config.Seed);
            wsServer = new WebSocketServer(config.Port, impairment);
            wsServer.OnCommandReceived += json => ParseGroundCommand(json, groundCmd);
            wsServer.Start();
        }

        // ── 5. Set Target Altitudes ──────────────────────────────
        if (config.TargetAlt.HasValue)
        {
            for (int i = 0; i < NUM_SATS; i++)
            {
                engines[i].SetTargetAltitude(config.TargetAlt.Value);
                obsBuilders[i].TargetAltKm = config.TargetAlt.Value;
            }
            groundCmd.TargetAltitudeKm = config.TargetAlt.Value;
            Console.WriteLine($"  [INIT] Set target altitude to: {config.TargetAlt.Value} km for all satellites");
        }
        Console.WriteLine();
        Console.WriteLine("  Starting constellation simulation loop...");
        Console.WriteLine("  ────────────────────────────────────────────────────────");

        var initAction = ActionPacket.CreateNoOp();
        for (int i = 0; i < NUM_SATS; i++)
        {
            engines[i].Step(ref initAction, ref states[i]);
        }

        int step = 0;
        int[] fdirOverrides = new int[NUM_SATS];
        int[] payloadOnSteps = new int[NUM_SATS];
        int[] eclipseSteps = new int[NUM_SATS];
        int[] saaSteps = new int[NUM_SATS];

        // ── 6. Main Simulation Loop ──────────────────────────────
        while (step < config.MaxSteps && !engines.All(e => e.IsDone))
        {
            // Apply one-shot inject SEU
            bool injectSeu = groundCmd.InjectSeu;
            if (injectSeu) groundCmd.InjectSeu = false;

            if (groundCmd.PresetDirty)
            {
                string preset = groundCmd.ActivePresetName;
                groundCmd.PresetDirty = false;
                
                double seu = 1.0, noise = 1.0, drift = 1.0, density = 0.01;
                double baseTime = 0.0;
                switch (preset)
                {
                    case "solarmax":
                        seu = 5.0; noise = 1.2; drift = 1.5; density = 0.3;
                        baseTime = 86400 * 180;
                        break;
                    case "halloween":
                        seu = 100.0; noise = 2.0; drift = 1.2; density = 0.15;
                        baseTime = 120700800.0; // Year 2003, DOY 302
                        break;
                    case "fuel_critical":
                        groundCmd.TargetAltitudeKm = 550.0;
                        break;
                    case "cold_eclipse":
                        seu = 1.0; noise = 1.2; drift = 1.0; density = 0.01;
                        baseTime = 86400 * 355; // DOY 355
                        break;
                }

                if (preset != "fuel_critical")
                {
                    for (int i = 0; i < NUM_SATS; i++)
                    {
                        engines[i].SetEnvironment(seu, noise, drift, density);
                        engines[i].SetTime(baseTime + timeOffsets[i]);
                    }
                    groundCmd.SeuMultiplier = seu;
                    groundCmd.NoiseMultiplier = noise;
                    groundCmd.DriftMultiplier = drift;
                    groundCmd.DensityMultiplier = density;
                }
            }

            if (groundCmd.EnvironmentDirty)
            {
                for (int i = 0; i < NUM_SATS; i++)
                {
                    engines[i].SetEnvironment(
                        groundCmd.SeuMultiplier,
                        groundCmd.NoiseMultiplier,
                        groundCmd.DriftMultiplier,
                        groundCmd.DensityMultiplier);
                }
                groundCmd.EnvironmentDirty = false;
                Console.WriteLine($"  [ENV] Applied to all satellites: SEU={groundCmd.SeuMultiplier}x " +
                    $"Noise={groundCmd.NoiseMultiplier}x Drift={groundCmd.DriftMultiplier}x " +
                    $"Density={groundCmd.DensityMultiplier}");
            }

            bool[] overriddenSats = new bool[NUM_SATS];

            for (int i = 0; i < NUM_SATS; i++)
            {
                if (engines[i].IsDone) continue;

                // a. Build normalised observation
                float[] obs = obsBuilders[i].Build(in states[i]);

                // b. Run ONNX inference (or use manual override)
                ActionPacket action;
                bool overridden;

                if (groundCmd.ManualOverride)
                {
                    action = groundCmd.ManualAction;
                    action.Version = 1;
                    overridden = true;
                }
                else
                {
                    AgentActions aiActions;
                    using (var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500)))
                    {
                        try
                        {
                            var inferTask = Task.Run(() => inference.Infer(obs), cts.Token);
                            if (await Task.WhenAny(inferTask, Task.Delay(500, cts.Token)) == inferTask)
                            {
                                aiActions = await inferTask;
                            }
                            else
                            {
                                throw new TimeoutException("Inference timed out (exceeded 500ms)");
                            }
                        }
                        catch (Exception ex)
                        {
                            // Inference timeout / error → force SAFE mode
                            states[i].FdirMode = (byte)FdirMode.Safe;
                            Console.WriteLine($"  [WATCHDOG] SAT {i} FAILED: {ex.Message} — forcing SAFE mode!");
                            aiActions = new AgentActions
                            {
                                Nav = new NavigationAction(),
                                DeepSleep = 1,
                                PayloadOn = 0
                            };
                        }
                    }
                    action = governors[i].Apply(aiActions, in states[i], out overridden);
                }
                overriddenSats[i] = overridden;
                if (overridden) fdirOverrides[i]++;

                // c. Apply ground command overrides
                if (injectSeu)
                {
                    action.InjectSeu = 1;
                }

                // c2. Altitude hold correction
                if (!groundCmd.ManualOverride && Math.Abs(groundCmd.TargetAltitudeKm - 600.0) > 1.0)
                {
                    double altError = groundCmd.TargetAltitudeKm - states[i].AltitudeKm;
                    float correction = (float)Math.Clamp(altError * 0.005, -0.5, 0.5);
                    action.ThrustZ = Math.Clamp(action.ThrustZ + correction, -1f, 1f);
                    float throttleBoost = (float)Math.Min(Math.Abs(altError) * 0.002, 0.3);
                    action.Throttle = Math.Clamp(action.Throttle + throttleBoost, 0f, 1f);
                }

                // d. Step the engine
                engines[i].Step(ref action, ref states[i]);

                // e. Apply forced FDIR mode
                if (groundCmd.ForcedFdirMode.HasValue)
                    states[i].FdirMode = (byte)groundCmd.ForcedFdirMode.Value;

                // f. Track metrics
                if (action.PayloadOn == 1) payloadOnSteps[i]++;
                if (states[i].InEclipse == 1) eclipseSteps[i]++;
                if (states[i].InSaa == 1) saaSteps[i]++;

                actions[i] = action;
            }
            step++;

            // g. Log + broadcast only on skip boundary
            bool isBroadcastStep = (step % config.Skip == 0) || engines.Any(e => e.IsDone);

            if (isBroadcastStep)
            {
                for (int i = 0; i < NUM_SATS; i++)
                {
                    loggers[i].LogStep(step, in states[i], in actions[i], overriddenSats[i],
                        manualOverride: groundCmd.ManualOverride,
                        seuMult: groundCmd.SeuMultiplier,
                        noiseMult: groundCmd.NoiseMultiplier,
                        driftMult: groundCmd.DriftMultiplier,
                        densityMult: groundCmd.DensityMultiplier);

                    if (wsServer != null)
                    {
                        byte[] packet = TelemetryPacket.Serialise(
                            (byte)i, (uint)step, in states[i], in actions[i], overriddenSats[i]);
                        wsServer.EnqueueFrame(packet);
                    }
                }

                if (wsServer != null)
                {
                    await wsServer.FlushAsync();
                }
            }

            // h. Periodic console output
            int consoleInterval = Math.Max(1000, config.Skip * 100);
            if (step % consoleInterval == 0 || engines.Any(e => e.IsDone))
            {
                Console.WriteLine($"[Step {step,6}] Constellation Status:");
                for (int i = 0; i < NUM_SATS; i++)
                {
                    double simHours = states[i].SimTimeS / 3600.0;
                    double nadirDeg = states[i].NadirError * (180.0 / Math.PI);
                    string status = states[i].IsDone == 1 ? $"DEAD ({states[i].DoneReasonEnum})" : "ALIVE";
                    Console.WriteLine($"  SAT {i}: Alt={states[i].AltitudeKm:F1}km | SoC={states[i].BatterySoc * 100:F1}% | Nadir={nadirDeg:F1}° | FDIR={FdirGovernor.ModeLabel(states[i].FdirMode)} | {status}");
                }
                Console.WriteLine();
            }
        }

        Console.WriteLine();
        Console.WriteLine("  ════════════════════════════════════════════════════════");
        Console.WriteLine("  CONSTELLATION EPISODE SUMMARY");
        Console.WriteLine("  ════════════════════════════════════════════════════════");
        Console.WriteLine($"  Total steps:      {step}");
        for (int i = 0; i < NUM_SATS; i++)
        {
            Console.WriteLine($"  SAT {i}:");
            Console.WriteLine($"    Final altitude:   {states[i].AltitudeKm:F2} km");
            Console.WriteLine($"    Final SoC:        {states[i].BatterySoc * 100:F2}%");
            Console.WriteLine($"    Final FDIR mode:  {FdirGovernor.ModeLabel(states[i].FdirMode)}");
            Console.WriteLine($"    FDIR overrides:   {fdirOverrides[i]} ({100.0 * fdirOverrides[i] / Math.Max(1, step):F1}%)");
            Console.WriteLine($"    Payload ON steps: {payloadOnSteps[i]} ({100.0 * payloadOnSteps[i] / Math.Max(1, step):F1}%)");
            Console.WriteLine($"    Eclipse steps:    {eclipseSteps[i]} ({100.0 * eclipseSteps[i] / Math.Max(1, step):F1}%)");
            Console.WriteLine($"    SAA steps:        {saaSteps[i]} ({100.0 * saaSteps[i] / Math.Max(1, step):F1}%)");
            Console.WriteLine($"    Telemetry log:    {loggers[i].FilePath}");
        }

        if (impairment != null)
            impairment.PrintSummary();

        Console.WriteLine("  ════════════════════════════════════════════════════════");

        // ── 8. Cleanup ───────────────────────────────────────────
        wsServer?.Dispose();
        for (int i = 0; i < NUM_SATS; i++)
        {
            engines[i].Dispose();
            loggers[i].Dispose();
        }
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
            Assert(csState == 230, $"StatePacket size: expected 230, got {csState}");
            Assert(csAction == 20, $"ActionPacket size: expected 20, got {csAction}");
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
            Assert(obs.Length == 42, $"Obs dim: expected 42, got {obs.Length}");
            // Check no NaN/Inf
            for (int i = 0; i < obs.Length; i++)
                Assert(!float.IsNaN(obs[i]) && !float.IsInfinity(obs[i]),
                       $"Obs[{i}] is NaN or Inf");
            Pass(4, "Observation builder", $"dim={obs.Length}, range=[{obs.Min():F3}, {obs.Max():F3}]");
            passed++;
        }
        catch (Exception ex) { Fail(4, "Observation builder", ex.Message); obs = new float[42]; }

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
        public double? TargetAlt { get; init; } = null;
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
                case "--target-alt" when i + 1 < args.Length:
                    config = config with { TargetAlt = double.Parse(args[++i]) };
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

    // ═════════════════════════════════════════════════════════════
    //  GROUND COMMAND STATE & PARSING
    // ═════════════════════════════════════════════════════════════

    /// <summary>
    /// Mutable state tracking ground commands from the WebGPU Developer Testbed.
    /// Thread-safe for simple field writes from WebSocket callbacks.
    /// </summary>
    private class GroundCommandState
    {
        // ── Manual Control ──
        public volatile bool ManualOverride;
        public ActionPacket ManualAction = ActionPacket.CreateNoOp();
        public double TargetAltitudeKm = 600.0;
        public volatile bool InjectSeu;
        public int? ForcedFdirMode;

        // ── Environment Tuning ──
        public double SeuMultiplier = 1.0;
        public double NoiseMultiplier = 1.0;
        public double DriftMultiplier = 1.0;
        public double DensityMultiplier = 0.01;
        public volatile bool EnvironmentDirty;

        // ── Presets ──
        public string ActivePresetName = "";
        public volatile bool PresetDirty;
    }

    private static void ParseGroundCommand(string json, GroundCommandState cmd)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            string type = root.GetProperty("type").GetString() ?? "";

            switch (type)
            {
                case "manual_override":
                    cmd.ManualOverride = root.GetProperty("manualOverride").GetBoolean();
                    if (cmd.ManualOverride && root.TryGetProperty("action", out var act))
                    {
                        cmd.ManualAction = new ActionPacket
                        {
                            Version = 1,
                            ThrustX = (float)act.GetProperty("thrustX").GetDouble(),
                            ThrustY = (float)act.GetProperty("thrustY").GetDouble(),
                            ThrustZ = (float)act.GetProperty("thrustZ").GetDouble(),
                            Throttle = (float)act.GetProperty("throttle").GetDouble(),
                            DeepSleep = (byte)(act.GetProperty("deepSleep").GetBoolean() ? 1 : 0),
                            PayloadOn = (byte)(act.GetProperty("payloadOn").GetBoolean() ? 1 : 0),
                        };
                    }
                    Console.WriteLine($"  [CMD] Manual override: {cmd.ManualOverride}");
                    break;

                case "target_altitude":
                    cmd.TargetAltitudeKm = root.GetProperty("targetAltitudeKm").GetDouble();
                    Console.WriteLine($"  [CMD] Target altitude: {cmd.TargetAltitudeKm} km");
                    break;

                case "inject_seu":
                    cmd.InjectSeu = true;
                    Console.WriteLine("  [CMD] SEU injection triggered");
                    break;

                case "force_fdir":
                    int mode = root.GetProperty("fdirMode").GetInt32();
                    cmd.ForcedFdirMode = mode < 0 ? null : mode;
                    Console.WriteLine($"  [CMD] Force FDIR: {(cmd.ForcedFdirMode.HasValue ? ((FdirMode)cmd.ForcedFdirMode.Value).ToString() : "Auto")}");
                    break;

                case "environment_tuning":
                    var env = root.GetProperty("environment");
                    cmd.SeuMultiplier = env.GetProperty("seuMultiplier").GetDouble();
                    cmd.NoiseMultiplier = env.GetProperty("noiseMultiplier").GetDouble();
                    cmd.DriftMultiplier = env.GetProperty("driftMultiplier").GetDouble();
                    cmd.DensityMultiplier = env.GetProperty("densityMultiplier").GetDouble();
                    cmd.EnvironmentDirty = true;
                    Console.WriteLine($"  [CMD] Environment: SEU={cmd.SeuMultiplier}x Noise={cmd.NoiseMultiplier}x Drift={cmd.DriftMultiplier}x Density={cmd.DensityMultiplier}");
                    break;

                case "preset":
                    cmd.ActivePresetName = root.GetProperty("presetName").GetString() ?? "";
                    cmd.PresetDirty = true;
                    Console.WriteLine($"  [CMD] Preset triggered: {cmd.ActivePresetName}");
                    break;

                default:
                    Console.WriteLine($"  [CMD] Unknown command type: {type}");
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [CMD] Parse error: {ex.Message}");
        }
    }
}
