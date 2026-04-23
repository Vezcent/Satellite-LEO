/*
 * S-MAS Phase 4 — Interop/EngineApi.cs
 *
 * P/Invoke declarations for the C++ physics engine DLL.
 * Maps directly to backend_cpp/include/c_api.h.
 *
 * The DLL (smas_engine.dll) is copied to the output directory
 * by the .csproj, so no absolute paths are needed.
 */
using System.Runtime.InteropServices;

namespace SmasController.Interop;

/// <summary>
/// Raw P/Invoke bindings to the smas_engine.dll C API.
/// </summary>
public static class EngineApi
{
    private const string DllName = "smas_engine.dll";

    /// <summary>
    /// Create a simulation engine instance.
    /// </summary>
    /// <param name="dataDir">Path to the preprocessed-data directory.</param>
    /// <param name="seed">Random seed for stochastic modules.</param>
    /// <param name="densityMultiplier">Atmospheric density scaling factor (default 1.0).</param>
    /// <returns>Opaque engine handle (IntPtr.Zero on failure).</returns>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    public static extern IntPtr smas_create(
        [MarshalAs(UnmanagedType.LPStr)] string dataDir,
        ulong seed,
        double densityMultiplier = 1.0);

    /// <summary>
    /// Initialise the engine (load data files).
    /// </summary>
    /// <returns>0 on success, -1 on failure.</returns>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int smas_init(IntPtr engine);

    /// <summary>Reset the engine to initial orbital conditions.</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void smas_reset(IntPtr engine);

    /// <summary>
    /// Advance the simulation by one time step (dt = 5.0s).
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void smas_step(
        IntPtr engine,
        ref ActionPacket action,
        ref StatePacket outState);

    /// <summary>Check if the episode is done.</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int smas_is_done(IntPtr engine);

    /// <summary>Destroy and free the engine instance.</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void smas_destroy(IntPtr engine);

    /// <summary>Return sizeof(StatePacket) from C++ for ABI validation.</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int smas_state_packet_size();

    /// <summary>Return sizeof(ActionPacket) from C++ for ABI validation.</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int smas_action_packet_size();
}

/// <summary>
/// Managed wrapper around the raw P/Invoke engine handle.
/// Provides IDisposable cleanup and ABI validation.
/// </summary>
public sealed class PhysicsEngine : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    public PhysicsEngine(string dataDir, ulong seed = 42, double densityMultiplier = 0.1)
    {
        _handle = EngineApi.smas_create(dataDir, seed, densityMultiplier);
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("smas_create returned NULL — check data directory.");
    }

    /// <summary>
    /// Validate that C# struct sizes match the C++ sizes exactly.
    /// Must be called before any other operations.
    /// </summary>
    public void ValidateAbi()
    {
        int cStateSize  = EngineApi.smas_state_packet_size();
        int cActionSize = EngineApi.smas_action_packet_size();
        int csStateSize  = Marshal.SizeOf<StatePacket>();
        int csActionSize = Marshal.SizeOf<ActionPacket>();

        if (cStateSize != csStateSize || cActionSize != csActionSize)
        {
            throw new InvalidOperationException(
                $"ABI MISMATCH! C++: State={cStateSize} Action={cActionSize}, " +
                $"C#: State={csStateSize} Action={csActionSize}. " +
                "Check contracts.h vs Contracts.cs field alignment.");
        }

        Console.WriteLine($"  ABI check passed: StatePacket={csStateSize}B, ActionPacket={csActionSize}B");
    }

    /// <summary>Initialise the engine (load preprocessed data).</summary>
    public void Init()
    {
        int rc = EngineApi.smas_init(_handle);
        if (rc != 0)
            throw new InvalidOperationException("smas_init failed — check preprocessed-data paths.");
    }

    /// <summary>Reset the simulation to initial orbital state.</summary>
    public void Reset()
    {
        EngineApi.smas_reset(_handle);
    }

    /// <summary>Step the simulation forward by dt=5.0s.</summary>
    public void Step(ref ActionPacket action, ref StatePacket state)
    {
        EngineApi.smas_step(_handle, ref action, ref state);
    }

    /// <summary>Check if the current episode has ended.</summary>
    public bool IsDone => EngineApi.smas_is_done(_handle) != 0;

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            EngineApi.smas_destroy(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
