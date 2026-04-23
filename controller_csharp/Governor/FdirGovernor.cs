/*
 * S-MAS Phase 4 — Governor/FdirGovernor.cs
 *
 * FDIR (Failure Detection, Isolation and Recovery) Safety Governor.
 * Reads the FDIR mode from the C++ engine's StatePacket and applies
 * action overrides BEFORE sending actions to the engine.
 *
 * State Machine (matches C++ simulation_engine.cpp):
 *   NOMINAL  (mode=0): Full AI control.
 *   DEGRADED (mode=1): Payload OFF, throttle capped to 30%.
 *   SAFE     (mode=2): AI completely overridden — forced deep sleep, no thrust, no payload.
 *   RECOVERY (mode=3): AI control with payload OFF until NOMINAL.
 *
 * Meta-Coordination (Phase 3 §3.3):
 *   If deep_sleep == 1, payload_on is forced to 0 regardless of Mission Agent output.
 */
using SmasController.AI;
using SmasController.Interop;

namespace SmasController.Governor;

/// <summary>
/// Safety governor that filters AI actions based on the current FDIR mode.
/// Acts as the outer safety layer on top of the C++ engine's internal FDIR.
/// </summary>
public sealed class FdirGovernor
{
    // Throttle cap in DEGRADED mode (30%)
    private const float DegradedThrottleCap = 0.3f;

    /// <summary>
    /// Apply FDIR safety overrides to the raw AI actions.
    /// Returns a new ActionPacket safe for the current FDIR mode.
    /// </summary>
    /// <param name="aiActions">Raw agent inference output.</param>
    /// <param name="state">Current state from the C++ engine.</param>
    /// <param name="overridden">True if any action was modified by the governor.</param>
    public ActionPacket Apply(AgentActions aiActions, in StatePacket state, out bool overridden)
    {
        var action = new ActionPacket { Version = 1 };
        overridden = false;

        var mode = (FdirMode)state.FdirMode;

        switch (mode)
        {
            case FdirMode.Nominal:
                // Full AI control — pass through
                action.ThrustX   = aiActions.Nav.ThrustX;
                action.ThrustY   = aiActions.Nav.ThrustY;
                action.ThrustZ   = aiActions.Nav.ThrustZ;
                action.Throttle  = aiActions.Nav.Throttle;
                action.DeepSleep = aiActions.DeepSleep;
                action.PayloadOn = aiActions.PayloadOn;
                break;

            case FdirMode.Degraded:
                // Restricted control: payload OFF, throttle capped
                action.ThrustX   = aiActions.Nav.ThrustX;
                action.ThrustY   = aiActions.Nav.ThrustY;
                action.ThrustZ   = aiActions.Nav.ThrustZ;
                action.Throttle  = Math.Min(aiActions.Nav.Throttle, DegradedThrottleCap);
                action.DeepSleep = aiActions.DeepSleep;
                action.PayloadOn = 0;  // forced OFF

                if (aiActions.PayloadOn == 1 || aiActions.Nav.Throttle > DegradedThrottleCap)
                    overridden = true;
                break;

            case FdirMode.Safe:
                // Complete AI override — forced deep sleep, everything off
                action.ThrustX   = 0f;
                action.ThrustY   = 0f;
                action.ThrustZ   = 0f;
                action.Throttle  = 0f;
                action.DeepSleep = 1;
                action.PayloadOn = 0;
                overridden = true;
                break;

            case FdirMode.Recovery:
                // AI control allowed but payload stays OFF
                action.ThrustX   = aiActions.Nav.ThrustX;
                action.ThrustY   = aiActions.Nav.ThrustY;
                action.ThrustZ   = aiActions.Nav.ThrustZ;
                action.Throttle  = aiActions.Nav.Throttle;
                action.DeepSleep = aiActions.DeepSleep;
                action.PayloadOn = 0;  // OFF until NOMINAL

                if (aiActions.PayloadOn == 1)
                    overridden = true;
                break;

            default:
                // Unknown mode — safe fallback
                action.DeepSleep = 1;
                overridden = true;
                break;
        }

        // ── Meta-Coordination (Phase 3) ──────────────────────────
        // Deep sleep → force payload off (regardless of mission agent)
        if (action.DeepSleep == 1 && action.PayloadOn == 1)
        {
            action.PayloadOn = 0;
            overridden = true;
        }

        return action;
    }

    /// <summary>Get a human-readable label for the FDIR mode.</summary>
    public static string ModeLabel(byte mode) => mode switch
    {
        0 => "NOMINAL",
        1 => "DEGRADED",
        2 => "SAFE",
        3 => "RECOVERY",
        _ => $"UNKNOWN({mode})"
    };
}
