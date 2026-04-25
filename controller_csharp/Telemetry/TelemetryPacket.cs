/*
 * S-MAS Phase 4 — Telemetry/TelemetryPacket.cs
 *
 * Binary telemetry packet definition for WebSocket streaming.
 * Schema: [Version(1B) | Seq(4B) | PayloadLen(4B) | Payload(NB) | CRC32(4B)]
 */
using System.Buffers.Binary;
using SmasController.Interop;

namespace SmasController.Telemetry;

/// <summary>
/// Serialises a simulation frame into a compact binary packet
/// for WebSocket transmission.
/// </summary>
public static class TelemetryPacket
{
    public const byte PacketVersion = 1;

    /// <summary>
    /// Serialise a simulation frame into a binary telemetry packet.
    /// </summary>
    /// <param name="seq">Frame sequence number.</param>
    /// <param name="state">Current StatePacket from C++.</param>
    /// <param name="action">ActionPacket that was sent.</param>
    /// <param name="fdirOverridden">Whether the FDIR governor overrode any actions.</param>
    public static byte[] Serialise(uint seq, in StatePacket state, in ActionPacket action, bool fdirOverridden)
    {
        // Payload: key telemetry fields in compact binary form
        // This is a curated subset — sending the full 184B StatePacket + extras
        using var ms = new MemoryStream(256);
        using var bw = new BinaryWriter(ms);

        // ── Payload body ──
        bw.Write(state.SimTimeS);
        bw.Write(state.AltitudeKm);
        bw.Write(state.LatitudeDeg);
        bw.Write(state.LongitudeDeg);
        bw.Write(state.BatterySoc);
        bw.Write(state.SolarPowerW);
        bw.Write(state.PowerDrawW);
        bw.Write(state.InEclipse);
        bw.Write(state.InSaa);
        bw.Write(state.FdirMode);
        bw.Write(state.SeuActive);
        bw.Write(state.GsVisible);
        bw.Write(state.PanelEfficiency);
        bw.Write(state.DragCoeff);
        bw.Write(state.AtmDensity);
        bw.Write(state.SaaFlux10Mev);
        bw.Write(state.SaaFlux30Mev);
        bw.Write(state.IsDone);
        bw.Write(state.DoneReasonVal);
        // Actions taken
        bw.Write(action.ThrustX);
        bw.Write(action.ThrustY);
        bw.Write(action.ThrustZ);
        bw.Write(action.Throttle);
        bw.Write(action.DeepSleep);
        bw.Write(action.PayloadOn);
        bw.Write((byte)(fdirOverridden ? 1 : 0));

        bw.Flush();
        byte[] payload = ms.ToArray();

        // ── Build full packet ──
        // [Version(1) | Seq(4) | PayloadLen(4) | Payload(N) | CRC32(4)]
        int totalLen = 1 + 4 + 4 + payload.Length + 4;
        byte[] packet = new byte[totalLen];
        int offset = 0;

        packet[offset++] = PacketVersion;
        BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(offset), seq);
        offset += 4;
        BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(offset), (uint)payload.Length);
        offset += 4;
        Buffer.BlockCopy(payload, 0, packet, offset, payload.Length);
        offset += payload.Length;

        // CRC32 over everything before the checksum
        uint crc = Crc32(packet.AsSpan(0, offset));
        BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(offset), crc);

        return packet;
    }

    /// <summary>Simple CRC32 implementation (IEEE polynomial).</summary>
    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
                crc = (crc >> 1) ^ (0xEDB88320 & (uint)(-(int)(crc & 1)));
        }
        return ~crc;
    }
}
