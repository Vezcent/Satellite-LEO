/*
 * S-MAS Phase 4 — Telemetry/WebSocketServer.cs
 *
 * Async WebSocket server for real-time binary telemetry streaming.
 * Listens on ws://localhost:{port} and broadcasts simulation frames
 * to all connected clients using the TelemetryPacket binary format.
 *
 * Features:
 *   - Multi-client support via concurrent connection tracking
 *   - Circular frame buffer with backpressure (drops stale frames)
 *   - Network impairment injection (delay + packet drop)
 *   - Graceful shutdown via CancellationToken
 */
using System.Collections.Concurrent;
using System.Net;
using System.Net.WebSockets;

namespace SmasController.Telemetry;

/// <summary>
/// Async WebSocket server for broadcasting simulation telemetry.
/// </summary>
public sealed class WebSocketServer : IDisposable
{
    private readonly HttpListener _httpListener;
    private readonly ConcurrentDictionary<Guid, WebSocket> _clients = new();
    private readonly NetworkImpairment _impairment;
    private readonly int _maxBufferedFrames;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;

    // Circular frame buffer for backpressure
    private readonly ConcurrentQueue<byte[]> _frameQueue = new();

    public int Port { get; }
    public int ConnectedClients => _clients.Count;

    /// <summary>
    /// Create a WebSocket telemetry server.
    /// </summary>
    /// <param name="port">Port to listen on (default 8765).</param>
    /// <param name="impairment">Network impairment simulator (optional).</param>
    /// <param name="maxBufferedFrames">Max frames in backpressure buffer before dropping stale ones.</param>
    public WebSocketServer(int port = 8765, NetworkImpairment? impairment = null,
                           int maxBufferedFrames = 32)
    {
        Port = port;
        _impairment = impairment ?? new NetworkImpairment();
        _maxBufferedFrames = maxBufferedFrames;

        _httpListener = new HttpListener();
        _httpListener.Prefixes.Add($"http://localhost:{port}/");
    }

    /// <summary>Start the server and begin accepting connections.</summary>
    public void Start()
    {
        _cts = new CancellationTokenSource();
        _httpListener.Start();
        _listenTask = AcceptLoop(_cts.Token);
        Console.WriteLine($"  WebSocket server started on ws://localhost:{Port}/");
    }

    /// <summary>
    /// Enqueue a frame for broadcast to all connected clients.
    /// If the buffer is full, the oldest frame is dropped (backpressure).
    /// </summary>
    public void EnqueueFrame(byte[] packetData)
    {
        // Backpressure: drop oldest frames if queue is full
        while (_frameQueue.Count >= _maxBufferedFrames)
            _frameQueue.TryDequeue(out _);

        _frameQueue.Enqueue(packetData);
    }

    /// <summary>
    /// Flush all queued frames to connected clients.
    /// Should be called periodically from the simulation loop.
    /// </summary>
    public async Task FlushAsync(CancellationToken ct = default)
    {
        while (_frameQueue.TryDequeue(out byte[]? frame))
        {
            if (_impairment.ShouldDrop())
                continue;  // simulate packet loss

            // Apply delay (non-blocking for simulation loop)
            var delay = _impairment.GetDelay();

            var deadClients = new List<Guid>();

            foreach (var (id, ws) in _clients)
            {
                if (ws.State != WebSocketState.Open)
                {
                    deadClients.Add(id);
                    continue;
                }

                try
                {
                    // Send with optional delay
                    if (delay > TimeSpan.Zero)
                        await Task.Delay(delay, ct);

                    await ws.SendAsync(
                        new ArraySegment<byte>(frame),
                        WebSocketMessageType.Binary,
                        endOfMessage: true,
                        ct);
                }
                catch (Exception)
                {
                    deadClients.Add(id);
                }
            }

            // Cleanup dead connections
            foreach (var id in deadClients)
            {
                if (_clients.TryRemove(id, out var ws))
                {
                    try { ws.Dispose(); } catch { /* ignore */ }
                }
            }
        }
    }

    // ── Connection acceptance loop ────────────────────────────────

    private async Task AcceptLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var httpContext = await _httpListener.GetContextAsync();

                if (httpContext.Request.IsWebSocketRequest)
                {
                    var wsContext = await httpContext.AcceptWebSocketAsync(null);
                    var clientId = Guid.NewGuid();
                    _clients.TryAdd(clientId, wsContext.WebSocket);
                    Console.WriteLine($"  [WS] Client connected: {clientId} (total: {_clients.Count})");

                    // Start background receive loop (handles client disconnect)
                    _ = ReceiveLoop(clientId, wsContext.WebSocket, ct);
                }
                else
                {
                    httpContext.Response.StatusCode = 400;
                    httpContext.Response.Close();
                }
            }
            catch (ObjectDisposedException) { break; }
            catch (HttpListenerException) { break; }
            catch (Exception ex)
            {
                Console.WriteLine($"  [WS] Accept error: {ex.Message}");
            }
        }
    }

    private async Task ReceiveLoop(Guid clientId, WebSocket ws, CancellationToken ct)
    {
        var buf = new byte[256];
        try
        {
            while (ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                if (result.MessageType == WebSocketMessageType.Close)
                    break;
                // We don't expect client-to-server messages in this protocol,
                // but we keep the loop alive to detect disconnects.
            }
        }
        catch { /* Expected on disconnect */ }
        finally
        {
            if (_clients.TryRemove(clientId, out _))
                Console.WriteLine($"  [WS] Client disconnected: {clientId} (total: {_clients.Count})");
            try { ws.Dispose(); } catch { /* ignore */ }
        }
    }

    public void Dispose()
    {
        _cts?.Cancel();
        foreach (var (_, ws) in _clients)
        {
            try { ws.Dispose(); } catch { /* ignore */ }
        }
        _clients.Clear();
        try { _httpListener.Stop(); } catch { /* ignore */ }
        _cts?.Dispose();
    }
}
