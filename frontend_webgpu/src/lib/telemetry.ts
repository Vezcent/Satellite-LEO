import { useState, useEffect, useRef, useCallback } from 'react';

export const FdirMode = {
  Nominal: 0,
  Degraded: 1,
  Safe: 2,
  Recovery: 3
} as const;

export type FdirMode = typeof FdirMode[keyof typeof FdirMode];

export interface TelemetryData {
  seq: number;
  simTimeS: number;
  altitudeKm: number;
  latitudeDeg: number;
  longitudeDeg: number;
  batterySoc: number;
  solarPowerW: number;
  powerDrawW: number;
  inEclipse: boolean;
  inSaa: boolean;
  fdirMode: FdirMode;
  seuActive: boolean;
  gsVisible: boolean;
  panelEfficiency: number;
  dragCoeff: number;
  isDone: boolean;
  doneReason: number;
  
  // Actions
  thrustX: number;
  thrustY: number;
  thrustZ: number;
  throttle: number;
  deepSleep: boolean;
  payloadOn: boolean;
  fdirOverridden: boolean;
}

const RECONNECT_INTERVAL_MS = 2000;  // Retry every 2 seconds
const MAX_RECONNECT_ATTEMPTS = 999;  // Keep trying indefinitely

export function useTelemetry(url: string = 'ws://localhost:8765') {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Clean up previous socket
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }

    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      attemptRef.current = 0;
      console.log('[Telemetry] Connected to', url);
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;

      // Auto-reconnect
      if (mountedRef.current && attemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        attemptRef.current++;
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, RECONNECT_INTERVAL_MS);
      }
    };

    ws.onerror = () => {
      // onclose will fire after this, which handles reconnect
    };
    
    ws.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) return;
      const buffer = event.data;
      const view = new DataView(buffer);
      
      // Packet format: [Version(1) | Seq(4) | PayloadLen(4) | Payload(N) | CRC32(4)]
      if (buffer.byteLength < 9) return;
      
      const version = view.getUint8(0);
      if (version !== 1) return; // Unsupported version
      
      const seq = view.getUint32(1, true); // Little endian
      // const payloadLen = view.getUint32(5, true);
      
      // Parse Payload (offset 9)
      let offset = 9;
      
      const simTimeS = view.getFloat64(offset, true); offset += 8;
      const altitudeKm = view.getFloat64(offset, true); offset += 8;
      const latitudeDeg = view.getFloat64(offset, true); offset += 8;
      const longitudeDeg = view.getFloat64(offset, true); offset += 8;
      const batterySoc = view.getFloat64(offset, true); offset += 8;
      const solarPowerW = view.getFloat64(offset, true); offset += 8;
      const powerDrawW = view.getFloat64(offset, true); offset += 8;
      
      const inEclipse = view.getUint8(offset++) === 1;
      const inSaa = view.getUint8(offset++) === 1;
      const fdirMode = view.getUint8(offset++) as FdirMode;
      const seuActive = view.getUint8(offset++) === 1;
      const gsVisible = view.getUint8(offset++) === 1;
      
      const panelEfficiency = view.getFloat64(offset, true); offset += 8;
      const dragCoeff = view.getFloat64(offset, true); offset += 8;
      
      const isDone = view.getUint8(offset++) === 1;
      const doneReason = view.getUint8(offset++);
      
      const thrustX = view.getFloat32(offset, true); offset += 4;
      const thrustY = view.getFloat32(offset, true); offset += 4;
      const thrustZ = view.getFloat32(offset, true); offset += 4;
      const throttle = view.getFloat32(offset, true); offset += 4;
      
      const deepSleep = view.getUint8(offset++) === 1;
      const payloadOn = view.getUint8(offset++) === 1;
      const fdirOverridden = view.getUint8(offset++) === 1;

      setData({
        seq, simTimeS, altitudeKm, latitudeDeg, longitudeDeg,
        batterySoc, solarPowerW, powerDrawW, inEclipse, inSaa,
        fdirMode, seuActive, gsVisible, panelEfficiency, dragCoeff,
        isDone, doneReason, thrustX, thrustY, thrustZ, throttle,
        deepSleep, payloadOn, fdirOverridden
      });
    };
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { data, connected };
}
