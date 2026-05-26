import { useState, useEffect, useRef, useCallback } from 'react';

export const FdirMode = {
  Nominal: 0,
  Degraded: 1,
  Safe: 2,
  Recovery: 3
} as const;

export type FdirMode = typeof FdirMode[keyof typeof FdirMode];

export interface TelemetryData {
  satId: number;
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
  atmDensity: number;
  saaFlux10: number;
  saaFlux30: number;
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

  // Phase A Subsystems (Fuel & Thermal)
  fuelFraction: number;    // 0.0 - 1.0
  fuelDepleted: boolean;
  tempBus: number;         // °C
  tempBattery: number;     // °C
  tempPayload: number;     // °C
  heaterOn: boolean;

  // Phase A ADCS Subsystems (Attitude)
  sunAngle: number;
  nadirError: number;
  wheelMomentumX: number;
  wheelMomentumY: number;
  wheelMomentumZ: number;

  // Comms & Data (Phase B Step 7)
  dataBufferMb: number;
  snrDb: number;
}

// ── Ground Command Interface (Frontend → C# Controller) ──────────
export interface GroundCommand {
  type: 'manual_override' | 'target_altitude' | 'inject_seu'
      | 'force_fdir' | 'environment_tuning';
  manualOverride?: boolean;
  action?: {
    thrustX: number; thrustY: number; thrustZ: number;
    throttle: number; deepSleep: boolean; payloadOn: boolean;
  };
  targetAltitudeKm?: number;
  fdirMode?: number;           // 0-3, or -1 for auto
  environment?: {
    seuMultiplier: number;     // 0.1x – 100x
    noiseMultiplier: number;   // 0x – 5x
    driftMultiplier: number;   // 0x – 10x
    densityMultiplier: number; // 0.001 – 1.0
  };
}

const RECONNECT_INTERVAL_MS = 2000;  // Retry every 2 seconds
const MAX_RECONNECT_ATTEMPTS = 999;  // Keep trying indefinitely

export function useTelemetry(url: string = 'ws://localhost:8765') {
  const [satellites, setSatellites] = useState<(TelemetryData | null)[]>([null, null, null, null]);
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
      
      // Packet format: [Version(1) | SatID(1) | Seq(4) | PayloadLen(4) | Payload(N) | CRC32(4)]
      if (buffer.byteLength < 10) return;
      
      const version = view.getUint8(0);
      if (version !== 1) return; // Unsupported version
      
      const satId = view.getUint8(1);
      const seq = view.getUint32(2, true); // Little endian
      // const payloadLen = view.getUint32(6, true);
      
      // Parse Payload (offset 10)
      let offset = 10;
      
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
      const atmDensity = view.getFloat64(offset, true); offset += 8;
      const saaFlux10 = view.getFloat32(offset, true); offset += 4;
      const saaFlux30 = view.getFloat32(offset, true); offset += 4;
      
      const isDone = view.getUint8(offset++) === 1;
      const doneReason = view.getUint8(offset++);
      
      const thrustX = view.getFloat32(offset, true); offset += 4;
      const thrustY = view.getFloat32(offset, true); offset += 4;
      const thrustZ = view.getFloat32(offset, true); offset += 4;
      const throttle = view.getFloat32(offset, true); offset += 4;
      
      const deepSleep = view.getUint8(offset++) === 1;
      const payloadOn = view.getUint8(offset++) === 1;
      const fdirOverridden = view.getUint8(offset++) === 1;

      // Phase A Subsystems (Fuel & Thermal)
      const fuelFraction = view.getFloat32(offset, true); offset += 4;
      const fuelDepleted = view.getUint8(offset++) === 1;
      const tempBus = view.getFloat32(offset, true); offset += 4;
      const tempBattery = view.getFloat32(offset, true); offset += 4;
      const tempPayload = view.getFloat32(offset, true); offset += 4;
      const heaterOn = view.getUint8(offset++) === 1;

      // Phase A ADCS Subsystem (Attitude)
      const sunAngle = view.getFloat32(offset, true); offset += 4;
      const nadirError = view.getFloat32(offset, true); offset += 4;
      const wheelMomentumX = view.getFloat32(offset, true); offset += 4;
      const wheelMomentumY = view.getFloat32(offset, true); offset += 4;
      const wheelMomentumZ = view.getFloat32(offset, true); offset += 4;

      // Comms & Data (Phase B Step 7)
      const dataBufferMb = view.getFloat32(offset, true); offset += 4;
      const snrDb = view.getFloat32(offset, true); offset += 4;

      const parsedData: TelemetryData = {
        satId, seq, simTimeS, altitudeKm, latitudeDeg, longitudeDeg,
        batterySoc, solarPowerW, powerDrawW, inEclipse, inSaa,
        fdirMode, seuActive, gsVisible, panelEfficiency, dragCoeff,
        atmDensity, saaFlux10, saaFlux30,
        isDone, doneReason, thrustX, thrustY, thrustZ, throttle,
        deepSleep, payloadOn, fdirOverridden,
        fuelFraction, fuelDepleted, tempBus, tempBattery, tempPayload, heaterOn,
        sunAngle, nadirError, wheelMomentumX, wheelMomentumY, wheelMomentumZ,
        dataBufferMb, snrDb
      };

      setSatellites(prev => {
        const next = [...prev];
        if (satId >= 0 && satId < 4) {
          next[satId] = parsedData;
        }
        return next;
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

  const sendCommand = useCallback((cmd: GroundCommand) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
      console.log('[Telemetry] Sent command:', cmd.type);
    } else {
      console.warn('[Telemetry] Cannot send command — not connected');
    }
  }, []);

  return { satellites, connected, sendCommand };
}
