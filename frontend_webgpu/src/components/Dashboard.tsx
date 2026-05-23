import { useEffect, useRef } from 'react';
import { useTelemetry, FdirMode } from '../lib/telemetry';
import { Activity, Battery, ShieldAlert, Cpu, Radio, Orbit, ThermometerSun, Clock, Skull, Flame, Thermometer } from 'lucide-react';
import Renderer from '../engine/Renderer';
import GroundControlPanel from './GroundControlPanel';
import MusicPlayer from './MusicPlayer';
import '../App.css';

/** Convert simulation seconds to a human-readable lifetime string */
function formatLifetime(totalSeconds: number): string {
  const days = Math.floor(totalSeconds / 86400);
  const years = Math.floor(days / 365);
  const remainingDays = days - years * 365;
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);

  if (years > 0) {
    return `${years}y ${remainingDays}d ${hours}h`;
  } else if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  } else {
    return `${hours}h ${minutes}m`;
  }
}

// ── Circular Progress Ring for Fuel Gauge ────────────────────────
function CircularProgress({ value, label, size = 80, strokeWidth = 6 }: { value: number; label: string; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value * circumference);
  
  let color = '#22C55E'; // Green
  let glowColor = 'rgba(34, 197, 94, 0.3)';
  if (value === 0) {
    color = '#4B5563'; // Grey
    glowColor = 'rgba(75, 85, 99, 0.1)';
  } else if (value <= 0.2) {
    color = '#EF4444'; // Red
    glowColor = 'rgba(239, 68, 68, 0.4)';
  } else if (value <= 0.5) {
    color = '#F59E0B'; // Amber
    glowColor = 'rgba(245, 158, 11, 0.3)';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{
              transition: 'stroke-dashoffset 0.35s',
              filter: `drop-shadow(0 0 4px ${glowColor})`
            }}
          />
        </svg>
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace"
        }}>
          {value === 0 ? (
            <span style={{ fontSize: '0.6rem', fontWeight: 700, color: '#EF4444', letterSpacing: '0.05em' }}>DEPLETED</span>
          ) : (
            <>
              <span style={{ fontSize: '1.0rem', fontWeight: 700, color: 'white' }}>
                {(value * 100).toFixed(0)}
              </span>
              <span style={{ fontSize: '0.55rem', color: 'var(--color-text-secondary)' }}>%</span>
            </>
          )}
        </div>
      </div>
      <span style={{ fontSize: '0.65rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
        {label}
      </span>
    </div>
  );
}

// ── Temperature Bar with Safe Zone Highlight ─────────────────────
function TemperatureBar({ label, value, min = -40, max = 60, safeMin, safeMax }: { label: string; value: number; min?: number; max?: number; safeMin: number; safeMax: number }) {
  const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const isSafe = value >= safeMin && value <= safeMax;
  
  const safeLeft = ((safeMin - min) / (max - min)) * 100;
  const safeWidth = ((safeMax - safeMin) / (max - min)) * 100;

  const barColor = isSafe ? '#10B981' : value < safeMin ? '#3B82F6' : '#EF4444'; // Green, Blue (cold), Red (hot)
  const glowColor = isSafe ? 'rgba(16, 185, 129, 0.2)' : value < safeMin ? 'rgba(59, 130, 246, 0.2)' : 'rgba(239, 68, 68, 0.2)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', fontFamily: 'monospace' }}>
        <span style={{ color: 'var(--color-text-secondary)' }}>{label}</span>
        <span style={{ color: isSafe ? 'white' : barColor, fontWeight: 600 }}>{value.toFixed(1)}°C</span>
      </div>
      <div style={{ position: 'relative', height: '5px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute',
          left: `${safeLeft}%`,
          width: `${safeWidth}%`,
          height: '100%',
          background: 'rgba(255, 255, 255, 0.03)',
          borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)'
        }} />
        <div style={{
          width: `${percentage}%`,
          height: '100%',
          background: barColor,
          borderRadius: '3px',
          boxShadow: `0 0 4px ${glowColor}`,
          transition: 'width 0.3s ease-in-out'
        }} />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, connected, sendCommand } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Renderer | null>(null);

  useEffect(() => {
    if (canvasRef.current && !engineRef.current) {
      engineRef.current = new Renderer(canvasRef.current);
      engineRef.current.init().catch(console.error);
    }
    return () => {
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (engineRef.current && data) {
      engineRef.current.updateTelemetry(data);
    }
  }, [data]);

  const getFdirColor = (mode?: FdirMode) => {
    switch (mode) {
      case FdirMode.Nominal: return 'var(--color-nominal)';
      case FdirMode.Degraded: return 'var(--color-degraded)';
      case FdirMode.Safe: return 'var(--color-safe)';
      case FdirMode.Recovery: return 'var(--color-nominal)';
      default: return 'var(--color-text-secondary)';
    }
  };

  const getFdirLabel = (mode?: FdirMode) => {
    switch (mode) {
      case FdirMode.Nominal: return 'NOMINAL';
      case FdirMode.Degraded: return 'DEGRADED';
      case FdirMode.Safe: return 'SAFE';
      case FdirMode.Recovery: return 'RECOVERY';
      default: return 'UNKNOWN';
    }
  };

  const fdirLabel = getFdirLabel(data?.fdirMode);
  const isAlert = data?.fdirMode === FdirMode.Safe || data?.fdirMode === FdirMode.Degraded;
  const isDead = data?.isDone === true;
  const lifetime = data ? formatLifetime(data.simTimeS) : '---';

  return (
    <>
      <canvas ref={canvasRef} id="canvas-container" />

      <div className="hud-overlay">

        {/* ── Top Bar ── */}
        <div className="hud-header">
          {/* Left: Title + Connection */}
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
            <div className="glass-panel" style={{ padding: '1rem 1.25rem', minWidth: '16rem' }}>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '0.25rem' }} className="text-gradient">
                S-MAS OPS
              </h1>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  backgroundColor: connected ? '#22C55E' : '#EF4444',
                  display: 'inline-block'
                }} />
                {connected ? 'LIVE TELEMETRY' : 'DISCONNECTED'}
              </div>
            </div>

            {/* Lifetime Counter */}
            <div className="glass-panel" style={{
              padding: '1rem 1.25rem',
              minWidth: '14rem',
              borderColor: isDead ? 'var(--color-safe)' : 'rgba(59, 130, 246, 0.3)'
            }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', marginBottom: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                {isDead ? <Skull size={12} style={{ color: '#EF4444' }} /> : <Clock size={12} />}
                {isDead ? 'SATELLITE DECEASED' : 'MISSION LIFETIME'}
              </div>
              <div style={{
                fontSize: '1.5rem', fontWeight: 700, fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                letterSpacing: '0.05em',
                color: isDead ? '#EF4444' : '#60A5FA',
              }}>
                {lifetime}
              </div>
              {isDead && data && (
                <div style={{ fontSize: '0.7rem', color: '#F87171', marginTop: '0.2rem' }}>
                  Cause: {data.doneReason === 1 ? 'BATTERY DEPLETED' : data.doneReason === 2 ? 'COMMS LOST >72h' : data.doneReason === 3 ? 'REENTRY <200km' : data.doneReason === 4 ? 'SEU FATAL (RADIATION)' : data.doneReason === 5 ? 'FUEL DEPLETED' : `CODE ${data.doneReason}`}
                </div>
              )}
            </div>
          </div>

          {/* Right: System State */}
          <div className={`glass-panel ${isAlert ? 'pulse-alert' : ''}`}
            style={{
              padding: '1rem 1.25rem',
              display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
              borderColor: getFdirColor(data?.fdirMode),
              maxWidth: '14rem'
            }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', marginBottom: '0.25rem' }}>
              SYSTEM STATE
            </div>
            <div style={{
              fontSize: '1.75rem', fontWeight: 700, letterSpacing: '0.15em',
              color: getFdirColor(data?.fdirMode)
            }}>
              {fdirLabel}
            </div>
            {data?.seuActive && (
              <div style={{ fontSize: '0.7rem', color: '#F87171', marginTop: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <ShieldAlert size={12} /> SEU DETECTED
              </div>
            )}
          </div>
        </div>

        {/* ── Bottom Stats ── */}
        <div className="hud-bottom" style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', width: '100%' }}>

          {/* Main Stats Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', width: '18rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div className="stat-row">
              <div className="stat-label"><Orbit size={18} /> Altitude</div>
              <div className="stat-value">{data ? data.altitudeKm.toFixed(1) : '---'} km</div>
            </div>
            <div className="stat-row">
              <div className="stat-label"><Battery size={18} /> SoC</div>
              <div className="stat-value" style={{ color: data && data.batterySoc < 0.2 ? 'var(--color-safe)' : 'white' }}>
                {data ? (data.batterySoc * 100).toFixed(1) : '---'}%
              </div>
            </div>
            <div className="stat-row">
              <div className="stat-label"><Activity size={18} /> Velocity</div>
              <div className="stat-value" style={{ fontSize: '1.10rem' }}>
                7.60 km/s
              </div>
            </div>
          </div>

          {/* Propellant Panel (Fuel Gauge) */}
          <div className="glass-panel" style={{ padding: '1.25rem', width: '10rem', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <CircularProgress value={data ? data.fuelFraction : 1.0} label="PROPELLANT" />
          </div>

          {/* Thermal Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', width: '17rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
              <h3 style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Thermometer size={15} /> THERMAL
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{
                  width: 7, height: 7, borderRadius: '50%',
                  backgroundColor: data?.heaterOn ? '#22C55E' : '#4B5563',
                  boxShadow: data?.heaterOn ? '0 0 6px #22C55E' : 'none',
                  display: 'inline-block'
                }} />
                <span style={{ fontSize: '0.6rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace' }}>HEATER</span>
              </div>
            </div>
            <TemperatureBar label="Bus Temp" value={data ? data.tempBus : 20} safeMin={-20} safeMax={50} />
            <TemperatureBar label="Battery Temp" value={data ? data.tempBattery : 15} safeMin={-10} safeMax={45} />
            <TemperatureBar label="Payload Temp" value={data ? data.tempPayload : 10} safeMin={-15} safeMax={40} />
          </div>

          {/* AI / Action Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', flex: '1', minWidth: '20rem', maxWidth: '26rem' }}>
            <h3 style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Cpu size={16} /> AGENT ACTIONS
            </h3>
            <div className="action-grid">
              <div className="action-cell">
                <span className="label">Thrust Mag</span>
                <span className="value">{data ? data.throttle.toFixed(2) : '0.00'}</span>
              </div>
              <div className="action-cell">
                <span className="label">Payload</span>
                <span className="value" style={{ color: data?.payloadOn ? '#34D399' : 'var(--color-text-muted)' }}>
                  {data?.payloadOn ? 'ACTIVE' : 'STANDBY'}
                </span>
              </div>
              <div className="action-cell">
                <span className="label">Power Draw</span>
                <span className="value">{data ? data.powerDrawW.toFixed(0) : '0'} W</span>
              </div>
            </div>
          </div>

          {/* Environment Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem', justifyContent: 'center' }}>
            <div className="env-row">
              <ThermometerSun size={20} style={{ color: data?.inEclipse ? 'var(--color-text-muted)' : '#FACC15' }} />
              <span>{data?.inEclipse ? 'ECLIPSE' : 'SUNLIGHT'}</span>
            </div>
            <div className="env-row">
              <Radio size={20} style={{ color: data?.gsVisible ? '#34D399' : 'var(--color-text-muted)' }} />
              <span>{data?.gsVisible ? 'COMMS UP' : 'LOS'}</span>
            </div>
          </div>

        </div>
      </div>
      <GroundControlPanel sendCommand={sendCommand} connected={connected} />
      <MusicPlayer />
    </>
  );
}
