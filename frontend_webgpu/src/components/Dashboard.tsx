import { useEffect, useRef } from 'react';
import { useTelemetry, FdirMode } from '../lib/telemetry';
import type { TelemetryData } from '../lib/telemetry';
import { Battery, ShieldAlert, Radio, Orbit, ThermometerSun, Skull, Database } from 'lucide-react';
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
function CircularProgress({ value, label, size = 56, strokeWidth = 4 }: { value: number; label: string; size?: number; strokeWidth?: number }) {
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem' }}>
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
            <span style={{ fontSize: '0.5rem', fontWeight: 700, color: '#EF4444', letterSpacing: '0.05em' }}>EMPTY</span>
          ) : (
            <>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'white' }}>
                {(value * 100).toFixed(0)}
              </span>
              <span style={{ fontSize: '0.5rem', color: 'var(--color-text-secondary)' }}>%</span>
            </>
          )}
        </div>
      </div>
      <span style={{ fontSize: '0.55rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontFamily: 'monospace' }}>
        <span style={{ color: 'var(--color-text-secondary)' }}>{label}</span>
        <span style={{ color: isSafe ? 'white' : barColor, fontWeight: 600 }}>{value.toFixed(1)}°C</span>
      </div>
      <div style={{ position: 'relative', height: '4px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '2px', overflow: 'hidden' }}>
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
          borderRadius: '2px',
          boxShadow: `0 0 4px ${glowColor}`,
          transition: 'width 0.3s ease-in-out'
        }} />
      </div>
    </div>
  );
}

const SAT_NAMES = ['ALPHA', 'BETA', 'GAMMA', 'DELTA'];
const SAT_COLORS = ['#3B82F6', '#22C55E', '#F59E0B', '#A855F7'];
const SAT_GLOWS = [
  'rgba(59, 130, 246, 0.2)',
  'rgba(34, 197, 94, 0.2)',
  'rgba(245, 158, 11, 0.2)',
  'rgba(168, 85, 247, 0.2)'
];

function SatelliteCard({ satId, data }: { satId: number; data: TelemetryData | null }) {
  const color = SAT_COLORS[satId];
  const glow = SAT_GLOWS[satId];
  const name = SAT_NAMES[satId];

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

  if (!data) {
    return (
      <div className="glass-panel sat-card" style={{ borderTop: `3px solid ${color}`, boxShadow: `0 8px 32px 0 rgba(0,0,0,0.3), 0 0 10px ${glow}` }}>
        <div className="sat-card-header">
          <h2 style={{ fontSize: '0.9rem', fontWeight: 700, color: color, letterSpacing: '0.05em' }}>
            SAT-{satId} — {name}
          </h2>
          <span style={{ fontSize: '0.65rem', color: 'var(--color-text-muted)', fontFamily: 'monospace' }}>AWAITING UPLINK</span>
        </div>
        <div style={{ display: 'flex', flexGrow: 1, alignItems: 'center', justifyContent: 'center', minHeight: '120px' }}>
          <div className="uplink-scanner" style={{ color }} />
        </div>
      </div>
    );
  }

  const isDead = data.isDone;
  const isAlert = data.fdirMode === FdirMode.Safe || data.fdirMode === FdirMode.Degraded || data.conjunctionRisk > 0.5;
  const lifetime = formatLifetime(data.simTimeS);
  const fdirLabel = getFdirLabel(data.fdirMode);

  return (
    <div className={`glass-panel sat-card ${isAlert ? 'pulse-alert' : ''}`}
         style={{ 
           borderTop: `3px solid ${isDead ? '#4B5563' : color}`,
           boxShadow: isDead ? 'none' : `0 8px 32px 0 rgba(0,0,0,0.3), 0 0 10px ${glow}`
         }}>
      
      {isDead && (
        <div className="sat-dead-overlay">
          <Skull size={24} style={{ color: '#EF4444', marginBottom: '0.5rem' }} />
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#EF4444', letterSpacing: '0.05em' }}>MISSION TERMINATED</span>
          <span style={{ fontSize: '0.65rem', color: '#F87171', marginTop: '0.25rem', fontFamily: 'monospace' }}>
            {data.doneReason === 1 ? 'BATTERY DEPLETED' : 
             data.doneReason === 2 ? 'COMMS LOST >72h' : 
             data.doneReason === 3 ? 'REENTRY <200km' : 
             data.doneReason === 4 ? 'SEU FATAL (RADIATION)' : 
             data.doneReason === 5 ? 'FUEL DEPLETED' : `CODE ${data.doneReason}`}
          </span>
        </div>
      )}

      {/* Header */}
      <div className="sat-card-header">
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '0.9rem', fontWeight: 700, color: isDead ? '#6b7280' : color, letterSpacing: '0.05em' }}>
            SAT-{satId} — {name}
          </h2>
          <span style={{ fontSize: '0.6rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace' }}>
            LIFETIME: {lifetime}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.6rem',
            fontWeight: 700,
            letterSpacing: '0.05em',
            backgroundColor: isDead ? 'rgba(75,85,99,0.2)' : `${getFdirColor(data.fdirMode)}22`,
            color: isDead ? '#9ca3af' : getFdirColor(data.fdirMode),
            border: `1px solid ${isDead ? '#4b5563' : getFdirColor(data.fdirMode)}33`
          }}>
            {isDead ? 'DECEASED' : fdirLabel}
          </span>
        </div>
      </div>

      {/* Debris Conjunction Warning Banner */}
      {!isDead && data.conjunctionRisk > 0.05 && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          borderRadius: '4px',
          padding: '0.25rem 0.5rem',
          margin: '0.25rem 0.75rem 0 0.75rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          color: '#EF4444',
          fontSize: '0.6rem',
          fontFamily: 'monospace',
          fontWeight: 700,
          letterSpacing: '0.05em'
        }}>
          <ShieldAlert size={12} className="pulse-alert" />
          <span>COLLISION RISK: {(data.conjunctionRisk * 100).toFixed(0)}% (TCA: {data.timeToTcaS.toFixed(0)}s)</span>
        </div>
      )}

      {/* Body */}
      <div className="sat-card-body" style={{ opacity: isDead ? 0.25 : 1 }}>
        
        {/* Left Column: Core Stats */}
        <div className="sat-card-left">
          <div className="stat-row" style={{ paddingBottom: '0.2rem' }}>
            <div className="stat-label" style={{ fontSize: '0.7rem' }}><Orbit size={13} /> Alt</div>
            <div className="stat-value" style={{ fontSize: '0.9rem' }}>{data.altitudeKm.toFixed(1)} km</div>
          </div>
          <div className="stat-row" style={{ paddingBottom: '0.2rem' }}>
            <div className="stat-label" style={{ fontSize: '0.7rem' }}><Battery size={13} /> SoC</div>
            <div className="stat-value" style={{ fontSize: '0.9rem', color: data.batterySoc < 0.2 ? 'var(--color-safe)' : 'white' }}>
              {(data.batterySoc * 100).toFixed(1)}%
            </div>
          </div>
          <div className="stat-row" style={{ paddingBottom: '0.2rem' }}>
            <div className="stat-label" style={{ fontSize: '0.7rem' }}><Database size={13} /> Buffer</div>
            <div className="stat-value" style={{ fontSize: '0.9rem' }}>
              {data.dataBufferMb !== undefined ? `${data.dataBufferMb.toFixed(1)} MB` : '0.0 MB'}
            </div>
          </div>
          <div className="stat-row" style={{ paddingBottom: '0.2rem' }}>
            <div className="stat-label" style={{ fontSize: '0.7rem' }}><Radio size={13} /> SNR</div>
            <div className="stat-value" style={{ fontSize: '0.9rem', color: data.gsVisible ? '#34D399' : 'var(--color-text-muted)' }}>
              {data.snrDb !== undefined && data.snrDb > -900 ? `${data.snrDb.toFixed(1)} dB` : 'LOS'}
            </div>
          </div>
        </div>

        {/* Middle Column: Propellant & Env */}
        <div className="sat-card-middle">
          <CircularProgress value={data.fuelFraction} label="Fuel" size={56} strokeWidth={4} />
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem' }}>
            <span title={data.inEclipse ? 'Eclipse' : 'Sunlight'}><ThermometerSun size={14} style={{ color: data.inEclipse ? 'var(--color-text-muted)' : '#FACC15' }} /></span>
            <span title={data.gsVisible ? 'Ground Station Contact' : 'LOS'}><Radio size={14} style={{ color: data.gsVisible ? '#34D399' : 'var(--color-text-muted)' }} /></span>
            {data.seuActive && <span title="SEU Detected"><ShieldAlert size={14} style={{ color: '#EF4444' }} /></span>}
          </div>
        </div>

        {/* Right Column: Thermal & Actions */}
        <div className="sat-card-right">
          {/* Thermal */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.6rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', fontWeight: 600 }}>THERMAL</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
              <span style={{
                width: 5, height: 5, borderRadius: '50%',
                backgroundColor: data.heaterOn ? '#22C55E' : '#4B5563',
                boxShadow: data.heaterOn ? '0 0 4px #22C55E' : 'none',
                display: 'inline-block'
              }} />
              <span style={{ fontSize: '0.5rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace' }}>HTR</span>
            </div>
          </div>
          <TemperatureBar label="Bus" value={data.tempBus} safeMin={-20} safeMax={50} />
          <TemperatureBar label="Bat" value={data.tempBattery} safeMin={-10} safeMax={45} />
          
          {/* Actions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem', marginTop: '0.2rem' }}>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '2px 4px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.03)', display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.45rem', color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Thrust</span>
              <span style={{ fontSize: '0.65rem', fontFamily: 'monospace', fontWeight: 700 }}>{data.throttle.toFixed(2)}</span>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '2px 4px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.03)', display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.45rem', color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Payload</span>
              <span style={{ fontSize: '0.65rem', fontFamily: 'monospace', fontWeight: 700, color: data.payloadOn ? '#34D399' : 'var(--color-text-muted)' }}>
                {data.payloadOn ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default function Dashboard() {
  const { satellites, connected, sendCommand } = useTelemetry();
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
    if (engineRef.current) {
      satellites.forEach(sat => {
        if (sat) {
          engineRef.current?.updateTelemetry(sat);
        }
      });
    }
  }, [satellites]);

  const activeCount = satellites.filter(s => s !== null && !s.isDone).length;
  const firstActiveSat = satellites.find(s => s !== null);
  const lifetime = firstActiveSat ? formatLifetime(firstActiveSat.simTimeS) : '---';

  return (
    <>
      <canvas ref={canvasRef} id="canvas-container" />

      {/* ── Top Bar — fixed at top, compact ── */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, padding: '0.5rem 1rem', zIndex: 10, pointerEvents: 'none' }}>
        <div className="glass-panel" style={{ padding: '0.35rem 1rem', display: 'inline-flex', alignItems: 'center', gap: '1.5rem', pointerEvents: 'auto', fontSize: '0.7rem', fontFamily: 'monospace' }}>
          <h1 style={{ fontSize: '0.85rem', fontWeight: 700, letterSpacing: '0.08em', whiteSpace: 'nowrap' }} className="text-gradient">
            S-MAS OPS
          </h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: 'var(--color-text-secondary)' }}>
            <span style={{
              width: 5, height: 5, borderRadius: '50%',
              backgroundColor: connected ? '#22C55E' : '#EF4444',
              display: 'inline-block'
            }} />
            {connected ? 'LIVE' : 'OFFLINE'}
          </div>
          <div>
            <span style={{ color: 'var(--color-text-secondary)' }}>SAT </span>
            <span style={{ color: activeCount > 0 ? '#34D399' : '#EF4444', fontWeight: 700 }}>
              {activeCount}/4
            </span>
          </div>
          <div>
            <span style={{ color: 'var(--color-text-secondary)' }}>TIME </span>
            <span style={{ color: '#60A5FA', fontWeight: 700 }}>
              {lifetime}
            </span>
          </div>
        </div>
      </div>

      {/* ── Bottom Strip — satellite cards ── */}
      <div className="hud-overlay">
        <div className="constellation-grid">
          {[0, 1, 2, 3].map(id => (
            <SatelliteCard key={id} satId={id} data={satellites[id]} />
          ))}
        </div>
      </div>

      <GroundControlPanel sendCommand={sendCommand} connected={connected} />
      <MusicPlayer />
    </>
  );
}
