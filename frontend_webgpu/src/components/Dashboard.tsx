import { useEffect, useRef } from 'react';
import { useTelemetry, FdirMode } from '../lib/telemetry';
import { Activity, Battery, ShieldAlert, Cpu, Radio, Orbit, ThermometerSun, Clock, Skull } from 'lucide-react';
import Renderer from '../engine/Renderer';
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

export default function Dashboard() {
  const { data, connected } = useTelemetry();
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
    switch(mode) {
      case FdirMode.Nominal: return 'var(--color-nominal)';
      case FdirMode.Degraded: return 'var(--color-degraded)';
      case FdirMode.Safe: return 'var(--color-safe)';
      case FdirMode.Recovery: return 'var(--color-nominal)';
      default: return 'var(--color-text-secondary)';
    }
  };

  const getFdirLabel = (mode?: FdirMode) => {
    switch(mode) {
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
                  Cause: {data.doneReason === 1 ? 'BATTERY DEPLETED' : data.doneReason === 2 ? 'COMMS LOST >72h' : data.doneReason === 3 ? 'REENTRY <200km' : data.doneReason === 4 ? 'SEU FATAL (RADIATION)' : `CODE ${data.doneReason}`}
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
                <ShieldAlert size={12}/> SEU DETECTED
              </div>
            )}
          </div>
        </div>

        {/* ── Bottom Stats ── */}
        <div className="hud-bottom">
          
          {/* Main Stats Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', width: '20rem' }}>
            <div className="stat-row">
              <div className="stat-label"><Orbit size={18}/> Altitude</div>
              <div className="stat-value">{data ? data.altitudeKm.toFixed(1) : '---'} km</div>
            </div>
            <div className="stat-row">
              <div className="stat-label"><Battery size={18}/> SoC</div>
              <div className="stat-value" style={{ color: data && data.batterySoc < 0.2 ? 'var(--color-safe)' : 'white' }}>
                {data ? (data.batterySoc * 100).toFixed(1) : '---'}%
              </div>
            </div>
            <div className="stat-row">
              <div className="stat-label"><Activity size={18}/> Velocity</div>
              <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                7.60 km/s
              </div>
            </div>
          </div>

          {/* AI / Action Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', flex: '1', maxWidth: '28rem' }}>
            <h3 style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Cpu size={16}/> AGENT ACTIONS
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
          <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div className="env-row">
               <ThermometerSun size={20} style={{ color: data?.inEclipse ? 'var(--color-text-muted)' : '#FACC15' }}/>
               <span>{data?.inEclipse ? 'ECLIPSE' : 'SUNLIGHT'}</span>
            </div>
            <div className="env-row">
               <Radio size={20} style={{ color: data?.gsVisible ? '#34D399' : 'var(--color-text-muted)' }}/>
               <span>{data?.gsVisible ? 'COMMS UP' : 'LOS'}</span>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
