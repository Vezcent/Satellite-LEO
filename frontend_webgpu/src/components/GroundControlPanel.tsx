/*
 * S-MAS — Ground Control Panel (Developer Testbed)
 * Glassmorphism panel with two tabs: Manual Control & Environment Tuning.
 * Sends JSON commands to C# Controller via WebSocket.
 */
import { useState, useCallback } from 'react';
import type { GroundCommand } from '../lib/telemetry';
import './GroundControlPanel.css';

interface Props {
  sendCommand: (cmd: GroundCommand) => void;
  connected: boolean;
}

const FDIR_OPTIONS = [
  { value: -1, label: 'Auto (AI)' },
  { value: 0,  label: '🟢 Nominal' },
  { value: 1,  label: '🟡 Degraded' },
  { value: 2,  label: '🔴 Safe' },
  { value: 3,  label: '🔵 Recovery' },
];

export default function GroundControlPanel({ sendCommand, connected }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<'manual' | 'tuning'>('manual');

  // ── Manual Control State ──
  const [manualOverride, setManualOverride] = useState(false);
  const [thrustX, setThrustX] = useState(0);
  const [thrustY, setThrustY] = useState(0);
  const [thrustZ, setThrustZ] = useState(0);
  const [throttle, setThrottle] = useState(0);
  const [deepSleep, setDeepSleep] = useState(false);
  const [payloadOn, setPayloadOn] = useState(false);
  const [targetAlt, setTargetAlt] = useState(600);
  const [fdirMode, setFdirMode] = useState(-1);

  // ── Environment Tuning State ──
  const [seuMult, setSeuMult] = useState(1);
  const [noiseMult, setNoiseMult] = useState(1);
  const [driftMult, setDriftMult] = useState(1);
  const [densityMult, setDensityMult] = useState(0.01);
  const [envApplied, setEnvApplied] = useState(false);
  const [activePreset, setActivePreset] = useState<string>('nominal');

  // ── Handlers ──
  const handleManualToggle = useCallback(() => {
    const next = !manualOverride;
    setManualOverride(next);
    sendCommand({
      type: 'manual_override',
      manualOverride: next,
      action: next ? {
        thrustX, thrustY, thrustZ,
        throttle, deepSleep, payloadOn
      } : undefined,
    });
  }, [manualOverride, thrustX, thrustY, thrustZ, throttle, deepSleep, payloadOn, sendCommand]);

  const handleManualUpdate = useCallback(() => {
    if (!manualOverride) return;
    sendCommand({
      type: 'manual_override',
      manualOverride: true,
      action: { thrustX, thrustY, thrustZ, throttle, deepSleep, payloadOn },
    });
  }, [manualOverride, thrustX, thrustY, thrustZ, throttle, deepSleep, payloadOn, sendCommand]);

  const handleInjectSeu = useCallback(() => {
    sendCommand({ type: 'inject_seu' });
  }, [sendCommand]);

  const handleFdirChange = useCallback((mode: number) => {
    setFdirMode(mode);
    sendCommand({ type: 'force_fdir', fdirMode: mode });
  }, [sendCommand]);

  const handleTargetAlt = useCallback((val: number) => {
    setTargetAlt(val);
    sendCommand({ type: 'target_altitude', targetAltitudeKm: val });
  }, [sendCommand]);

  const handleEnvApply = useCallback(() => {
    sendCommand({
      type: 'environment_tuning',
      environment: {
        seuMultiplier: seuMult,
        noiseMultiplier: noiseMult,
        driftMultiplier: driftMult,
        densityMultiplier: densityMult,
      },
    });
    setEnvApplied(true);
    setActivePreset('custom');
    setTimeout(() => setEnvApplied(false), 2000);
  }, [seuMult, noiseMult, driftMult, densityMult, sendCommand]);

  const applyPreset = useCallback((preset: 'nominal' | 'storm' | 'worst' | 'solarmax' | 'halloween' | 'fuel_critical' | 'cold_eclipse') => {
    let s = 1, n = 1, d = 1, dens = 0.01;
    switch (preset) {
      case 'nominal': break;
      case 'storm':  s = 50; dens = 0.1; break;
      case 'worst':  s = 100; n = 3; d = 5; dens = 0.5; break;
      case 'solarmax': s = 5; n = 1.2; d = 1.5; dens = 0.3; break;
      case 'halloween': s = 100; n = 2.0; d = 1.2; dens = 0.15; break;
      case 'fuel_critical': s = 1; n = 1; d = 1; dens = 0.01; break;
      case 'cold_eclipse': s = 1; n = 1.2; d = 1; dens = 0.01; break;
    }
    setSeuMult(s); setNoiseMult(n); setDriftMult(d); setDensityMult(dens);
    setActivePreset(preset);
    
    sendCommand({
      type: 'environment_tuning',
      environment: {
        seuMultiplier: s,
        noiseMultiplier: n,
        driftMultiplier: d,
        densityMultiplier: dens,
      },
    });

    if (preset === 'solarmax' || preset === 'halloween' || preset === 'fuel_critical' || preset === 'cold_eclipse') {
      sendCommand({
        type: 'preset',
        presetName: preset
      } as any);
    }

    setEnvApplied(true);
    setTimeout(() => setEnvApplied(false), 2000);
  }, [sendCommand]);

  const getSeuLabel = (val: number) => {
    if (val <= 0.5) return 'Quiet';
    if (val <= 2) return 'Normal';
    if (val <= 20) return 'Storm';
    return 'Extreme';
  };

  if (collapsed) {
    return (
      <button className="gcp-toggle" onClick={() => setCollapsed(false)} title="Open Ground Control">
        ⚙️ GCP
      </button>
    );
  }

  return (
    <div className="gcp-panel">
      {/* Header */}
      <div className="gcp-header">
        <span className="gcp-title">⚙️ Ground Control</span>
        <div className="gcp-header-actions">
          <span className={`gcp-status ${connected ? 'online' : 'offline'}`}>
            {connected ? '● ONLINE' : '○ OFFLINE'}
          </span>
          <button className="gcp-close" onClick={() => setCollapsed(true)}>✕</button>
        </div>
      </div>

      {/* Tabs */}
      <div className="gcp-tabs">
        <button
          className={`gcp-tab ${activeTab === 'manual' ? 'active' : ''}`}
          onClick={() => setActiveTab('manual')}
        >
          🎮 Manual
        </button>
        <button
          className={`gcp-tab ${activeTab === 'tuning' ? 'active' : ''}`}
          onClick={() => setActiveTab('tuning')}
        >
          🔬 Stress Test
        </button>
      </div>

      {/* Tab Content */}
      <div className="gcp-body">
        {activeTab === 'manual' ? (
          <>
            {/* Manual Override Toggle */}
            <div className="gcp-section">
              <label className="gcp-toggle-label">
                <input
                  type="checkbox"
                  checked={manualOverride}
                  onChange={handleManualToggle}
                />
                <span className={`gcp-toggle-text ${manualOverride ? 'active' : ''}`}>
                  Manual Override {manualOverride ? 'ON' : 'OFF'}
                </span>
              </label>
            </div>

            {manualOverride && (
              <div className="gcp-section gcp-manual-controls">
                <div className="gcp-slider-group">
                  <label>Thrust X <span className="gcp-val">{thrustX.toFixed(2)}</span></label>
                  <input type="range" min={-1} max={1} step={0.01} value={thrustX}
                    onChange={e => { setThrustX(+e.target.value); }}
                    onMouseUp={handleManualUpdate} />
                </div>
                <div className="gcp-slider-group">
                  <label>Thrust Y <span className="gcp-val">{thrustY.toFixed(2)}</span></label>
                  <input type="range" min={-1} max={1} step={0.01} value={thrustY}
                    onChange={e => { setThrustY(+e.target.value); }}
                    onMouseUp={handleManualUpdate} />
                </div>
                <div className="gcp-slider-group">
                  <label>Thrust Z <span className="gcp-val">{thrustZ.toFixed(2)}</span></label>
                  <input type="range" min={-1} max={1} step={0.01} value={thrustZ}
                    onChange={e => { setThrustZ(+e.target.value); }}
                    onMouseUp={handleManualUpdate} />
                </div>
                <div className="gcp-slider-group">
                  <label>Throttle <span className="gcp-val">{(throttle * 100).toFixed(0)}%</span></label>
                  <input type="range" min={0} max={1} step={0.01} value={throttle}
                    onChange={e => { setThrottle(+e.target.value); }}
                    onMouseUp={handleManualUpdate} />
                </div>
                <div className="gcp-toggle-row">
                  <label>
                    <input type="checkbox" checked={deepSleep}
                      onChange={e => { setDeepSleep(e.target.checked); handleManualUpdate(); }} />
                    Deep Sleep
                  </label>
                  <label>
                    <input type="checkbox" checked={payloadOn}
                      onChange={e => { setPayloadOn(e.target.checked); handleManualUpdate(); }} />
                    Payload
                  </label>
                </div>
              </div>
            )}

            {/* Target Altitude */}
            <div className="gcp-section">
              <div className="gcp-slider-group">
                <label>Target Altitude <span className="gcp-val">{targetAlt} km</span></label>
                <input type="range" min={500} max={700} step={1} value={targetAlt}
                  onChange={e => handleTargetAlt(+e.target.value)} />
              </div>
            </div>

            {/* SEU Injection */}
            <div className="gcp-section">
              <button className="gcp-btn gcp-btn-danger" onClick={handleInjectSeu}>
                ⚡ Inject SEU
              </button>
            </div>

            {/* FDIR Override */}
            <div className="gcp-section">
              <label>FDIR Override</label>
              <select
                className="gcp-select"
                value={fdirMode}
                onChange={e => handleFdirChange(+e.target.value)}
              >
                {FDIR_OPTIONS.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          </>
        ) : (
          <>
            {/* Active Environment Badge */}
            <div className="gcp-section">
              <div className={`gcp-env-badge ${envApplied ? 'gcp-env-flash' : ''} ${
                activePreset === 'worst' || activePreset === 'halloween' ? 'gcp-env-danger' :
                activePreset === 'storm' || activePreset === 'solarmax' || activePreset === 'fuel_critical' ? 'gcp-env-warn' : 'gcp-env-ok'
              }`}>
                <span className="gcp-env-dot" />
                <span>{envApplied ? '✓ APPLIED' :
                  activePreset === 'worst' ? '⚠ WORST CASE ACTIVE' :
                  activePreset === 'storm' ? '⚠ SOLAR STORM ACTIVE' :
                  activePreset === 'solarmax' ? '⚠ SOLAR MAX ACTIVE' :
                  activePreset === 'halloween' ? '⚠ HALLOWEEN STORM ACTIVE' :
                  activePreset === 'fuel_critical' ? '⚠ FUEL CRITICAL ACTIVE' :
                  activePreset === 'cold_eclipse' ? '⚠ COLD ECLIPSE ACTIVE' :
                  activePreset === 'custom' ? '⚙ CUSTOM ENV ACTIVE' :
                  '● NOMINAL'}</span>
              </div>
            </div>
            {/* Presets Grid */}
            <div className="gcp-section gcp-presets" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.4rem' }}>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'nominal' ? 'active' : ''}`} onClick={() => applyPreset('nominal')}>
                🟢 Nominal
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'storm' ? 'active' : ''}`} onClick={() => applyPreset('storm')}>
                🟡 Solar Storm
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'worst' ? 'active' : ''}`} onClick={() => applyPreset('worst')}>
                🔴 Worst Case
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'solarmax' ? 'active' : ''}`} onClick={() => applyPreset('solarmax')}>
                🔥 Solar Max
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'halloween' ? 'active' : ''}`} onClick={() => applyPreset('halloween')}>
                🎃 Halloween
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'fuel_critical' ? 'active' : ''}`} onClick={() => applyPreset('fuel_critical')}>
                ⛽ Fuel Critical
              </button>
              <button className={`gcp-btn gcp-btn-preset ${activePreset === 'cold_eclipse' ? 'active' : ''}`} onClick={() => applyPreset('cold_eclipse')}>
                ❄️ Cold Eclipse
              </button>
            </div>

            {/* SEU Multiplier */}
            <div className="gcp-section">
              <div className="gcp-slider-group">
                <label>SEU Rate <span className="gcp-val">{seuMult}x ({getSeuLabel(seuMult)})</span></label>
                <input type="range" min={0.1} max={100} step={0.1} value={seuMult}
                  onChange={e => setSeuMult(+e.target.value)} />
              </div>
            </div>

            {/* Noise Multiplier */}
            <div className="gcp-section">
              <div className="gcp-slider-group">
                <label>Sensor Noise <span className="gcp-val">{noiseMult.toFixed(1)}x</span></label>
                <input type="range" min={0} max={5} step={0.1} value={noiseMult}
                  onChange={e => setNoiseMult(+e.target.value)} />
              </div>
            </div>

            {/* Drift Multiplier */}
            <div className="gcp-section">
              <div className="gcp-slider-group">
                <label>Aging Rate <span className="gcp-val">{driftMult.toFixed(1)}x</span></label>
                <input type="range" min={0} max={10} step={0.1} value={driftMult}
                  onChange={e => setDriftMult(+e.target.value)} />
              </div>
            </div>

            {/* Density Multiplier */}
            <div className="gcp-section">
              <div className="gcp-slider-group">
                <label>Atm Density <span className="gcp-val">{densityMult.toFixed(3)}</span></label>
                <input type="range" min={0.001} max={1} step={0.001} value={densityMult}
                  onChange={e => setDensityMult(+e.target.value)} />
              </div>
            </div>

            {/* Apply */}
            <div className="gcp-section">
              <button className="gcp-btn gcp-btn-apply" onClick={handleEnvApply}>
                🚀 Apply Environment
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
