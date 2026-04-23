import { useEffect, useRef } from 'react';
import { useTelemetry, FdirMode } from '../lib/telemetry';
import { Activity, Battery, ShieldAlert, Cpu, Radio, Orbit, ThermometerSun } from 'lucide-react';
import Renderer from '../engine/Renderer';

export default function Dashboard() {
  const { data, connected } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Renderer | null>(null);

  useEffect(() => {
    if (canvasRef.current && !engineRef.current) {
      engineRef.current = new Renderer(canvasRef.current);
      engineRef.current.init().catch(console.error);
    }
    
    // Cleanup
    return () => {
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, []);

  // Update renderer with latest telemetry
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

  return (
    <>
      <canvas ref={canvasRef} id="canvas-container" />
      
      <div className="absolute inset-0 pointer-events-none p-6 flex flex-col justify-between" style={{ zIndex: 10 }}>
        
        {/* Top Bar */}
        <header className="flex justify-between items-start">
          <div className="glass-panel p-4 w-72 pointer-events-auto">
            <h1 className="text-xl font-bold tracking-wider text-gradient mb-1">S-MAS OPS</h1>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
              {connected ? 'LIVE TELEMETRY' : 'DISCONNECTED'}
            </div>
          </div>
          
          <div className={`glass-panel p-4 flex flex-col items-end pointer-events-auto ${isAlert ? 'pulse-alert' : ''}`} style={{ borderColor: getFdirColor(data?.fdirMode) }}>
            <div className="text-xs text-gray-400 font-mono mb-1">SYSTEM STATE</div>
            <div className="text-2xl font-bold tracking-widest" style={{ color: getFdirColor(data?.fdirMode) }}>
              {fdirLabel}
            </div>
            {data?.seuActive && <div className="text-xs text-red-400 mt-1 flex items-center gap-1"><ShieldAlert size={12}/> SEU DETECTED</div>}
          </div>
        </header>

        {/* Bottom / Side Stats */}
        <div className="flex gap-6 pointer-events-auto items-end">
          
          {/* Main Stats Panel */}
          <div className="glass-panel p-5 w-80 flex flex-col gap-4">
            
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <div className="flex items-center gap-2 text-gray-400"><Orbit size={18}/> Altitude</div>
              <div className="text-xl font-mono">{data ? data.altitudeKm.toFixed(1) : '---'} km</div>
            </div>
            
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <div className="flex items-center gap-2 text-gray-400"><Battery size={18}/> SoC</div>
              <div className="text-xl font-mono" style={{ color: data && data.batterySoc < 0.2 ? 'var(--color-safe)' : 'white' }}>
                {data ? (data.batterySoc * 100).toFixed(1) : '---'}%
              </div>
            </div>

            <div className="flex justify-between items-center pb-1">
              <div className="flex items-center gap-2 text-gray-400"><Activity size={18}/> Velocity</div>
              <div className="text-lg font-mono">7.6 km/s</div>
            </div>

          </div>

          {/* AI / Action Panel */}
          <div className="glass-panel p-5 flex flex-col gap-4 flex-1 max-w-md">
            <h3 className="text-sm text-gray-400 font-mono flex items-center gap-2 mb-2"><Cpu size={16}/> AGENT ACTIONS</h3>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="flex flex-col gap-1 bg-black/30 p-3 rounded border border-white/5">
                <span className="text-xs text-gray-500">THRUST MAG</span>
                <span className="font-mono text-lg">{data ? data.throttle.toFixed(2) : '0.00'}</span>
              </div>
              <div className="flex flex-col gap-1 bg-black/30 p-3 rounded border border-white/5">
                <span className="text-xs text-gray-500">PAYLOAD</span>
                <span className={`font-mono text-lg ${data?.payloadOn ? 'text-green-400' : 'text-gray-500'}`}>
                  {data?.payloadOn ? 'ACTIVE' : 'STANDBY'}
                </span>
              </div>
              <div className="flex flex-col gap-1 bg-black/30 p-3 rounded border border-white/5">
                <span className="text-xs text-gray-500">POWER DRAW</span>
                <span className="font-mono text-lg">{data ? data.powerDrawW.toFixed(0) : '0'} W</span>
              </div>
            </div>
          </div>
          
          {/* Environment Panel */}
          <div className="glass-panel p-5 flex flex-col gap-3">
            <div className="flex items-center gap-3">
               <ThermometerSun size={20} className={data?.inEclipse ? 'text-gray-500' : 'text-yellow-400'}/>
               <span className="font-mono">{data?.inEclipse ? 'ECLIPSE' : 'SUNLIGHT'}</span>
            </div>
            <div className="flex items-center gap-3">
               <Radio size={20} className={data?.gsVisible ? 'text-green-400' : 'text-gray-500'}/>
               <span className="font-mono">{data?.gsVisible ? 'COMMS UP' : 'LOS'}</span>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
