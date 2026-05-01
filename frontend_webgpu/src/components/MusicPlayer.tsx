import { useState, useEffect, useRef, useCallback } from 'react';
import { Music, Play, Pause, SkipForward, SkipBack, Volume2, VolumeX, ChevronUp, ChevronDown } from 'lucide-react';

export default function MusicPlayer() {
  const [tracks, setTracks] = useState<string[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [muted, setMuted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressTimerRef = useRef<number>(0);

  // Fetch track list from server
  useEffect(() => {
    fetch('/api/music-list')
      .then(r => r.json())
      .then((files: string[]) => setTracks(files))
      .catch(() => setTracks([]));
  }, []);

  // Setup audio element
  useEffect(() => {
    const audio = new Audio();
    audio.volume = volume;
    audio.addEventListener('ended', () => {
      // Auto-next
      setCurrentIdx(prev => (prev + 1) % tracks.length);
    });
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration);
    });
    audioRef.current = audio;
    return () => {
      audio.pause();
      audio.src = '';
    };
  }, []);

  // Update progress
  useEffect(() => {
    if (playing) {
      progressTimerRef.current = window.setInterval(() => {
        if (audioRef.current) {
          setProgress(audioRef.current.currentTime);
        }
      }, 250);
    } else {
      clearInterval(progressTimerRef.current);
    }
    return () => clearInterval(progressTimerRef.current);
  }, [playing]);

  // Load track when index changes
  useEffect(() => {
    if (tracks.length === 0 || !audioRef.current) return;
    const audio = audioRef.current;
    audio.src = `/music/${encodeURIComponent(tracks[currentIdx])}`;
    audio.load();
    setProgress(0);
    if (playing) {
      audio.play().catch(() => {});
    }
  }, [currentIdx, tracks]);

  // Volume sync
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = muted ? 0 : volume;
    }
  }, [volume, muted]);

  const togglePlay = useCallback(() => {
    if (!audioRef.current || tracks.length === 0) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play().then(() => setPlaying(true)).catch(() => {});
    }
  }, [playing, tracks]);

  const prevTrack = () => setCurrentIdx(prev => (prev - 1 + tracks.length) % tracks.length);
  const nextTrack = () => setCurrentIdx(prev => (prev + 1) % tracks.length);

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || duration === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    audioRef.current.currentTime = pct * duration;
    setProgress(pct * duration);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const trackName = tracks.length > 0
    ? tracks[currentIdx].replace(/\.[^.]+$/, '')
    : 'No tracks';

  // Don't render if no music files found
  if (tracks.length === 0) return null;

  return (
    <div style={{
      position: 'fixed',
      bottom: '1.5rem',
      right: '1.5rem',
      zIndex: 50,
      pointerEvents: 'auto',
    }}>
      {/* Expanded Panel */}
      {expanded && (
        <div style={{
          marginBottom: '0.5rem',
          background: 'rgba(10, 12, 20, 0.85)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          borderRadius: '12px',
          padding: '1rem',
          width: '280px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}>
          {/* Track Name */}
          <div style={{
            fontSize: '0.8rem',
            color: '#A5B4FC',
            marginBottom: '0.75rem',
            fontFamily: "'JetBrains Mono', monospace",
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            <Music size={12} style={{ display: 'inline', marginRight: '0.4rem', verticalAlign: 'middle' }} />
            {trackName}
          </div>

          {/* Progress Bar */}
          <div
            onClick={seek}
            style={{
              height: '4px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '2px',
              cursor: 'pointer',
              marginBottom: '0.5rem',
              position: 'relative',
            }}
          >
            <div style={{
              height: '100%',
              width: duration > 0 ? `${(progress / duration) * 100}%` : '0%',
              background: 'linear-gradient(90deg, #6366F1, #A855F7)',
              borderRadius: '2px',
              transition: 'width 0.25s linear',
            }} />
          </div>

          {/* Time */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.65rem',
            color: 'rgba(255,255,255,0.4)',
            fontFamily: 'monospace',
            marginBottom: '0.75rem',
          }}>
            <span>{formatTime(progress)}</span>
            <span>{formatTime(duration)}</span>
          </div>

          {/* Controls */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
          }}>
            <button onClick={prevTrack} style={btnStyle} title="Previous">
              <SkipBack size={16} />
            </button>
            <button onClick={togglePlay} style={{
              ...btnStyle,
              background: 'linear-gradient(135deg, #6366F1, #A855F7)',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
            }} title={playing ? 'Pause' : 'Play'}>
              {playing ? <Pause size={16} /> : <Play size={16} style={{ marginLeft: '2px' }} />}
            </button>
            <button onClick={nextTrack} style={btnStyle} title="Next">
              <SkipForward size={16} />
            </button>
          </div>

          {/* Volume */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginTop: '0.75rem',
          }}>
            <button onClick={() => setMuted(!muted)} style={btnStyle}>
              {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={muted ? 0 : volume}
              onChange={e => { setVolume(parseFloat(e.target.value)); setMuted(false); }}
              style={{
                flex: 1,
                height: '4px',
                appearance: 'none',
                background: 'rgba(255,255,255,0.15)',
                borderRadius: '2px',
                outline: 'none',
                cursor: 'pointer',
                accentColor: '#6366F1',
              }}
            />
          </div>

          {/* Track List */}
          {tracks.length > 1 && (
            <div style={{
              marginTop: '0.75rem',
              maxHeight: '120px',
              overflowY: 'auto',
              borderTop: '1px solid rgba(255,255,255,0.08)',
              paddingTop: '0.5rem',
            }}>
              {tracks.map((t, i) => (
                <div
                  key={t}
                  onClick={() => { setCurrentIdx(i); setPlaying(true); }}
                  style={{
                    padding: '0.3rem 0.5rem',
                    fontSize: '0.7rem',
                    color: i === currentIdx ? '#A855F7' : 'rgba(255,255,255,0.5)',
                    fontFamily: 'monospace',
                    cursor: 'pointer',
                    borderRadius: '4px',
                    background: i === currentIdx ? 'rgba(168,85,247,0.1)' : 'transparent',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {i === currentIdx && playing ? '♫ ' : ''}{t.replace(/\.[^.]+$/, '')}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Toggle Button */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          width: '44px',
          height: '44px',
          borderRadius: '50%',
          background: playing
            ? 'linear-gradient(135deg, #6366F1, #A855F7)'
            : 'rgba(10, 12, 20, 0.85)',
          border: `1px solid ${playing ? 'rgba(168,85,247,0.6)' : 'rgba(99,102,241,0.3)'}`,
          color: 'white',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backdropFilter: 'blur(12px)',
          boxShadow: playing
            ? '0 0 20px rgba(99,102,241,0.4)'
            : '0 4px 16px rgba(0,0,0,0.3)',
          transition: 'all 0.3s ease',
          float: 'right',
        }}
        title="Music Player"
      >
        {expanded ? <ChevronDown size={18} /> : <Music size={18} />}
      </button>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  color: 'rgba(255,255,255,0.7)',
  cursor: 'pointer',
  padding: '4px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  borderRadius: '4px',
  transition: 'color 0.2s',
};
