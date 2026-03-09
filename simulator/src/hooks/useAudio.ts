import { useCallback, useRef, useState } from "react";

export interface AudioHandle {
  loaded: boolean;
  playing: boolean;
  duration: number;
  currentTime: number;
  waveformData: Float32Array;
  loadFile: (file: File) => Promise<{ filename: string; duration: number }>;
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  analyserNode: AnalyserNode | null;
}

const FFT_SIZE = 2048;

export function useAudio(): AudioHandle {
  const [loaded, setLoaded] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const bufferRef = useRef<AudioBuffer | null>(null);
  const startTimeRef = useRef(0);
  const offsetRef = useRef(0);
  const waveformRef = useRef(new Float32Array(FFT_SIZE / 2) as Float32Array<ArrayBuffer>);
  const rafRef = useRef<number>(0);

  const getContext = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext();
      analyserRef.current = ctxRef.current.createAnalyser();
      analyserRef.current.fftSize = FFT_SIZE;
      analyserRef.current.connect(ctxRef.current.destination);
    }
    return ctxRef.current;
  }, []);

  const stopSource = useCallback(() => {
    if (sourceRef.current) {
      try {
        sourceRef.current.stop();
      } catch {
        // Already stopped
      }
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    cancelAnimationFrame(rafRef.current);
  }, []);

  const updateTime = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx || !playing) return;
    const elapsed = ctx.currentTime - startTimeRef.current + offsetRef.current;
    setCurrentTime(Math.min(elapsed, duration));

    if (analyserRef.current) {
      analyserRef.current.getFloatFrequencyData(waveformRef.current);
    }

    rafRef.current = requestAnimationFrame(updateTime);
  }, [playing, duration]);

  const loadFile = useCallback(
    async (file: File): Promise<{ filename: string; duration: number }> => {
      const ctx = getContext();
      stopSource();

      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      bufferRef.current = audioBuffer;
      setDuration(audioBuffer.duration);
      setCurrentTime(0);
      setLoaded(true);
      setPlaying(false);
      offsetRef.current = 0;

      return { filename: file.name, duration: audioBuffer.duration };
    },
    [getContext, stopSource],
  );

  const play = useCallback(() => {
    const ctx = getContext();
    const buffer = bufferRef.current;
    if (!buffer || !analyserRef.current) return;

    stopSource();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyserRef.current);
    source.start(0, offsetRef.current);
    sourceRef.current = source;
    startTimeRef.current = ctx.currentTime;
    setPlaying(true);

    source.onended = () => {
      setPlaying(false);
      offsetRef.current = 0;
      setCurrentTime(0);
    };

    const tick = () => {
      const elapsed = ctx.currentTime - startTimeRef.current + offsetRef.current;
      setCurrentTime(Math.min(elapsed, buffer.duration));
      if (analyserRef.current) {
        analyserRef.current.getFloatFrequencyData(waveformRef.current);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [getContext, stopSource]);

  const pause = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    offsetRef.current += ctx.currentTime - startTimeRef.current;
    stopSource();
    setPlaying(false);
  }, [stopSource]);

  const seek = useCallback(
    (time: number) => {
      offsetRef.current = time;
      setCurrentTime(time);
      if (playing) {
        play();
      }
    },
    [playing, play],
  );

  return {
    loaded,
    playing,
    duration,
    currentTime,
    waveformData: waveformRef.current,
    loadFile,
    play,
    pause,
    seek,
    analyserNode: analyserRef.current,
  };
}
