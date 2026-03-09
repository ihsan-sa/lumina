import { useEffect, useRef } from "react";

interface SpectrogramProps {
  analyserNode: AnalyserNode | null;
  playing: boolean;
}

const BAR_COUNT = 32;
const CANVAS_WIDTH = 300;
const CANVAS_HEIGHT = 80;

export function Spectrogram({ analyserNode, playing }: SpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const dataRef = useRef<Float32Array<ArrayBuffer> | null>(null);

  useEffect(() => {
    if (!analyserNode) return;
    dataRef.current = new Float32Array(analyserNode.frequencyBinCount) as Float32Array<ArrayBuffer>;
  }, [analyserNode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyserNode || !playing) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      if (!dataRef.current) return;
      analyserNode.getFloatFrequencyData(dataRef.current);

      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

      const binWidth = CANVAS_WIDTH / BAR_COUNT;
      const binsPerBar = Math.floor(dataRef.current.length / BAR_COUNT);

      for (let i = 0; i < BAR_COUNT; i++) {
        let sum = 0;
        for (let j = 0; j < binsPerBar; j++) {
          sum += dataRef.current[i * binsPerBar + j];
        }
        const avg = sum / binsPerBar;
        // Map dB range [-100, 0] to [0, 1]
        const normalized = Math.max(0, Math.min(1, (avg + 100) / 100));
        const barHeight = normalized * CANVAS_HEIGHT;

        const hue = 260 - normalized * 60; // purple to blue
        ctx.fillStyle = `hsl(${hue}, 80%, ${30 + normalized * 40}%)`;
        ctx.fillRect(i * binWidth + 1, CANVAS_HEIGHT - barHeight, binWidth - 2, barHeight);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [analyserNode, playing]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_WIDTH}
      height={CANVAS_HEIGHT}
      className="w-full rounded bg-gray-900"
    />
  );
}
