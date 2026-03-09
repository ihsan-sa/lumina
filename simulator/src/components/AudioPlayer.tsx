import { useCallback, useRef } from "react";
import type { AudioHandle } from "../hooks/useAudio";
import type { ClientMessage } from "../types/fixtures";

interface AudioPlayerProps {
  audio: AudioHandle;
  sendMessage: (msg: ClientMessage) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioPlayer({ audio, sendMessage }: AudioPlayerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const { filename, duration } = await audio.loadFile(file);
      sendMessage({ type: "audio_loaded", filename, duration });
    },
    [audio, sendMessage],
  );

  const handlePlayPause = useCallback(() => {
    if (audio.playing) {
      audio.pause();
      sendMessage({ type: "transport", action: "pause" });
    } else {
      audio.play();
      sendMessage({ type: "transport", action: "play" });
    }
  }, [audio, sendMessage]);

  const handleSeek = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const time = parseFloat(e.target.value);
      audio.seek(time);
      sendMessage({ type: "transport", action: "seek", position: time });
    },
    [audio, sendMessage],
  );

  return (
    <div className="flex flex-col gap-2 p-3 bg-gray-900 rounded-lg">
      <div className="flex items-center gap-2">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
        >
          Load Audio
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFile}
          className="hidden"
        />
        {audio.loaded && (
          <button
            onClick={handlePlayPause}
            className="px-3 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 rounded"
          >
            {audio.playing ? "Pause" : "Play"}
          </button>
        )}
      </div>

      {audio.loaded && (
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="w-10 text-right">{formatTime(audio.currentTime)}</span>
          <input
            type="range"
            min={0}
            max={audio.duration}
            step={0.1}
            value={audio.currentTime}
            onChange={handleSeek}
            className="flex-1 h-1 accent-indigo-500"
          />
          <span className="w-10">{formatTime(audio.duration)}</span>
        </div>
      )}
    </div>
  );
}
