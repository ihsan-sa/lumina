import { useCallback } from "react";
import type { AudioHandle } from "../hooks/useAudio";
import type { ClientMessage, MusicState } from "../types/fixtures";
import { AudioPlayer } from "./AudioPlayer";
import { Spectrogram } from "./Spectrogram";

const GENRE_PROFILES = [
  "Auto",
  "rage_trap",
  "psych_rnb",
  "french_melodic",
  "french_hard",
  "euro_alt",
  "theatrical",
  "festival_edm",
  "uk_bass",
] as const;

interface ControlPanelProps {
  connected: boolean;
  musicState: MusicState | null;
  audio: AudioHandle;
  sendMessage: (msg: ClientMessage) => void;
}

function topGenre(weights: Record<string, number>): string {
  let best = "";
  let bestVal = -1;
  for (const [k, v] of Object.entries(weights)) {
    if (v > bestVal) {
      bestVal = v;
      best = k;
    }
  }
  return best || "—";
}

export function ControlPanel({ connected, musicState, audio, sendMessage }: ControlPanelProps) {
  const handleGenre = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      sendMessage({ type: "genre_override", profile: val === "Auto" ? null : val });
    },
    [sendMessage],
  );

  const handleIntensity = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      sendMessage({ type: "intensity", value: parseInt(e.target.value, 10) });
    },
    [sendMessage],
  );

  const handleEffect = useCallback(
    (effect: "blackout" | "strobe_burst" | "uv_flash") => {
      sendMessage({ type: "manual_effect", effect });
    },
    [sendMessage],
  );

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-y-auto bg-gray-950 text-sm">
      {/* Connection status */}
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
        <span className="text-xs text-gray-400">{connected ? "Connected" : "Disconnected"}</span>
      </div>

      {/* Mode */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Mode</h3>
        <div className="flex gap-2">
          <label className="flex items-center gap-1 text-xs">
            <input type="radio" name="mode" value="A" defaultChecked className="accent-indigo-500" />
            Live (A)
          </label>
          <label className="flex items-center gap-1 text-xs">
            <input type="radio" name="mode" value="C" className="accent-indigo-500" />
            Queue (C)
          </label>
          <label className="flex items-center gap-1 text-xs text-gray-600">
            <input type="radio" name="mode" value="B" disabled className="accent-indigo-500" />
            DJ (B)
          </label>
        </div>
      </section>

      {/* Audio player */}
      <AudioPlayer audio={audio} sendMessage={sendMessage} />

      {/* Spectrogram */}
      <Spectrogram analyserNode={audio.analyserNode} playing={audio.playing} />

      {/* Genre override */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Genre Profile</h3>
        <select
          onChange={handleGenre}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
        >
          {GENRE_PROFILES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </section>

      {/* Intensity */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Intensity</h3>
        <input
          type="range"
          min={0}
          max={100}
          defaultValue={80}
          onChange={handleIntensity}
          className="w-full h-1 accent-indigo-500"
        />
      </section>

      {/* Manual effects */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Manual Effects</h3>
        <div className="flex gap-1">
          <button
            onClick={() => handleEffect("blackout")}
            className="flex-1 px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded"
          >
            Blackout
          </button>
          <button
            onClick={() => handleEffect("strobe_burst")}
            className="flex-1 px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded"
          >
            Strobe
          </button>
          <button
            onClick={() => handleEffect("uv_flash")}
            className="flex-1 px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded"
          >
            UV Flash
          </button>
        </div>
      </section>

      {/* Music state display */}
      {musicState && (
        <section className="mt-2 p-2 bg-gray-900 rounded text-xs space-y-1">
          <h3 className="font-semibold text-gray-500 uppercase">Music State</h3>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-300">
            <span>BPM</span>
            <span className="text-right">{musicState.bpm.toFixed(1)}</span>
            <span>Segment</span>
            <span className="text-right">{musicState.segment}</span>
            <span>Genre</span>
            <span className="text-right">{topGenre(musicState.genre_weights)}</span>
            <span>Drop Prob</span>
            <span className="text-right">{(musicState.drop_probability * 100).toFixed(0)}%</span>
          </div>
          {/* Energy bar */}
          <div className="mt-1">
            <span className="text-gray-500">Energy</span>
            <div className="w-full h-2 bg-gray-800 rounded mt-0.5">
              <div
                className="h-full bg-indigo-500 rounded transition-all duration-75"
                style={{ width: `${musicState.energy * 100}%` }}
              />
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
