import { useCallback, useEffect, useRef, useState } from "react";
import type { AudioHandle } from "../hooks/useAudio";
import type { ClientMessage, MusicState } from "../types/fixtures";
import { AudioPlayer } from "./AudioPlayer";
import { Spectrogram } from "./Spectrogram";

// ── Profile definitions ───────────────────────────────────────────────────────

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

/**
 * Pattern showcase list — add new patterns here when they are implemented in patterns.py.
 * Name must match a key in PATTERN_REGISTRY on the backend.
 */
const SHOWCASE_PATTERNS: { name: string; label: string; description: string }[] = [
  { name: "chase_lr",         label: "Chase L→R",        description: "Bright spot sweeps left to right across the bar" },
  { name: "chase_bounce",     label: "Chase Bounce",      description: "Ping-pong: sweeps left, then right, then left" },
  { name: "chase_mirror",     label: "Chase Mirror",      description: "Both walls sweep inward symmetrically" },
  { name: "converge",         label: "Converge",          description: "Edge fixtures fire first, collapse inward to center" },
  { name: "diverge",          label: "Diverge",           description: "Center blooms outward to the edges" },
  { name: "alternate",        label: "Alternate",         description: "Odd/even fixtures swap colors on every beat" },
  { name: "color_split",      label: "Color Split",       description: "Left half one color, right half complementary" },
  { name: "wash_hold",        label: "Wash Hold",         description: "Static color wash with gentle intensity drift" },
  { name: "breathe",          label: "Breathe",           description: "All fixtures sine-wave fade in and out together" },
  { name: "spotlight_isolate",label: "Spotlight",         description: "One fixture lit at a time, cycling through all" },
  { name: "strobe_burst",     label: "Strobe Burst",      description: "All strobes fire simultaneously at max rate" },
  { name: "strobe_chase",     label: "Strobe Chase",      description: "Strobes fire one at a time in rotating sequence" },
  { name: "lightning_flash",  label: "Lightning Flash",   description: "Main flash then exponential aftershock decay" },
  { name: "stutter",          label: "Stutter",           description: "Rapid binary on/off at 16th-note subdivisions" },
  { name: "color_pop",        label: "Color Pop",         description: "Complement color flashes on beats against wash" },
  { name: "rainbow_roll",     label: "Rainbow Roll",      description: "Each fixture a different hue, rotating continuously" },
  { name: "flicker",          label: "Flicker",           description: "Organic per-fixture intensity jitter (fire effect)" },
  { name: "gradient_y",       label: "Gradient Y",        description: "Front-to-back color gradient across the room" },
  { name: "random_scatter",   label: "Random Scatter",    description: "Deterministic pseudo-random fixture activation" },
  { name: "blinder",          label: "Blinder",           description: "All fixtures max white + max strobe — the nuclear option" },
];

const SHOWCASE_DURATION_S = 6; // seconds per pattern

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// ── Component ─────────────────────────────────────────────────────────────────

interface ControlPanelProps {
  connected: boolean;
  musicState: MusicState | null;
  audio: AudioHandle;
  sendMessage: (msg: ClientMessage) => void;
}

export function ControlPanel({ connected, musicState, audio, sendMessage }: ControlPanelProps) {
  // ── Pattern Showcase state ──────────────────────────────────────────────────
  const [showcaseIndex, setShowcaseIndex] = useState<number | null>(null);
  const [showcaseTimeLeft, setShowcaseTimeLeft] = useState(SHOWCASE_DURATION_S);
  const showcaseIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const showcaseIndexRef = useRef(0);
  const showcaseTimeRef = useRef(SHOWCASE_DURATION_S);

  const stopShowcase = useCallback(() => {
    if (showcaseIntervalRef.current !== null) {
      clearInterval(showcaseIntervalRef.current);
      showcaseIntervalRef.current = null;
    }
    setShowcaseIndex(null);
    setShowcaseTimeLeft(SHOWCASE_DURATION_S);
    sendMessage({ type: "pattern_override", pattern: null });
  }, [sendMessage]);

  // Start the auto-advance interval if it isn't already running.
  // The interval reads from refs so jumpToPattern can update position mid-run.
  const startIntervalIfNeeded = useCallback(() => {
    if (showcaseIntervalRef.current !== null) return;
    showcaseIntervalRef.current = setInterval(() => {
      showcaseTimeRef.current -= 0.1;
      if (showcaseTimeRef.current <= 0) {
        const nextIndex = showcaseIndexRef.current + 1;
        if (nextIndex >= SHOWCASE_PATTERNS.length) {
          if (showcaseIntervalRef.current !== null) {
            clearInterval(showcaseIntervalRef.current);
            showcaseIntervalRef.current = null;
          }
          setShowcaseIndex(null);
          setShowcaseTimeLeft(SHOWCASE_DURATION_S);
          sendMessage({ type: "pattern_override", pattern: null });
          return;
        }
        showcaseIndexRef.current = nextIndex;
        showcaseTimeRef.current = SHOWCASE_DURATION_S;
        setShowcaseIndex(nextIndex);
        setShowcaseTimeLeft(SHOWCASE_DURATION_S);
        sendMessage({ type: "pattern_override", pattern: SHOWCASE_PATTERNS[nextIndex].name });
      } else {
        setShowcaseTimeLeft(Math.max(0, showcaseTimeRef.current));
      }
    }, 100);
  }, [sendMessage]);

  // Jump to a specific pattern (starts auto-advance if not already running).
  const jumpToPattern = useCallback((index: number) => {
    showcaseIndexRef.current = index;
    showcaseTimeRef.current = SHOWCASE_DURATION_S;
    setShowcaseIndex(index);
    setShowcaseTimeLeft(SHOWCASE_DURATION_S);
    sendMessage({ type: "pattern_override", pattern: SHOWCASE_PATTERNS[index].name });
    startIntervalIfNeeded();
  }, [sendMessage, startIntervalIfNeeded]);

  const startShowcase = useCallback(() => jumpToPattern(0), [jumpToPattern]);

  useEffect(() => {
    return () => {
      if (showcaseIntervalRef.current !== null) clearInterval(showcaseIntervalRef.current);
    };
  }, []);

  // ── Genre / intensity / effects handlers ────────────────────────────────────
  const handleGenre = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      if (showcaseIndex !== null) stopShowcase();
      const val = e.target.value;
      sendMessage({ type: "genre_override", profile: val === "Auto" ? null : val });
    },
    [showcaseIndex, stopShowcase, sendMessage],
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

  const isShowcaseRunning = showcaseIndex !== null;
  const currentShowcase = isShowcaseRunning ? SHOWCASE_PATTERNS[showcaseIndex] : null;
  const progressPct = ((SHOWCASE_DURATION_S - showcaseTimeLeft) / SHOWCASE_DURATION_S) * 100;

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
      <AudioPlayer audio={audio} sendMessage={sendMessage} lastTimestamp={musicState?.timestamp} />

      {/* Spectrogram */}
      <Spectrogram analyserNode={audio.analyserNode} playing={audio.playing} />

      {/* Pattern Showcase */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Pattern Showcase</h3>

        {isShowcaseRunning && (
          <div className="flex flex-col gap-1.5 mb-1.5">
            {/* Current pattern card */}
            <div className="p-2.5 bg-violet-950/60 border border-violet-700/40 rounded">
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-bold text-violet-300">{currentShowcase?.label}</span>
                <span className="text-xs text-violet-500 tabular-nums">
                  {Math.ceil(showcaseTimeLeft)}s — {showcaseIndex + 1}/{SHOWCASE_PATTERNS.length}
                </span>
              </div>
              <p className="text-xs text-gray-400 mt-0.5 leading-tight">{currentShowcase?.description}</p>
              <div className="w-full h-1 bg-violet-950 rounded mt-2">
                <div
                  className="h-full bg-violet-500 rounded transition-all duration-100"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            <button
              onClick={stopShowcase}
              className="w-full px-2 py-1 text-xs rounded bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
            >
              Stop Showcase
            </button>
          </div>
        )}

        {/* Pattern pills — always visible, click to jump */}
        <div className="flex flex-wrap gap-1">
          {SHOWCASE_PATTERNS.map((p, i) => (
            <button
              key={p.name}
              onClick={() => jumpToPattern(i)}
              title={p.description}
              className={`px-1.5 py-0.5 text-xs rounded transition-colors ${
                isShowcaseRunning && i < showcaseIndex
                  ? "bg-gray-800 text-gray-600 line-through hover:bg-gray-700"
                  : isShowcaseRunning && i === showcaseIndex
                    ? "bg-violet-600 text-white hover:bg-violet-500"
                    : "bg-gray-800 text-gray-400 hover:bg-violet-700 hover:text-white"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </section>

      {/* Genre override */}
      <section>
        <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">Genre Profile</h3>
        <select
          onChange={handleGenre}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
          disabled={isShowcaseRunning}
        >
          {GENRE_PROFILES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        {isShowcaseRunning && (
          <p className="text-xs text-gray-600 mt-0.5">Disabled during showcase</p>
        )}
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
