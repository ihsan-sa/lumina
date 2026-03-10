import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useCallback, useEffect, useRef, useState } from "react";
import { Room } from "./components/Room";
import { Fixture } from "./components/Fixture";
import { ControlPanel } from "./components/ControlPanel";
import { useWebSocket } from "./hooks/useWebSocket";
import { useAudio } from "./hooks/useAudio";
import type { FixtureCommand, MusicState } from "./types/fixtures";

// ── Test mode types ───────────────────────────────────────────────

type TestPhase = "idle" | "all" | "pars" | "strobes" | "bars" | "laser";

const PHASE_LABELS: Record<TestPhase, string> = {
  idle: "",
  all: "All Fixtures",
  pars: "Pars",
  strobes: "Strobes",
  bars: "LED Bars",
  laser: "Laser",
};

/** Build a FixtureCommand with all fields specified. */
function cmd(
  id: number,
  r: number,
  g: number,
  b: number,
  w: number,
  rate: number,
  intensity: number,
  special: number,
): FixtureCommand {
  return {
    fixture_id: id,
    red: r,
    green: g,
    blue: b,
    white: w,
    strobe_rate: rate,
    strobe_intensity: intensity,
    special,
  };
}

const OFF = (id: number) => cmd(id, 0, 0, 0, 0, 0, 0, 0);
// Par: full white + full dimmer
const PAR_WHITE = (id: number) => cmd(id, 255, 255, 255, 255, 0, 0, 255);
// Strobe: steady on (rate=0, intensity=255 → DC mode)
const STROBE_STEADY = (id: number) => cmd(id, 255, 255, 255, 0, 0, 255, 0);
// LED bar: warm white
const BAR_WARM = (id: number) => cmd(id, 255, 200, 100, 255, 0, 0, 255);
// Laser: green, pattern 1
const LASER_GREEN = (id: number) => cmd(id, 0, 255, 0, 0, 0, 0, 1);

function writeTestFrame(
  phase: TestPhase,
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>,
) {
  const map = commandsRef.current;
  // Clear all fixtures first
  for (let i = 1; i <= 15; i++) map.set(i, OFF(i));

  if (phase === "all" || phase === "pars") {
    for (let i = 1; i <= 8; i++) map.set(i, PAR_WHITE(i));
  }
  if (phase === "all" || phase === "strobes") {
    for (let i = 9; i <= 12; i++) map.set(i, STROBE_STEADY(i));
  }
  if (phase === "all" || phase === "bars") {
    map.set(13, BAR_WARM(13));
    map.set(14, BAR_WARM(14));
  }
  if (phase === "all" || phase === "laser") {
    map.set(15, LASER_GREEN(15));
  }
}

// ── Music state polling (15fps for control panel) ─────────────────

function useMusicStatePolled(ref: React.MutableRefObject<MusicState | null>): MusicState | null {
  const [state, setState] = useState<MusicState | null>(null);

  useEffect(() => {
    const id = setInterval(() => setState(ref.current), 66);
    return () => clearInterval(id);
  }, [ref]);

  return state;
}

// ── App ───────────────────────────────────────────────────────────

export default function App() {
  const { connected, commandsRef, musicStateRef, playbackInfoRef, fixtures, sendMessage } =
    useWebSocket();
  const audio = useAudio();
  const musicState = useMusicStatePolled(musicStateRef);
  const [autoPlayActive, setAutoPlayActive] = useState(false);
  const autoPlayTriggered = useRef(false);

  // Test mode state
  const [testPhase, setTestPhase] = useState<TestPhase>("idle");
  const testPhaseRef = useRef<TestPhase>("idle");
  const testIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const testStartRef = useRef(0);

  // Room lights toggle
  const [roomLights, setRoomLights] = useState(false);

  // Keep ref in sync with state for use inside interval closure
  testPhaseRef.current = testPhase;

  const stopTest = useCallback(() => {
    if (testIntervalRef.current !== null) {
      clearInterval(testIntervalRef.current);
      testIntervalRef.current = null;
    }
    // Clear all fixture overrides so WebSocket takes back over
    const map = commandsRef.current;
    for (let i = 1; i <= 15; i++) map.set(i, OFF(i));
    setTestPhase("idle");
  }, [commandsRef]);

  const startTest = useCallback(() => {
    if (testIntervalRef.current !== null) return; // already running
    testStartRef.current = Date.now();
    setTestPhase("all");
    testPhaseRef.current = "all";

    testIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - testStartRef.current;

      // Phase schedule: 0-5s all, 5-7s pars, 7-9s strobes, 9-11s bars, 11-13s laser
      let newPhase: TestPhase;
      if (elapsed < 5000) newPhase = "all";
      else if (elapsed < 7000) newPhase = "pars";
      else if (elapsed < 9000) newPhase = "strobes";
      else if (elapsed < 11000) newPhase = "bars";
      else if (elapsed < 13000) newPhase = "laser";
      else {
        stopTest();
        return;
      }

      if (newPhase !== testPhaseRef.current) {
        testPhaseRef.current = newPhase;
        setTestPhase(newPhase);
      }

      writeTestFrame(newPhase, commandsRef);
    }, 16);
  }, [commandsRef, stopTest]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (testIntervalRef.current !== null) clearInterval(testIntervalRef.current);
    };
  }, []);

  // Auto-play banner
  useEffect(() => {
    const id = setInterval(() => {
      if (playbackInfoRef.current && !autoPlayTriggered.current) {
        autoPlayTriggered.current = true;
        setAutoPlayActive(true);
      }
    }, 100);
    return () => clearInterval(id);
  }, [playbackInfoRef]);

  const isTestRunning = testPhase !== "idle";

  return (
    <div className="flex h-screen w-screen">
      {/* 3D viewport */}
      <div className="flex-1 relative">
        <Canvas
          camera={{ position: [0, 2, 6], fov: 60, near: 0.1, far: 50 }}
          gl={{ antialias: true, toneMapping: 3 /* ACESFilmic */ }}
        >
          <Room ambient={roomLights ? 0.3 : 0.02} />
          {fixtures.map((fc) => (
            <Fixture key={fc.id} config={fc} commandsRef={commandsRef} />
          ))}
          <OrbitControls target={[0, 1.2, 0]} maxPolarAngle={Math.PI * 0.85} />
        </Canvas>

        {/* Top-right overlay: Room Lights + Test All */}
        <div className="absolute top-3 right-3 flex flex-col items-end gap-2">
          {/* Room Lights toggle */}
          <button
            onClick={() => setRoomLights((v) => !v)}
            className={`px-3 py-1.5 text-xs rounded border transition-colors ${
              roomLights
                ? "bg-amber-500/20 border-amber-500/50 text-amber-300"
                : "bg-gray-900/80 border-gray-700 text-gray-400 hover:text-gray-200"
            }`}
          >
            Room Lights {roomLights ? "ON" : "OFF"}
          </button>

          {/* Test All / stop + phase indicator */}
          {!isTestRunning ? (
            <button
              onClick={startTest}
              className="px-3 py-1.5 text-xs rounded border bg-gray-900/80 border-gray-700 text-gray-300 hover:bg-gray-700/80 hover:text-white transition-colors"
            >
              Test All
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs rounded bg-green-900/60 border border-green-700/50 text-green-300">
                {PHASE_LABELS[testPhase]}
              </span>
              <button
                onClick={stopTest}
                className="px-3 py-1.5 text-xs rounded border bg-red-900/40 border-red-700/50 text-red-300 hover:bg-red-800/50 transition-colors"
              >
                Stop
              </button>
            </div>
          )}
        </div>

        {/* Auto-play banner */}
        {autoPlayActive && !isTestRunning && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-indigo-600/80 rounded text-xs text-white backdrop-blur">
            Receiving light show from backend
            {playbackInfoRef.current ? ` — ${playbackInfoRef.current.filename}` : ""}
          </div>
        )}
      </div>

      {/* Control panel */}
      <div className="w-80 border-l border-gray-800 flex-shrink-0">
        <ControlPanel
          connected={connected}
          musicState={musicState}
          audio={audio}
          sendMessage={sendMessage}
        />
      </div>
    </div>
  );
}
