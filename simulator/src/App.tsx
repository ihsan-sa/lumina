import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useState } from "react";
import { Room } from "./components/Room";
import { Fixture } from "./components/Fixture";
import { ControlPanel } from "./components/ControlPanel";
import { useWebSocket } from "./hooks/useWebSocket";
import { useAudio } from "./hooks/useAudio";
import { DEFAULT_FIXTURES } from "./types/fixtures";
import type { MusicState } from "./types/fixtures";

/** Polls musicStateRef at ~15fps for the control panel display. */
function useMusicStatePolled(ref: React.MutableRefObject<MusicState | null>): MusicState | null {
  const [state, setState] = useState<MusicState | null>(null);

  useEffect(() => {
    const id = setInterval(() => {
      setState(ref.current);
    }, 66); // ~15fps
    return () => clearInterval(id);
  }, [ref]);

  return state;
}

export default function App() {
  const { connected, commandsRef, musicStateRef, sendMessage } = useWebSocket();
  const audio = useAudio();
  const musicState = useMusicStatePolled(musicStateRef);

  return (
    <div className="flex h-screen w-screen">
      {/* 3D viewport */}
      <div className="flex-1 relative">
        <Canvas
          camera={{ position: [0, 2, 6], fov: 60, near: 0.1, far: 50 }}
          gl={{ antialias: true, toneMapping: 3 /* ACESFilmic */ }}
        >
          <Room />
          {DEFAULT_FIXTURES.map((fc) => (
            <Fixture key={fc.id} config={fc} commandsRef={commandsRef} />
          ))}
          <OrbitControls target={[0, 1.2, 0]} maxPolarAngle={Math.PI * 0.85} />
        </Canvas>
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
