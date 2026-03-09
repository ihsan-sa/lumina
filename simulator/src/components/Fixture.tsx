import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { FixtureCommand, FixtureConfig } from "../types/fixtures";

interface FixtureProps {
  config: FixtureConfig;
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>;
}

const _color = new THREE.Color();

/** Convert RGBW bytes (0-255) to a THREE.Color. W adds uniform brightness. */
function rgbwToColor(r: number, g: number, b: number, w: number): THREE.Color {
  const wn = w / 255;
  return _color.setRGB(
    Math.min(1, r / 255 + wn),
    Math.min(1, g / 255 + wn),
    Math.min(1, b / 255 + wn),
    THREE.SRGBColorSpace,
  );
}

function RgbwPar({ config, commandsRef }: FixtureProps) {
  const lightRef = useRef<THREE.SpotLight>(null);
  const coneRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    const cmd = commandsRef.current.get(config.id);
    if (!lightRef.current) return;

    if (!cmd || cmd.special === 0) {
      lightRef.current.intensity = 0;
      if (coneRef.current) coneRef.current.visible = false;
      return;
    }

    const dimmer = cmd.special / 255;
    const color = rgbwToColor(cmd.red, cmd.green, cmd.blue, cmd.white);
    lightRef.current.color.copy(color);
    lightRef.current.intensity = dimmer * 5;

    if (coneRef.current) {
      coneRef.current.visible = dimmer > 0.05;
      const mat = coneRef.current.material as THREE.MeshBasicMaterial;
      mat.color.copy(color);
      mat.opacity = dimmer * 0.15;
    }
  });

  return (
    <group position={config.position} rotation={config.rotation}>
      <spotLight
        ref={lightRef}
        angle={Math.PI / 4}
        penumbra={0.5}
        distance={8}
        intensity={0}
        castShadow={false}
      />
      {/* Volumetric cone hint */}
      <mesh ref={coneRef} position={[0, -2, 0]} visible={false}>
        <coneGeometry args={[1.8, 4, 16, 1, true]} />
        <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* Fixture body */}
      <mesh>
        <cylinderGeometry args={[0.08, 0.12, 0.06, 8]} />
        <meshStandardMaterial color="#333" />
      </mesh>
    </group>
  );
}

function StrobeFixture({ config, commandsRef }: FixtureProps) {
  const lightRef = useRef<THREE.SpotLight>(null);
  const strobeTimer = useRef(0);
  const strobeOn = useRef(false);

  useFrame((_state, delta) => {
    const cmd = commandsRef.current.get(config.id);
    if (!lightRef.current) return;

    if (!cmd || cmd.strobe_rate === 0) {
      lightRef.current.intensity = 0;
      strobeTimer.current = 0;
      strobeOn.current = false;
      return;
    }

    const hz = (cmd.strobe_rate / 255) * 25;
    const period = 1 / hz;
    strobeTimer.current += delta;

    if (strobeTimer.current >= period) {
      strobeTimer.current -= period;
      strobeOn.current = !strobeOn.current;
    }

    const brightness = cmd.strobe_intensity / 255;
    lightRef.current.intensity = strobeOn.current ? brightness * 8 : 0;
    lightRef.current.color.setRGB(1, 1, 1, THREE.SRGBColorSpace);
  });

  return (
    <group position={config.position} rotation={config.rotation}>
      <spotLight
        ref={lightRef}
        angle={Math.PI / 6}
        penumbra={0.3}
        distance={6}
        intensity={0}
        castShadow={false}
      />
      <mesh>
        <boxGeometry args={[0.15, 0.04, 0.1]} />
        <meshStandardMaterial color="#222" />
      </mesh>
    </group>
  );
}

function UvBar({ config, commandsRef }: FixtureProps) {
  const lightRef = useRef<THREE.PointLight>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    const cmd = commandsRef.current.get(config.id);
    const intensity = cmd ? cmd.special / 255 : 0;

    if (lightRef.current) {
      lightRef.current.intensity = intensity * 3;
    }
    if (meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = intensity * 2;
    }
  });

  return (
    <group position={config.position} rotation={config.rotation}>
      <pointLight ref={lightRef} color="#7b00ff" distance={4} intensity={0} />
      <mesh ref={meshRef}>
        <boxGeometry args={[0.05, 0.6, 0.04]} />
        <meshStandardMaterial color="#1a0033" emissive="#7b00ff" emissiveIntensity={0} />
      </mesh>
    </group>
  );
}

export function Fixture({ config, commandsRef }: FixtureProps) {
  switch (config.type) {
    case "rgbw_par":
      return <RgbwPar config={config} commandsRef={commandsRef} />;
    case "strobe":
      return <StrobeFixture config={config} commandsRef={commandsRef} />;
    case "uv_bar":
      return <UvBar config={config} commandsRef={commandsRef} />;
  }
}
