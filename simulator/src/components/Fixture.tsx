import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { FixtureCommand, FixtureConfig } from "../types/fixtures";

interface FixtureProps {
  config: FixtureConfig;
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>;
}

const _color = new THREE.Color();
const _aimObj = new THREE.Object3D();

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

/**
 * Compute a group rotation so that the local -Y axis points from
 * `position` toward `target`. Used to aim downward-pointing fixtures
 * (SpotLights and their volumetric cone visuals).
 */
function computeAimRotation(
  position: [number, number, number],
  target: [number, number, number],
): [number, number, number] {
  _aimObj.position.set(position[0], position[1], position[2]);
  _aimObj.up.set(0, 1, 0);
  _aimObj.lookAt(target[0], target[1], target[2]);
  // lookAt aligns local -Z with direction to target.
  // Rotate -90° around local X: maps old -Z → new -Y, so -Y points at target.
  _aimObj.rotateX(-Math.PI / 2);
  return [_aimObj.rotation.x, _aimObj.rotation.y, _aimObj.rotation.z];
}

// ── RGBW Par (wall-mounted, aims toward room center) ─────────────

function RgbwPar({ config, commandsRef }: FixtureProps) {
  const lightRef = useRef<THREE.SpotLight>(null);
  const coneRef = useRef<THREE.Mesh>(null);

  const rotation = useMemo(
    () => computeAimRotation(config.position, [0, 0, 0]),
    [config.position],
  );

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
    <group position={config.position} rotation={rotation}>
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

// ── Strobe (corner-mounted, PointLight for room-filling flash) ────

function StrobeFixture({ config, commandsRef }: FixtureProps) {
  const pointLightRef = useRef<THREE.PointLight>(null);
  const flashMeshRef = useRef<THREE.Mesh>(null);
  const strobeTimer = useRef(0);
  const strobeOn = useRef(false);

  useFrame((_state, delta) => {
    const cmd = commandsRef.current.get(config.id);

    // Determine whether the strobe is emitting this frame
    let on = false;

    if (!cmd || (cmd.strobe_rate === 0 && cmd.strobe_intensity === 0)) {
      strobeTimer.current = 0;
      strobeOn.current = false;
      on = false;
    } else if (cmd.strobe_rate === 0) {
      // DC mode: strobe_intensity > 0, rate = 0 → steady on
      on = true;
    } else {
      // Flashing mode
      const hz = (cmd.strobe_rate / 255) * 25;
      strobeTimer.current += delta;
      if (strobeTimer.current >= 1 / hz) {
        strobeTimer.current -= 1 / hz;
        strobeOn.current = !strobeOn.current;
      }
      on = strobeOn.current;
    }

    const brightness = cmd ? cmd.strobe_intensity / 255 : 0;

    // PointLight — high intensity, large radius so flash hits walls and floor
    if (pointLightRef.current) {
      pointLightRef.current.intensity = on ? brightness * 30 : 0;
    }

    // Flash tube glow on the mesh body
    if (flashMeshRef.current) {
      const mat = flashMeshRef.current.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = on ? brightness * 6 : 0;
    }
  });

  return (
    <group position={config.position}>
      {/*
       * PointLight radiates in all directions — no aim rotation needed.
       * High intensity + large distance lets the flash splash across
       * walls and floor visibly.
       */}
      <pointLight
        ref={pointLightRef}
        color="white"
        intensity={0}
        distance={14}
        decay={1.5}
        castShadow={false}
      />
      {/* Housing box */}
      <mesh>
        <boxGeometry args={[0.15, 0.04, 0.1]} />
        <meshStandardMaterial color="#2a2a2a" />
      </mesh>
      {/* Flash tube — glows white when firing */}
      <mesh ref={flashMeshRef} position={[0, -0.03, 0]}>
        <sphereGeometry args={[0.05, 8, 6]} />
        <meshStandardMaterial color="#555" emissive="white" emissiveIntensity={0} />
      </mesh>
    </group>
  );
}

// ── UV Bar ────────────────────────────────────────────────────────

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
    <group position={config.position}>
      <pointLight ref={lightRef} color="#7b00ff" distance={4} intensity={0} />
      <mesh ref={meshRef}>
        <boxGeometry args={[0.05, 0.6, 0.04]} />
        <meshStandardMaterial color="#1a0033" emissive="#7b00ff" emissiveIntensity={0} />
      </mesh>
    </group>
  );
}

// ── LED Bar (ceiling-mounted, rectangular light) ──────────────────

function LedBarFixture({ config, commandsRef }: FixtureProps) {
  const lightRef = useRef<THREE.SpotLight>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    const cmd = commandsRef.current.get(config.id);
    const intensity = cmd ? cmd.special / 255 : 0;

    let color: THREE.Color | null = null;
    if (cmd && intensity >= 0.01) {
      color = rgbwToColor(cmd.red, cmd.green, cmd.blue, cmd.white);
    }

    if (lightRef.current) {
      if (color) {
        lightRef.current.color.copy(color);
        lightRef.current.intensity = intensity * 6;
      } else {
        lightRef.current.intensity = 0;
      }
    }

    if (meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      if (color) {
        mat.emissive.copy(color);
        mat.emissiveIntensity = intensity * 3;
      } else {
        mat.emissiveIntensity = 0;
      }
    }
  });

  return (
    <group position={config.position}>
      {/* Wide downward-facing spot */}
      <spotLight
        ref={lightRef}
        angle={Math.PI / 3}
        penumbra={0.6}
        distance={4}
        intensity={0}
        castShadow={false}
      />
      {/* Bar body — long rectangle flush with ceiling, oriented along Z (room depth) */}
      <mesh ref={meshRef} position={[0, -0.02, 0]}>
        <boxGeometry args={[0.12, 0.04, 2.0]} />
        <meshStandardMaterial color="#1a1a1a" emissive="#000000" emissiveIntensity={0} />
      </mesh>
    </group>
  );
}

// ── Laser (rear wall, beam fan with atmospheric haze) ─────────────

const LASER_BEAM_COUNT = 7;
const LASER_FAN_WIDTH = 4.5; // X spread at target (meters)
const LASER_FORWARD = 6.5; // Z reach into room
const LASER_DROP = -2.4; // Y drop to floor

// Shared geometry instances — created once, reused by all beams
const _coreGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 6);
const _hazeGeo = new THREE.CylinderGeometry(0.055, 0.055, 1, 8);

function LaserBeam({
  endpoint,
  commandsRef,
  fixtureId,
}: {
  endpoint: [number, number, number];
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>;
  fixtureId: number;
}) {
  const coreRef = useRef<THREE.Mesh>(null);
  const hazeRef = useRef<THREE.Mesh>(null);

  const { scale, midpoint, quaternion } = useMemo(() => {
    const len = Math.sqrt(endpoint[0] ** 2 + endpoint[1] ** 2 + endpoint[2] ** 2);
    const mid: [number, number, number] = [endpoint[0] / 2, endpoint[1] / 2, endpoint[2] / 2];
    const dir = new THREE.Vector3(...endpoint).normalize();
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    // Scale Y to stretch the unit-height geometry to the beam length
    return { scale: [1, len, 1] as [number, number, number], midpoint: mid, quaternion: quat };
  }, [endpoint]);

  useFrame(() => {
    const cmd = commandsRef.current.get(fixtureId);
    const active = !!(cmd && cmd.special > 0);

    const r = cmd ? Math.max(cmd.red / 255, 0.05) : 0.05;
    const g = cmd ? Math.max(cmd.green / 255, 0.05) : 0.05;
    const b = cmd ? Math.max(cmd.blue / 255, 0.05) : 0.05;

    if (coreRef.current) {
      coreRef.current.visible = active;
      if (active) {
        (coreRef.current.material as THREE.MeshBasicMaterial).color.setRGB(
          r,
          g,
          b,
          THREE.SRGBColorSpace,
        );
      }
    }

    if (hazeRef.current) {
      hazeRef.current.visible = active;
      if (active) {
        (hazeRef.current.material as THREE.MeshBasicMaterial).color.setRGB(
          r,
          g,
          b,
          THREE.SRGBColorSpace,
        );
      }
    }
  });

  return (
    <group position={midpoint} quaternion={quaternion} scale={scale}>
      {/* Core — thin, bright, additive */}
      <mesh ref={coreRef} geometry={_coreGeo} visible={false}>
        <meshBasicMaterial
          transparent
          opacity={0.92}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      {/* Atmospheric haze — wider cylinder, low-opacity glow */}
      <mesh ref={hazeRef} geometry={_hazeGeo} visible={false}>
        <meshBasicMaterial
          transparent
          opacity={0.14}
          side={THREE.BackSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function LaserFixture({ config, commandsRef }: FixtureProps) {
  const beamGroupRef = useRef<THREE.Group>(null);
  const glowRef = useRef<THREE.PointLight>(null);

  const beamEndpoints = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < LASER_BEAM_COUNT; i++) {
      const t = LASER_BEAM_COUNT > 1 ? i / (LASER_BEAM_COUNT - 1) : 0.5;
      const x = (t - 0.5) * LASER_FAN_WIDTH;
      pts.push([x, LASER_DROP, LASER_FORWARD]);
    }
    return pts;
  }, []);

  useFrame((state) => {
    const cmd = commandsRef.current.get(config.id);
    const active = !!(cmd && cmd.special > 0);

    if (beamGroupRef.current) {
      beamGroupRef.current.visible = active;
      if (active) {
        // Slow left-right sweep
        const sweep = Math.sin(state.clock.elapsedTime * 0.5) * 0.15;
        beamGroupRef.current.rotation.y = sweep;
      }
    }

    if (glowRef.current) {
      if (active && cmd) {
        glowRef.current.intensity = 1.5;
        glowRef.current.color.setRGB(
          cmd.red / 255,
          cmd.green / 255,
          cmd.blue / 255,
          THREE.SRGBColorSpace,
        );
      } else {
        glowRef.current.intensity = 0;
      }
    }
  });

  return (
    <group position={config.position}>
      {/* Ambient glow at the laser source when active */}
      <pointLight ref={glowRef} distance={3} intensity={0} />
      {/* Fan of beam lines */}
      <group ref={beamGroupRef} visible={false}>
        {beamEndpoints.map((end, i) => (
          <LaserBeam key={i} endpoint={end} commandsRef={commandsRef} fixtureId={config.id} />
        ))}
      </group>
      {/* Fixture body */}
      <mesh>
        <boxGeometry args={[0.12, 0.08, 0.12]} />
        <meshStandardMaterial color="#222" />
      </mesh>
    </group>
  );
}

// ── Fixture router ────────────────────────────────────────────────

export function Fixture({ config, commandsRef }: FixtureProps) {
  switch (config.type) {
    case "par":
      return <RgbwPar config={config} commandsRef={commandsRef} />;
    case "strobe":
      return <StrobeFixture config={config} commandsRef={commandsRef} />;
    case "uv":
      return <UvBar config={config} commandsRef={commandsRef} />;
    case "led_bar":
      return <LedBarFixture config={config} commandsRef={commandsRef} />;
    case "laser":
      return <LaserFixture config={config} commandsRef={commandsRef} />;
    default:
      return null;
  }
}
