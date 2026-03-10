interface RoomProps {
  /** Ambient light intensity. 0.02 = dark show mode, 0.3 = room lights on. */
  ambient?: number;
}

/**
 * 3D room interior: 5m (X) x 7m (Z) x 2.5m (Y) box.
 * Walls/floor/ceiling use MeshStandardMaterial so they respond to ambient light.
 */
export function Room({ ambient = 0.02 }: RoomProps) {
  const width = 5;
  const depth = 7;
  const height = 2.5;
  const hw = width / 2;
  const hd = depth / 2;

  return (
    <group>
      {/* Floor — dark gray, slightly reflective */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color="#222222" roughness={0.6} metalness={0.1} />
      </mesh>

      {/* Ceiling — very dark */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, height, 0]}>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color="#111111" roughness={0.9} />
      </mesh>

      {/* Back wall (-Z) */}
      <mesh position={[0, height / 2, -hd]}>
        <planeGeometry args={[width, height]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
      </mesh>

      {/* Front wall (+Z) */}
      <mesh rotation={[0, Math.PI, 0]} position={[0, height / 2, hd]}>
        <planeGeometry args={[width, height]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
      </mesh>

      {/* Left wall (-X) */}
      <mesh rotation={[0, Math.PI / 2, 0]} position={[-hw, height / 2, 0]}>
        <planeGeometry args={[depth, height]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
      </mesh>

      {/* Right wall (+X) */}
      <mesh rotation={[0, -Math.PI / 2, 0]} position={[hw, height / 2, 0]}>
        <planeGeometry args={[depth, height]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
      </mesh>

      <ambientLight intensity={ambient} />
    </group>
  );
}
