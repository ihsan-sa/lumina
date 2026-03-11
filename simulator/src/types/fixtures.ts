/** Mirrors lumina.control.protocol.FixtureCommand. */
export interface FixtureCommand {
  fixture_id: number;
  red: number;
  green: number;
  blue: number;
  white: number;
  strobe_rate: number;
  strobe_intensity: number;
  special: number;
}

/** Mirrors lumina.audio.models.MusicState. */
export interface MusicState {
  timestamp: number;
  bpm: number;
  beat_phase: number;
  bar_phase: number;
  is_beat: boolean;
  is_downbeat: boolean;
  energy: number;
  energy_derivative: number;
  segment: string;
  genre_weights: Record<string, number>;
  vocal_energy: number;
  spectral_centroid: number;
  sub_bass_energy: number;
  onset_type: string | null;
  drop_probability: number;
  layer_count: number;
  layer_mask: Record<string, number>;
  motif_id: number | null;
  motif_repetition: number;
  notes_per_beat: number;
  note_pattern_phase: number;
  headroom: number;
}

/** Fixture types matching the backend FixtureType enum values. */
export type FixtureType = "par" | "strobe" | "uv" | "led_bar" | "laser";

export interface FixtureConfig {
  id: number;
  type: FixtureType;
  position: [number, number, number]; // Three.js world coordinates
  label: string;
}

/** Fixture data sent by the backend in fixture_layout messages. */
export interface BackendFixture {
  fixture_id: number;
  fixture_type: string;
  position: [number, number, number]; // Backend coords: x(0..5), y(0..7), z(0..2.5)
  role: string;
  name: string;
}

// Room dimensions (must match backend fixture_map.py)
const ROOM_WIDTH = 5.0;
const ROOM_DEPTH = 7.0;

/**
 * Convert backend coordinate system to Three.js world coordinates.
 *
 * Backend: x = left(0) to right(5), y = front(0) to back(7), z = floor(0) to ceiling(2.5)
 * Three.js: X = centered, Y = floor to ceiling, Z = back(-) to front(+)
 */
export function backendToThreeJS(bx: number, by: number, bz: number): [number, number, number] {
  return [bx - ROOM_WIDTH / 2, bz, -(by - ROOM_DEPTH / 2)];
}

/** Convert a BackendFixture to a FixtureConfig for Three.js rendering. */
export function backendFixtureToConfig(bf: BackendFixture): FixtureConfig {
  return {
    id: bf.fixture_id,
    type: bf.fixture_type as FixtureType,
    position: backendToThreeJS(...bf.position),
    label: bf.name,
  };
}

/**
 * Fallback fixture layout matching the 15-fixture default from fixture_map.py.
 * Used when the backend has not yet sent a fixture_layout message.
 */
export const FALLBACK_FIXTURES: BackendFixture[] = [
  // RGBW Pars — left wall
  { fixture_id: 1, fixture_type: "par", position: [0.0, 1.4, 2.0], role: "left", name: "Par L1" },
  { fixture_id: 2, fixture_type: "par", position: [0.0, 2.8, 2.1], role: "left", name: "Par L2" },
  { fixture_id: 3, fixture_type: "par", position: [0.0, 4.2, 2.2], role: "left", name: "Par L3" },
  { fixture_id: 4, fixture_type: "par", position: [0.0, 5.6, 2.3], role: "left", name: "Par L4" },
  // RGBW Pars — right wall
  { fixture_id: 5, fixture_type: "par", position: [5.0, 1.4, 2.0], role: "right", name: "Par R1" },
  { fixture_id: 6, fixture_type: "par", position: [5.0, 2.8, 2.1], role: "right", name: "Par R2" },
  { fixture_id: 7, fixture_type: "par", position: [5.0, 4.2, 2.2], role: "right", name: "Par R3" },
  { fixture_id: 8, fixture_type: "par", position: [5.0, 5.6, 2.3], role: "right", name: "Par R4" },
  // Strobes — corners
  { fixture_id: 9, fixture_type: "strobe", position: [0.3, 0.3, 2.4], role: "front_left", name: "Strobe FL" },
  { fixture_id: 10, fixture_type: "strobe", position: [4.7, 0.3, 2.4], role: "front_right", name: "Strobe FR" },
  { fixture_id: 11, fixture_type: "strobe", position: [0.3, 6.7, 2.4], role: "back_left", name: "Strobe BL" },
  { fixture_id: 12, fixture_type: "strobe", position: [4.7, 6.7, 2.4], role: "back_right", name: "Strobe BR" },
  // LED Bars — ceiling
  { fixture_id: 13, fixture_type: "led_bar", position: [2.5, 2.333, 2.5], role: "center", name: "Bar Front" },
  { fixture_id: 14, fixture_type: "led_bar", position: [2.5, 4.667, 2.5], role: "center", name: "Bar Rear" },
  // Laser — rear wall
  { fixture_id: 15, fixture_type: "laser", position: [2.5, 7.0, 2.4], role: "back", name: "Laser" },
];

// ── WebSocket message types ─────────────────────────────────────

export type ServerMessage =
  | {
      type: "fixture_commands";
      sequence: number;
      timestamp_ms: number;
      commands: FixtureCommand[];
    }
  | {
      type: "music_state";
      state: MusicState;
    }
  | {
      type: "playback_start";
      filename: string;
      duration: number;
      audio_url?: string;
      start_timestamp?: number;
    }
  | {
      type: "fixture_layout";
      fixtures: BackendFixture[];
    };

export type ClientMessage =
  | {
      type: "transport";
      action: "play" | "pause" | "seek";
      position?: number;
    }
  | {
      type: "audio_loaded";
      filename: string;
      duration: number;
    }
  | {
      type: "genre_override";
      profile: string | null;
    }
  | {
      type: "intensity";
      value: number;
    }
  | {
      type: "manual_effect";
      effect: "blackout" | "strobe_burst" | "uv_flash";
    }
  | {
      type: "pattern_override";
      pattern: string | null;
    };
