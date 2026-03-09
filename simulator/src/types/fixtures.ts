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
}

export type FixtureType = "rgbw_par" | "strobe" | "uv_bar";

export interface FixtureConfig {
  id: number;
  type: FixtureType;
  position: [number, number, number];
  rotation: [number, number, number];
  label: string;
}

/** Default 8-fixture layout for a 5m x 7m x 2.5m room. */
export const DEFAULT_FIXTURES: FixtureConfig[] = [
  // 4x RGBW Par — ceiling corners, angled down 45°
  { id: 1, type: "rgbw_par", position: [-2.2, 2.4, -3.2], rotation: [-0.78, 0, 0], label: "Par FL" },
  { id: 2, type: "rgbw_par", position: [2.2, 2.4, -3.2], rotation: [-0.78, 0, 0], label: "Par FR" },
  { id: 3, type: "rgbw_par", position: [-2.2, 2.4, 3.2], rotation: [0.78, 0, 0], label: "Par BL" },
  { id: 4, type: "rgbw_par", position: [2.2, 2.4, 3.2], rotation: [0.78, 0, 0], label: "Par BR" },
  // 2x Strobe — ceiling center-line
  { id: 5, type: "strobe", position: [-1.0, 2.4, 0], rotation: [-1.57, 0, 0], label: "Strobe L" },
  { id: 6, type: "strobe", position: [1.0, 2.4, 0], rotation: [-1.57, 0, 0], label: "Strobe R" },
  // 2x UV Bar — walls at 2m height
  { id: 7, type: "uv_bar", position: [-2.45, 2.0, 0], rotation: [0, 0, 0.3], label: "UV Left" },
  { id: 8, type: "uv_bar", position: [2.45, 2.0, 0], rotation: [0, 0, -0.3], label: "UV Right" },
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
    };
