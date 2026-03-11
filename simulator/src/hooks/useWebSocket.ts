import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, FixtureCommand, FixtureConfig, MusicState, ServerMessage } from "../types/fixtures";
import { backendFixtureToConfig, FALLBACK_FIXTURES } from "../types/fixtures";

const WS_URL = `ws://${window.location.hostname}:8765/ws`;
export const BACKEND_BASE_URL = `http://${window.location.hostname}:8765`;
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 10000;

export interface PlaybackInfo {
  filename: string;
  duration: number;
  audio_url?: string;
  start_timestamp?: number;
}

export interface WebSocketHandle {
  connected: boolean;
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>;
  musicStateRef: React.MutableRefObject<MusicState | null>;
  playbackInfoRef: React.MutableRefObject<PlaybackInfo | null>;
  fixtures: FixtureConfig[];
  sendMessage: (msg: ClientMessage) => void;
}

const DEFAULT_MUSIC_STATE: MusicState = {
  timestamp: 0,
  bpm: 120,
  beat_phase: 0,
  bar_phase: 0,
  is_beat: false,
  is_downbeat: false,
  energy: 0,
  energy_derivative: 0,
  segment: "verse",
  genre_weights: {},
  vocal_energy: 0,
  spectral_centroid: 0,
  sub_bass_energy: 0,
  onset_type: null,
  drop_probability: 0,
};

/** Fallback configs from the default 15-fixture layout. */
const FALLBACK_CONFIGS: FixtureConfig[] = FALLBACK_FIXTURES.map(backendFixtureToConfig);

export function useWebSocket(): WebSocketHandle {
  const [connected, setConnected] = useState(false);
  const [fixtures, setFixtures] = useState<FixtureConfig[]>(FALLBACK_CONFIGS);
  const wsRef = useRef<WebSocket | null>(null);
  const commandsRef = useRef<Map<number, FixtureCommand>>(new Map());
  const musicStateRef = useRef<MusicState | null>(null);
  const playbackInfoRef = useRef<PlaybackInfo | null>(null);
  const reconnectDelay = useRef(RECONNECT_MIN_MS);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      reconnectDelay.current = RECONNECT_MIN_MS;
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      if (!mountedRef.current) return;
      const delay = reconnectDelay.current;
      reconnectDelay.current = Math.min(delay * 2, RECONNECT_MAX_MS);
      setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as ServerMessage;
        if (msg.type === "fixture_commands") {
          for (const cmd of msg.commands) {
            commandsRef.current.set(cmd.fixture_id, cmd);
          }
        } else if (msg.type === "music_state") {
          musicStateRef.current = { ...DEFAULT_MUSIC_STATE, ...msg.state };
        } else if (msg.type === "playback_start") {
          console.log("[LUMINA ws] playback_start received:", msg);
          playbackInfoRef.current = {
            filename: msg.filename,
            duration: msg.duration,
            audio_url: msg.audio_url,
            start_timestamp: msg.start_timestamp,
          };
        } else if (msg.type === "fixture_layout") {
          const configs = msg.fixtures.map(backendFixtureToConfig);
          setFixtures(configs);
        }
      } catch {
        // Ignore malformed messages
      }
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { connected, commandsRef, musicStateRef, playbackInfoRef, fixtures, sendMessage };
}
