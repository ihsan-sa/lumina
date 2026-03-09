import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, FixtureCommand, MusicState, ServerMessage } from "../types/fixtures";

const WS_URL = `ws://${window.location.hostname}:8765/ws`;
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 10000;

export interface PlaybackInfo {
  filename: string;
  duration: number;
}

export interface WebSocketHandle {
  connected: boolean;
  commandsRef: React.MutableRefObject<Map<number, FixtureCommand>>;
  musicStateRef: React.MutableRefObject<MusicState | null>;
  playbackInfoRef: React.MutableRefObject<PlaybackInfo | null>;
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

export function useWebSocket(): WebSocketHandle {
  const [connected, setConnected] = useState(false);
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
          playbackInfoRef.current = {
            filename: msg.filename,
            duration: msg.duration,
          };
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

  return { connected, commandsRef, musicStateRef, playbackInfoRef, sendMessage };
}
