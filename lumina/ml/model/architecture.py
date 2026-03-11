"""Temporal Fusion Transformer for lighting prediction.

Architecture: Genre-conditioned sequence-to-sequence model that maps
MusicState features to high-level LightingIntent predictions.

Input (per timestep):
  - MusicState features (11 floats): energy, beat_phase, bar_phase,
    spectral_centroid, sub_bass, vocal_energy, drop_probability, is_beat,
    is_downbeat, energy_derivative, bpm
  - Genre embedding (8-dim): learned embedding from genre_profile label
  - Segment embedding (8-dim): learned embedding from segment label
  - Positional encoding: bar_phase and beat_phase (already in features)

Context window: 4 seconds (40 frames at 10fps)

Encoder:
  - Input projection: Linear(~36 -> 128)
  - 4x Transformer encoder layers (d_model=128, nhead=8, dim_ff=256)
  - Causal attention mask (model only sees past + current, not future)

Decoder heads (multi-task):
  - Color head: Linear(128 -> 6) -> sigmoid
  - Spatial head: Linear(128 -> 5) -> sigmoid
  - Effect head: Linear(128 -> 3) -> sigmoid

Total parameters: ~500K
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ── Constants ────────────────────────────────────────────────────────

CONTEXT_WINDOW = 40  # 4 seconds at 10fps
ANALYSIS_FPS = 10  # Video/training frame rate

# Number of raw MusicState float features extracted per frame.
# Core features only — advanced fields (layer_count, notes_per_beat,
# note_pattern_phase, headroom, motif_repetition) are excluded until
# the corresponding analyzers are implemented.
NUM_MUSIC_FEATURES = 11

# Genre profiles known to the system.
GENRE_LABELS: list[str] = [
    "rage_trap",
    "psych_rnb",
    "french_melodic",
    "french_hard",
    "euro_alt",
    "theatrical",
    "festival_edm",
    "uk_bass",
    "generic",
]
NUM_GENRES = len(GENRE_LABELS)

# Segment labels known to the system.
SEGMENT_LABELS: list[str] = [
    "intro",
    "verse",
    "chorus",
    "drop",
    "breakdown",
    "bridge",
    "outro",
]
NUM_SEGMENTS = len(SEGMENT_LABELS)

# Embedding dimensions.
GENRE_EMBED_DIM = 8
SEGMENT_EMBED_DIM = 8

# Total input dimension after projection: features + genre_embed + segment_embed.
TOTAL_INPUT_DIM = NUM_MUSIC_FEATURES + GENRE_EMBED_DIM + SEGMENT_EMBED_DIM  # 32

# Transformer hyper-parameters.
D_MODEL = 128
N_HEAD = 8
DIM_FF = 256
NUM_LAYERS = 4
DROPOUT = 0.1

# Decoder head output sizes.
COLOR_HEAD_DIM = 6  # hue, saturation, secondary_hue, diversity, temperature, brightness
SPATIAL_HEAD_DIM = 5  # left, center, right, symmetry, variance
EFFECT_HEAD_DIM = 3  # strobe_prob, blackout_prob, brightness_delta_magnitude


# ── LightingIntent dataclass ─────────────────────────────────────────


@dataclass
class LightingIntent:
    """High-level lighting intent predicted by the ML model.

    This is an intermediate representation between the raw model outputs
    and per-fixture FixtureCommands.  The intent captures the overall
    visual feel that gets translated by the intent_mapper.

    Args:
        dominant_color: HSV tuple (hue 0-360, saturation 0-1, value 0-1).
        secondary_color: HSV tuple for accent / secondary colour.
        overall_brightness: 0-1 global brightness level.
        color_diversity: 0-1 how many distinct colours are visible.
        spatial_distribution: (left, center, right) brightness levels.
        spatial_symmetry: 0-1 how symmetric left vs right should be.
        strobe_active: Whether strobe effect is active.
        strobe_intensity: 0-1 strobe brightness when active.
        blackout: Whether a full blackout is active.
    """

    dominant_color: tuple[float, float, float]  # HSV
    secondary_color: tuple[float, float, float]  # HSV
    overall_brightness: float
    color_diversity: float
    spatial_distribution: tuple[float, float, float]  # left, center, right
    spatial_symmetry: float
    strobe_active: bool
    strobe_intensity: float
    blackout: bool


# ── Positional encoding ──────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to input embeddings.

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Causal mask helper ───────────────────────────────────────────────


def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate a causal (upper-triangular) attention mask.

    The mask prevents each position from attending to future positions.

    Args:
        seq_len: Length of the sequence.
        device: Target device.

    Returns:
        Float mask of shape (seq_len, seq_len) with -inf for masked positions.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


# ── Model ────────────────────────────────────────────────────────────


class LightingTransformer(nn.Module):
    """Temporal Fusion Transformer variant for lighting prediction.

    Encodes a window of MusicState frames with genre and segment
    conditioning, then decodes through three task-specific heads
    (color, spatial, effect).

    Args:
        num_music_features: Number of float features per frame.
        num_genres: Number of genre classes for the embedding table.
        num_segments: Number of segment classes for the embedding table.
        genre_embed_dim: Dimension of genre embeddings.
        segment_embed_dim: Dimension of segment embeddings.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        dim_ff: Feed-forward hidden dimension.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_music_features: int = NUM_MUSIC_FEATURES,
        num_genres: int = NUM_GENRES,
        num_segments: int = NUM_SEGMENTS,
        genre_embed_dim: int = GENRE_EMBED_DIM,
        segment_embed_dim: int = SEGMENT_EMBED_DIM,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        dim_ff: int = DIM_FF,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Embedding tables for categorical inputs.
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_dim)
        self.segment_embedding = nn.Embedding(num_segments, segment_embed_dim)

        # Input projection: concat(music_features, genre_embed, segment_embed) -> d_model.
        input_dim = num_music_features + genre_embed_dim + segment_embed_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding.
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=CONTEXT_WINDOW * 2)

        # Transformer encoder with causal masking.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm before decoder heads.
        self.pre_head_norm = nn.LayerNorm(d_model)

        # Decoder heads — each predicts a different aspect of lighting.
        self.color_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, COLOR_HEAD_DIM),
            nn.Sigmoid(),
        )

        self.spatial_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, SPATIAL_HEAD_DIM),
            nn.Sigmoid(),
        )

        self.effect_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, EFFECT_HEAD_DIM),
            nn.Sigmoid(),
        )

    def forward(
        self,
        music_features: torch.Tensor,
        genre_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the lighting transformer.

        Args:
            music_features: Float features, shape (batch, seq_len, num_music_features).
            genre_ids: Genre class indices, shape (batch, seq_len). Integer tensor.
            segment_ids: Segment class indices, shape (batch, seq_len). Integer tensor.

        Returns:
            Tuple of (color_out, spatial_out, effect_out):
              - color_out:   (batch, seq_len, 6) — sigmoid-activated
              - spatial_out: (batch, seq_len, 5) — sigmoid-activated
              - effect_out:  (batch, seq_len, 3) — sigmoid-activated
        """
        batch_size, seq_len, _ = music_features.shape

        # Embed categorical inputs.
        genre_emb = self.genre_embedding(genre_ids)  # (B, T, genre_embed_dim)
        segment_emb = self.segment_embedding(segment_ids)  # (B, T, segment_embed_dim)

        # Concatenate all inputs.
        x = torch.cat([music_features, genre_emb, segment_emb], dim=-1)  # (B, T, input_dim)

        # Project to model dimension.
        x = self.input_projection(x)  # (B, T, d_model)

        # Add positional encoding.
        x = self.pos_encoder(x)

        # Generate causal mask — each position attends only to itself and past.
        causal_mask = _generate_causal_mask(seq_len, x.device)

        # Transformer encoder.
        x = self.transformer_encoder(x, mask=causal_mask)  # (B, T, d_model)

        # Pre-head layer norm.
        x = self.pre_head_norm(x)

        # Decode through three heads.
        color_out = self.color_head(x)  # (B, T, 6)
        spatial_out = self.spatial_head(x)  # (B, T, 5)
        effect_out = self.effect_head(x)  # (B, T, 3)

        return color_out, spatial_out, effect_out

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Feature extraction helpers ───────────────────────────────────────


def genre_to_index(genre: str) -> int:
    """Convert a genre profile name to an integer index.

    Args:
        genre: Genre profile name (e.g. "rage_trap").

    Returns:
        Integer index for the embedding table. Returns index of
        "generic" for unknown genres.
    """
    try:
        return GENRE_LABELS.index(genre)
    except ValueError:
        return GENRE_LABELS.index("generic")


def segment_to_index(segment: str) -> int:
    """Convert a segment label to an integer index.

    Args:
        segment: Segment name (e.g. "chorus", "drop").

    Returns:
        Integer index for the embedding table. Returns 0 ("intro")
        for unknown segments.
    """
    try:
        return SEGMENT_LABELS.index(segment)
    except ValueError:
        return 0
