# LUMINA Genre Lighting Profiles

## Overview

LUMINA uses 8 genre-specific lighting profiles that define how music translates into light.
Each profile encodes a distinct lighting philosophy — not just color palettes, but timing,
contrast, spatial distribution, and emotional intent.

Tracks are classified into weighted blends across profiles (e.g., a Weeknd track might be
60% psych_rnb, 20% festival_edm, 10% rage_trap, 10% theatrical).

## Profile Summary

| # | Profile ID | Profile Name | Reference Artists | Key Principle |
|---|-----------|-------------|-------------------|---------------|
| 1 | `rage_trap` | Rage / Experimental Trap | Playboi Carti, Travis Scott | Extreme contrast — BLINDING or DARK, nothing in between |
| 2 | `psych_rnb` | Psychedelic Trap / Dark R&B | Don Toliver, The Weeknd | Smooth and flowing — transitions over bars, not beats |
| 3 | `french_melodic` | French Rap (Melodic) | Ninho, Jul | Warm and colorful — hi-hat bounce drives light rhythm |
| 4 | `french_hard` | French Rap (Hard) | Kaaris | Regimented power — every hit is deliberate like a punch |
| 5 | `euro_alt` | European Alt Hip-Hop | AyVe, Exetra Archive | Artistic restraint — visual silence makes hits impactful |
| 6 | `theatrical` | Theatrical Electronic | Stromae | Storytelling — lights follow emotional arc, not just beats |
| 7 | `festival_edm` | Festival EDM / Trance | Guetta, Armin, Edward Maya | Build-drop cycle — everything serves tension or release |
| 8 | `uk_bass` | UK Bass / Dubstep / Grime | Fred again.. | Underground rave — raw, DIY, imperfect by design |

## Profile Details

### 1. Rage / Experimental Trap (`rage_trap`)

- **Color palette**: Red, white, black (darkness). Minimal color variety.
- **Timing**: Hard cuts on beats. No smooth fades. Instant on/off.
- **Contrast**: Maximum. Alternates between blinding strobes and total blackout.
- **Strobe usage**: Heavy. Strobes are the primary instrument.
- **Spatial**: Chaotic — no symmetry, random fixture activation.
- **Energy mapping**: Low energy = darkness. High energy = sensory overload.

### 2. Psychedelic Trap / Dark R&B (`psych_rnb`)

- **Color palette**: Purple, deep blue, amber, magenta. Rich and saturated.
- **Timing**: Slow transitions over 2-4 bars. Changes feel like breathing.
- **Contrast**: Low-to-medium. Gradual shifts, not hard cuts.
- **Strobe usage**: Rare. Only on specific high-impact moments.
- **Spatial**: Symmetric washes. Even coverage, immersive.
- **Energy mapping**: Energy drives color warmth and saturation, not brightness.

### 3. French Rap — Melodic (`french_melodic`)

- **Color palette**: Warm tones — gold, orange, pink, cyan accents.
- **Timing**: Hi-hat patterns drive subtle brightness pulses. Bouncy feel.
- **Contrast**: Medium. Playful variety without harshness.
- **Strobe usage**: Minimal. Occasional accent on drops.
- **Spatial**: Balanced, center-focused with warm fills.
- **Energy mapping**: Energy maps to color variety and saturation.

### 4. French Rap — Hard (`french_hard`)

- **Color palette**: Cold whites, steel blue, occasional blood red.
- **Timing**: Locked to kick and snare. Military precision.
- **Contrast**: High. Each hit is a deliberate visual punch.
- **Strobe usage**: Moderate. Controlled bursts, never chaotic.
- **Spatial**: Structured. Left-right alternation on snares.
- **Energy mapping**: Energy maps directly to brightness. Linear relationship.

### 5. European Alt Hip-Hop (`euro_alt`)

- **Color palette**: Desaturated pastels, white, occasional single saturated accent.
- **Timing**: Irregular. Deliberately breaks from beat grid for artistic effect.
- **Contrast**: Extreme restraint. Long periods of near-darkness punctuated by single moments.
- **Strobe usage**: Almost never. One strobe in an entire song is impactful.
- **Spatial**: Asymmetric. Single fixture activation. Negative space is the design.
- **Energy mapping**: Inverted relationship — high musical energy may mean lighting pulls back.

### 6. Theatrical Electronic (`theatrical`)

- **Color palette**: Full spectrum, scene-dependent. Each song section has a color world.
- **Timing**: Follows song narrative arc, not individual beats.
- **Contrast**: Dramatic but motivated. Every change serves the story.
- **Strobe usage**: Only at narrative climax. Used as exclamation mark, not texture.
- **Spatial**: Deliberate stage design. Lights create scenes and environments.
- **Energy mapping**: Emotional arc > energy level. A quiet moment can be brightly lit if narratively appropriate.

### 7. Festival EDM / Trance (`festival_edm`)

- **Color palette**: Full RGB spectrum. Bright, saturated, crowd-pleasing.
- **Timing**: Locked to build-drop structure. Build = rising, drop = explosion.
- **Contrast**: Maximum at drops. Builds use gradual increase.
- **Strobe usage**: Heavy during drops. Synchronized with kick drum.
- **Spatial**: Full coverage. All fixtures active during drops. Builds use progressive activation.
- **Energy mapping**: Direct 1:1. More energy = more light, more color, more strobes.

### 8. UK Bass / Dubstep / Grime (`uk_bass`)

- **Color palette**: Neon green, cyan, magenta, UV. Rave aesthetic.
- **Timing**: Syncopated. Responds to bass wobbles and hi-hat rolls.
- **Contrast**: High but organic. Imperfect, intentionally rough.
- **Strobe usage**: Frequent but erratic. Not perfectly on-beat.
- **Spatial**: Asymmetric, rotating. Feels like lights have their own rhythm.
- **Energy mapping**: Sub-bass energy is the primary driver, not overall energy.

## Two-Stage Genre Classification

Classification happens in two stages:

1. **Family classification** (3 classes): Hip-Hop/Rap, Electronic, Hybrid
2. **Profile classification** (8 classes): Specific profile within or across families

Output is a weighted dictionary summing to 1.0:

```
"Sao Paulo" (Weeknd):     { psych_rnb: 0.6, festival_edm: 0.2, rage_trap: 0.1, theatrical: 0.1 }
"Victory Lap 5" (Fred):   { uk_bass: 0.7, rage_trap: 0.2, festival_edm: 0.1 }
"Magnolia" (Carti):        { rage_trap: 0.9, french_hard: 0.1 }
```

## Implementation

- Base profile class: `lumina/lighting/profiles/base.py`
- Individual profiles: `lumina/lighting/profiles/<profile_id>.py`
- Profile registry: `lumina/lighting/engine.py`
- Genre classifier: `lumina/audio/genre_classifier.py`
