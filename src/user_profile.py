"""The listener's own taste profile (no Streamlit in here).

Stored in data/user_profile.json as a recommender "user_prefs" dict, the same shape
recommender.score_song takes:

    {"genre": str, "mood": str, "energy": 0-1, "valence": 0-1,
     "tempo_bpm": 60-180, "likes_acoustic": bool}

Every key is optional: leaving one out means "no preference", and the recommender then
ignores that feature (the others count for more). "likes_acoustic" is three-way: True,
False, or missing (does not mind). No file means the listener has not set a profile yet; a
file holding {} means they chose "no preference" for everything.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "user_profile.json"

TEMPO_MIN = 60
TEMPO_MAX = 180

# Where the profile page's sliders start (these are only starting positions: a feature the
# listener has no preference about is not stored at all)
DEFAULTS: Dict = {"energy": 0.5, "valence": 0.5, "tempo_bpm": 100}


def _clamp(value, low, high):
    return max(low, min(high, value))


def clean_profile(raw: Dict) -> Dict:
    """A tidy profile from whatever was given: numbers pulled into range, and anything
    blank, missing or None left out (no preference)."""
    prefs: Dict = {}
    for key in ("genre", "mood"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            prefs[key] = value.strip()
    for key in ("energy", "valence"):
        if raw.get(key) is not None:
            prefs[key] = round(_clamp(float(raw[key]), 0.0, 1.0), 2)
    if raw.get("tempo_bpm") is not None:
        prefs["tempo_bpm"] = int(_clamp(float(raw["tempo_bpm"]), TEMPO_MIN, TEMPO_MAX))
    if isinstance(raw.get("likes_acoustic"), bool):
        prefs["likes_acoustic"] = raw["likes_acoustic"]
    return prefs


def load_profile(path: Path = PROFILE_PATH) -> Optional[Dict]:
    """The saved profile, or None if there is none (or the file is unreadable)."""
    try:
        return clean_profile(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def save_profile(prefs: Dict, path: Path = PROFILE_PATH) -> Dict:
    """Cleans and writes the profile atomically (temp file, then rename). Returns what
    was saved."""
    prefs = clean_profile(prefs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(prefs, indent=1), encoding="utf-8")
    os.replace(temp, path)
    return prefs


def clear_profile(path: Path = PROFILE_PATH) -> None:
    """Forgets the saved profile."""
    Path(path).unlink(missing_ok=True)
