"""Which LLM the chat agent uses (no Streamlit in here).

Stored in data/agent_settings.json, like the listener's profile is in user_profile.json:

    {"client": "ollama:<model name>" | "gemini" | "mock"}

(the choices are llm_client.CLIENT_CHOICES, which has one "ollama:<model>" per model in
OLLAMA_MODELS). A missing or unreadable file, or a choice that is no longer on offer (an
older "ollama", or a model taken off the list), means DEFAULT_CLIENT.
"""

import json
import os
from pathlib import Path
from typing import Dict

try:
    from llm_client import CLIENT_CHOICES, DEFAULT_CLIENT
except ImportError:  # imported as part of the "src" package (tests)
    from src.llm_client import CLIENT_CHOICES, DEFAULT_CLIENT

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_settings.json"


def load_settings(path: Path = SETTINGS_PATH) -> Dict:
    """The saved agent settings; anything missing or invalid falls back to the default."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        client = data.get("client")
    except (OSError, ValueError, TypeError, AttributeError):
        client = None
    return {"client": client if client in CLIENT_CHOICES else DEFAULT_CLIENT}


def save_settings(settings: Dict, path: Path = SETTINGS_PATH) -> Dict:
    """Saves the settings (an unknown client name is saved as the default) and returns
    what was saved. Written atomically (temp file, then rename)."""
    client = settings.get("client")
    tidy = {"client": client if client in CLIENT_CHOICES else DEFAULT_CLIENT}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(tidy, indent=1), encoding="utf-8")
    os.replace(temp, path)
    return tidy
