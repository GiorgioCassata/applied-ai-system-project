"""Album-art helpers shared by the app and scripts/populate_songs.py.

Nothing here imports Streamlit, so it can run from a plain script.
"""

import io
import os
from pathlib import Path

# Optional folder the share is mounted at (e.g. /Volumes/music). A song's
# file_path column may be absolute, or relative to this folder.
MUSIC_ROOT = os.environ.get("NOTIFY_MUSIC_ROOT", "")

# Thumbnails live here so Streamlit's static file serving can hand them to the
# browser (needs `enableStaticServing = true` in .streamlit/config.toml).
ART_DIR = Path(__file__).resolve().parent / "static" / "art"
ART_URL_PREFIX = "/app/static/art/"

COVER_NAMES = ("cover", "folder", "front", "album", "albumart")
COVER_EXTS = (".jpg", ".jpeg", ".png")


def resolve_audio_path(song):
    """Returns the Path of the song's audio file, or None if missing/unmounted."""
    raw = (song.get("file_path") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and MUSIC_ROOT:
        path = Path(MUSIC_ROOT).expanduser() / path
    try:
        return path if path.is_file() else None
    except OSError:  # e.g. the share dropped mid-check
        return None


def _embedded_art_bytes(audio_path):
    """Returns the cover image bytes embedded in the audio file's tags, or None."""
    try:
        import mutagen
        from mutagen.flac import FLAC
        from mutagen.id3 import ID3
        from mutagen.mp4 import MP4
    except ImportError:
        return None
    try:
        suffix = audio_path.suffix.lower()
        if suffix in (".m4a", ".aac", ".mp4"):
            covers = (MP4(audio_path).tags or {}).get("covr")
            return bytes(covers[0]) if covers else None
        if suffix == ".flac":
            pics = FLAC(audio_path).pictures
            return pics[0].data if pics else None
        if suffix == ".mp3":
            frames = ID3(audio_path).getall("APIC")
            return frames[0].data if frames else None
        audio = mutagen.File(audio_path)
        pics = getattr(audio, "pictures", None)
        return pics[0].data if pics else None
    except Exception:  # unreadable tags or a dropped share: just show no art
        return None


def art_jpeg_bytes(audio_path_str, size=96):
    """Returns the album art of a song file as a small JPEG (bytes), or None.

    Looks for cover.jpg/folder.jpg/etc. next to the audio file first, then falls
    back to art embedded in the file's tags.
    """
    from PIL import Image

    audio_path = Path(audio_path_str)
    data = None
    try:
        for img in sorted(audio_path.parent.iterdir()):
            if img.stem.lower() in COVER_NAMES and img.suffix.lower() in COVER_EXTS:
                data = img.read_bytes()
                break
    except OSError:
        pass
    if data is None:
        data = _embedded_art_bytes(audio_path)
    if not data:
        return None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((size, size), Image.LANCZOS)  # sharpest downscale filter
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=92)
    except Exception:
        return None
    return out.getvalue()
