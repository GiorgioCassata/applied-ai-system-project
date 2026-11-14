"""
Adds songs from a music folder to data/songs.csv, then builds their album-art
thumbnails. One run does both, so the CSV and the artwork never get out of step.

Expected layout:  <root>/ARTIST/ALBUM/SONGFILE

Usage:
    python scripts/populate_songs.py /Volumes/music
    python scripts/populate_songs.py /Volumes/music --relative   # store paths relative to root
    python scripts/populate_songs.py /Volumes/music --dry-run    # preview only
    python scripts/populate_songs.py /Volumes/music --force      # rebuild every thumbnail
    python scripts/populate_songs.py /Volumes/music --no-art     # songs only, skip the artwork

Step 1, songs. Songs already in the CSV (matched by file_path) are left alone, so it
is safe to re-run after adding music. Existing rows are never modified (except for the
art_file column, see step 2).

Audio features (energy, valence, ...) cannot be read from files, so new songs
get neutral placeholder values; scripts/analyze_songs.py estimates them afterwards.
If `mutagen` is installed (pip install mutagen), title, artist, album, genre, year,
BPM, track and disc number, and duration are read from the files. Without it (or where
a tag is missing), title/artist/album fall back to the file and folder names.

Step 2, album art. For each album (artist + album) the cover is read from the music share
(a cover/folder image in the album folder, or art embedded in the audio tags), saved as a
small JPEG in src/static/art/, and its file name is written to the `art_file` column of
every song of that album. The app then never has to touch the music share to show artwork.
Thumbnails that already exist are reused, so only new albums cost any time; albums with no
art keep a blank art_file (the app shows a gray bubble) and are tried again next run.

Progress is saved as it goes (the CSV is rewritten every few songs and every few albums,
through a temp file so an interrupted write can't corrupt it), so Ctrl-C loses very little
work and re-running carries on where it stopped.
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import mutagen
except ImportError:  # tags are optional
    mutagen = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from album_art import ART_DIR, art_jpeg_bytes, resolve_audio_path  # noqa: E402

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav"}
DEFAULT_CSV = ROOT / "data" / "songs.csv"

BASE_COLUMNS = [
    "id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
    "valence", "acousticness", "file_path",
]
# Placed right after "artist", in this order.
METADATA_COLUMNS = ["album", "release_year", "track_number", "disc_number", "duration_seconds"]
ART_COLUMN = "art_file"

# Placeholder values for features that can't be read from the files.
DEFAULTS = {
    "genre": "unknown",
    "mood": "unknown",
    "energy": 0.5,
    "tempo_bpm": 100,
    "valence": 0.5,
    "acousticness": 0.5,
}

THUMB_SIZE = 360  # album page cover is 180px; 2x keeps it sharp on high-density screens
TRIES_PER_ALBUM = 3  # songs to try in an album before giving up on finding art

SCAN_REPORT_EVERY = 200   # print a scan update every N entries examined
TAG_REPORT_EVERY = 25     # print a tag-reading update every N songs
ART_REPORT_EVERY = 25     # print an album-art update every N albums
CHECKPOINT_EVERY = 25     # write the CSV every N new songs, and every N albums with new art


def title_from_filename(path: Path) -> str:
    """'01 - Some Song.mp3' / '01. Some Song.mp3' -> 'Some Song'."""
    return re.sub(r"^\s*\d{1,3}\s*[-._)]*\s*", "", path.stem).strip() or path.stem


def leading_int(text: str):
    """'3' -> 3, '3/12' -> 3, '' or 'abc' -> None."""
    match = re.match(r"\d+", text or "")
    return int(match.group()) if match else None


def read_tags(path: Path) -> dict:
    """Returns whatever the file provides of: title, artist, album, genre,
    release_year, tempo_bpm, track_number, disc_number, duration_seconds."""
    if mutagen is None:
        return {}
    try:
        audio = mutagen.File(path, easy=True)
    except Exception:
        return {}
    if not audio:
        return {}

    tags = {}
    # Duration comes from the audio stream itself, not from a tag.
    length = getattr(getattr(audio, "info", None), "length", None)
    if length:
        tags["duration_seconds"] = round(length)
    if not audio.tags:
        return tags

    def first(key):
        values = audio.tags.get(key)
        return str(values[0]).strip() if values else ""

    for key in ("title", "artist", "album"):
        if first(key):
            tags[key] = first(key)
    track, disc = leading_int(first("tracknumber")), leading_int(first("discnumber"))
    if track is not None:
        tags["track_number"] = track
    if disc is not None:
        tags["disc_number"] = disc
    if first("genre"):
        tags["genre"] = first("genre").lower()
    # "date" may be "2019", "2019-05-03", etc.; keep just the 4-digit year.
    year = re.match(r"\d{4}", first("date") or first("originaldate"))
    if year:
        tags["release_year"] = int(year.group())
    try:
        if first("bpm"):
            tags["tempo_bpm"] = round(float(first("bpm")))
    except ValueError:
        pass
    return tags


def scan(root: Path):
    """Returns a sorted list of (path, artist, album) for each audio file under root."""
    found, examined = [], 0
    extensions = Counter()
    for path in root.rglob("*"):
        examined += 1
        extensions[path.suffix.lower() or "(no extension / folder)"] += 1
        if examined % SCAN_REPORT_EVERY == 0:
            print(
                f"  scanning... {examined} entries checked, {len(found)} songs found "
                f"(now in: {path.parent.relative_to(root)})",
                flush=True,
            )
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in parts):
            continue  # hidden files/folders, e.g. macOS ._ files
        if len(parts) >= 3:
            artist, album = parts[0], parts[1]
        elif len(parts) == 2:
            artist, album = parts[0], ""
        else:
            artist, album = "Unknown Artist", ""
        found.append((path, artist, album))
    print(f"  scan complete: {examined} entries checked, {len(found)} songs found", flush=True)
    top = ", ".join(f"{ext}: {n}" for ext, n in extensions.most_common(8))
    print(f"  most common entry types: {top}", flush=True)
    return sorted(found)


def save_csv(csv_path: Path, columns: list, rows: list) -> None:
    """Writes via a temp file so an interrupted write can't corrupt the CSV."""
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


def album_key(row):
    """Songs share art when they share an artist and album (genre if no album)."""
    return (row["artist"], row.get("album") or row["genre"].title())


def thumb_name(key):
    return hashlib.sha1("\0".join(key).encode("utf-8")).hexdigest()[:12] + ".jpg"


def audio_file(row, root: Path):
    """Where a song's file is: its file_path, taken relative to the music folder if it is
    not absolute; otherwise whatever the app itself would use. None if it can't be found."""
    stored = (row.get("file_path") or "").strip()
    if stored:
        path = Path(stored)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            return path
    return resolve_audio_path(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="music folder (the one containing ARTIST folders)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="songs.csv to update")
    parser.add_argument("--relative", action="store_true",
                        help="store file_path relative to root (set NOTIFY_MUSIC_ROOT when running the app)")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done, change nothing")
    parser.add_argument("--force", action="store_true", help="rebuild album-art thumbnails that already exist")
    parser.add_argument("--no-art", action="store_true", help="add songs only; skip the album art")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"Music folder not found: {root} (is the share mounted?)", file=sys.stderr)
        return 1

    existing, columns = [], list(BASE_COLUMNS)
    if args.csv.exists():
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            columns = list(reader.fieldnames or BASE_COLUMNS)
    for column in BASE_COLUMNS:
        if column not in columns:
            columns.append(column)
    # Keep the metadata columns together right after artist, in this order
    # (also moves them there if an earlier run put them elsewhere).
    previous = "artist"
    for column in METADATA_COLUMNS:
        if column in columns:
            columns.remove(column)
        columns.insert(columns.index(previous) + 1, column)
        previous = column
    if ART_COLUMN not in columns:
        columns.append(ART_COLUMN)

    rows = list(existing)  # grows as songs are added; this is what every checkpoint writes
    backed_up = False

    def checkpoint(message: str) -> None:
        """Saves everything so far (and, the first time, a backup of the CSV as it was)."""
        nonlocal backed_up
        if args.dry_run:
            return
        if not backed_up and args.csv.exists():
            shutil.copy2(args.csv, args.csv.with_suffix(".csv.bak"))
        backed_up = True
        save_csv(args.csv, columns, rows)
        print(f"  checkpoint saved ({message})", flush=True)

    try:
        # --- Step 1: songs ---
        known_paths = {(row.get("file_path") or "").strip() for row in existing}
        next_id = max((int(row["id"]) for row in existing if row.get("id")), default=0) + 1

        print(f"Scanning {root} ...", flush=True)
        songs = scan(root)
        to_add = [
            (path, artist, album, str(path.relative_to(root)) if args.relative else str(path))
            for path, artist, album in songs
        ]
        to_add = [item for item in to_add if item[3] not in known_paths]
        print(f"{len(songs) - len(to_add)} already in the CSV, {len(to_add)} to add.", flush=True)

        new_rows = []
        for count, (path, artist, album, stored) in enumerate(to_add, start=1):
            if count % TAG_REPORT_EVERY == 0 or count == len(to_add):
                print(f"  reading tags... {count}/{len(to_add)}", flush=True)
            tags = read_tags(path)
            row = {column: "" for column in columns}
            row.update(DEFAULTS)
            row.update(
                id=next_id,
                title=tags.get("title") or title_from_filename(path),
                artist=tags.get("artist") or artist,   # tag first, folder name as fallback
                album=tags.get("album") or album,
                file_path=stored,
            )
            row.update(
                {
                    k: v
                    for k, v in tags.items()
                    if k in ("genre", "tempo_bpm", "release_year", "track_number",
                             "disc_number", "duration_seconds")
                }
            )
            new_rows.append(row)
            rows.append(row)
            next_id += 1
            if count % CHECKPOINT_EVERY == 0:
                checkpoint(f"{count} songs added")

        print(f"Found {len(new_rows)} new song(s) under {root}.")
        for row in new_rows[:10]:
            print(f"  + {row['artist']} / {row['album']} / {row['title']}")
        if len(new_rows) > 10:
            print(f"  ... and {len(new_rows) - 10} more")
        if new_rows:
            checkpoint(f"{len(new_rows)} songs added")

        # --- Step 2: album art ---
        if args.no_art:
            print("Skipping album art (--no-art).")
        else:
            albums = {}
            for row in rows:
                albums.setdefault(album_key(row), []).append(row)
            print(f"Album art: {len(rows)} songs in {len(albums)} albums.", flush=True)

            if not args.dry_run:
                ART_DIR.mkdir(parents=True, exist_ok=True)
            built = reused = missing = since_checkpoint = 0
            for n, (key, album_songs) in enumerate(albums.items(), 1):
                name = thumb_name(key)
                target = ART_DIR / name
                if target.exists() and not args.force:
                    reused += 1
                else:
                    data, tried = None, 0
                    for song in album_songs:
                        path = audio_file(song, root)
                        if path is None:
                            continue
                        data = art_jpeg_bytes(str(path), THUMB_SIZE)
                        tried += 1
                        if data or tried >= TRIES_PER_ALBUM:
                            break
                    if not data:
                        missing += 1
                        name = ""
                    else:
                        built += 1
                        if not args.dry_run:
                            target.write_bytes(data)
                changed = False
                for song in album_songs:
                    if (song.get(ART_COLUMN) or "") != name:
                        song[ART_COLUMN] = name
                        changed = True
                since_checkpoint += changed
                if n % ART_REPORT_EVERY == 0:
                    print(f"  {n}/{len(albums)} albums...", flush=True)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    checkpoint(f"{n}/{len(albums)} albums")
                    since_checkpoint = 0
            print(f"Album art: built {built}, reused {reused}, no art found for {missing} albums.")
            if since_checkpoint:
                checkpoint("album art done")
    except KeyboardInterrupt:
        print("\nInterrupted; saving what has been done so far...", flush=True)
        checkpoint("interrupted")
        return 130

    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        print(f"{len(rows)} songs in {args.csv} (backup: {args.csv.with_suffix('.csv.bak').name}). "
              f"Thumbnails are in {ART_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
