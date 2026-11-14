"""
Estimates tempo_bpm, energy, valence, acousticness, and mood for the songs in
data/songs.csv by analyzing the audio files themselves.

Run scripts/populate_songs.py first; this script fills in the placeholder
values it leaves behind. It reads each song's file_path straight from the CSV
(no folder scan). The music folder argument is only needed if the CSV holds
paths relative to it (populate_songs.py --relative).

Usage:
    python scripts/analyze_songs.py
    python scripts/analyze_songs.py --dry-run      # analyze, don't write
    python scripts/analyze_songs.py --limit 10     # first 10 only (try it out)
    python scripts/analyze_songs.py --workers 6    # parallel analysis
    python scripts/analyze_songs.py --force        # redo already-analyzed songs
    python scripts/analyze_songs.py /Volumes/MEDIA/MUSIC   # only for relative file_paths

A song counts as "not yet analyzed" while its mood is still "unknown" (the
placeholder from populate_songs.py), so re-running only does new songs.
The CSV is checkpointed as it goes, so Ctrl-C loses very little work.

Requires: pip install librosa numpy mutagen
          brew install ffmpeg   (needed for .m4a/.aac; called directly, not via librosa)

IMPORTANT: these are heuristic estimates from signal features, not a streaming service's
trained models. Tempo is the most trustworthy; valence and mood are the
roughest. The constants below are meant to be tuned.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from populate_songs import DEFAULT_CSV

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = np = None

SAMPLE_RATE = 22050
CLIP_SECONDS = 60          # analyze a clip from the middle of each song (faster, skips intros)
CHECKPOINT_EVERY = 25      # write the CSV every N analyzed songs
REMOVED_COLUMNS = ["danceability"]
OUTPUT_COLUMNS = ["mood", "energy", "tempo_bpm", "valence", "acousticness"]

TEMPO_FOLD_RANGE = (70, 160)   # halve/double tempo until it lands here (fixes octave errors)

# Krumhansl-Kessler key profiles, used to judge major vs. minor.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def scale(value: float, low: float, high: float) -> float:
    """Linearly maps [low, high] to [0, 1], clipped."""
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def fold_tempo(bpm: float) -> float:
    low, high = TEMPO_FOLD_RANGE
    while bpm < low:
        bpm *= 2
    while bpm >= high:
        bpm /= 2
    return bpm


def mood_from(energy: float, valence: float) -> str:
    """3x3 energy/valence grid, using the mood names the recommender knows."""
    e = 0 if energy < 0.40 else 1 if energy < 0.65 else 2   # low / mid / high
    v = 0 if valence < 0.40 else 1 if valence < 0.60 else 2
    grid = [
        # valence:  low          mid         high
        ["sad",         "chill",     "relaxed"],    # low energy
        ["melancholic", "focused",   "happy"],      # mid energy
        ["angry",       "intense",   "euphoric"],   # high energy
    ]
    return grid[e][v]


FFMPEG_FIRST = {".m4a", ".aac", ".mp4"}   # librosa can't reliably decode these itself
FFMPEG_LOCATIONS = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]  # Homebrew, if not on PATH


def song_length(path_str: str) -> float:
    """Length in seconds from the file header (fast, no decoding); 0.0 if unknown."""
    try:
        import mutagen
        audio = mutagen.File(path_str)
        if audio is not None and audio.info.length:
            return float(audio.info.length)
    except Exception:
        pass
    try:
        return float(librosa.get_duration(path=path_str))
    except Exception:
        return 0.0


def decode_with_ffmpeg(path_str: str, offset: float, duration: float):
    """Decodes a clip to mono float samples by calling ffmpeg directly."""
    exe = shutil.which("ffmpeg") or next((p for p in FFMPEG_LOCATIONS if os.path.exists(p)), None)
    if exe is None:
        raise RuntimeError("ffmpeg not found (install it with: brew install ffmpeg)")
    proc = subprocess.run(
        [exe, "-v", "error", "-ss", f"{offset:.2f}", "-t", f"{duration:.2f}", "-i", path_str,
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode: {proc.stderr.decode(errors='replace').strip()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32), SAMPLE_RATE


def load_clip(path_str: str):
    """Loads a CLIP_SECONDS clip from the middle of the song as (samples, sample_rate)."""
    offset = max(0.0, song_length(path_str) / 2 - CLIP_SECONDS / 2)
    if Path(path_str).suffix.lower() in FFMPEG_FIRST:
        return decode_with_ffmpeg(path_str, offset, CLIP_SECONDS)
    try:
        return librosa.load(path_str, sr=SAMPLE_RATE, mono=True, offset=offset, duration=CLIP_SECONDS)
    except Exception:
        return decode_with_ffmpeg(path_str, offset, CLIP_SECONDS)  # e.g. an mp3 libsndfile rejects


def analyze_file(path_str: str) -> dict:
    """Returns {energy, tempo_bpm, valence, acousticness, mood} for one audio file.
    Raises on unreadable files (the caller reports and skips them)."""
    y, sr = load_clip(path_str)
    if y.size < sr * 5:
        raise ValueError("audio too short or empty")
    seconds = y.size / sr

    # --- shared features ---
    harmonic, percussive = librosa.effects.hpss(y)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms_db = float(20 * np.log10(np.mean(librosa.feature.rms(y=y)) + 1e-9))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / seconds

    # --- tempo ---
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_bpm = fold_tempo(float(np.atleast_1d(tempo)[0]))

    # --- energy: loud + busy + bright ---
    energy = (
        0.4 * scale(rms_db, -35, -10)
        + 0.3 * scale(onset_rate, 0.5, 5.0)
        + 0.3 * scale(centroid, 1000, 4000)
    )

    # --- acousticness: mostly-harmonic, darker timbre ---
    h_energy, p_energy = float(np.sum(harmonic**2)), float(np.sum(percussive**2))
    harmonic_ratio = h_energy / (h_energy + p_energy + 1e-12)
    acousticness = 0.55 * scale(harmonic_ratio, 0.5, 0.95) + 0.45 * (1 - scale(centroid, 800, 3500))

    # --- valence: major vs. minor mode, faster tempo, brighter timbre ---
    chroma = np.mean(librosa.feature.chroma_stft(y=harmonic, sr=sr), axis=1)
    major = max(np.corrcoef(chroma, np.roll(MAJOR_PROFILE, i))[0, 1] for i in range(12))
    minor = max(np.corrcoef(chroma, np.roll(MINOR_PROFILE, i))[0, 1] for i in range(12))
    mode_score = scale(0.5 + (major - minor) * 1.5, 0.0, 1.0)
    valence = (
        0.5 * mode_score
        + 0.25 * scale(tempo_bpm, 60, 180)
        + 0.25 * scale(centroid, 1000, 4000)
    )

    energy, valence, acousticness = (round(float(v), 2) for v in (energy, valence, acousticness))
    return {
        "energy": energy,
        "tempo_bpm": round(tempo_bpm),
        "valence": valence,
        "acousticness": acousticness,
        "mood": mood_from(energy, valence),
    }


def save_csv(csv_path: Path, columns: list, rows: list) -> None:
    """Writes via a temp file so an interrupted write can't corrupt the CSV."""
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, nargs="?",
                        help="music folder; only needed if the CSV stores relative file_paths")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="songs.csv to update")
    parser.add_argument("--workers", type=int, default=4, help="songs analyzed in parallel (default 4)")
    parser.add_argument("--limit", type=int, help="only analyze the first N songs")
    parser.add_argument("--force", action="store_true", help="re-analyze songs that already have a mood")
    parser.add_argument("--dry-run", action="store_true", help="analyze and print, but change nothing")
    args = parser.parse_args()

    if librosa is None:
        print("This script needs librosa and numpy: pip install librosa numpy", file=sys.stderr)
        return 1
    root = args.root.expanduser().resolve() if args.root else None
    if root is not None and not root.is_dir():
        print(f"Music folder not found: {root} (is the share mounted?)", file=sys.stderr)
        return 1
    if not args.csv.exists():
        print(f"{args.csv} not found. Run scripts/populate_songs.py first.", file=sys.stderr)
        return 1

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    for column in OUTPUT_COLUMNS:
        if column not in columns:
            columns.append(column)
    dropped = [c for c in REMOVED_COLUMNS if c in columns]
    columns = [c for c in columns if c not in REMOVED_COLUMNS]
    for row in rows:
        for column in REMOVED_COLUMNS:
            row.pop(column, None)

    def resolve(file_path: str) -> str:
        p = Path(file_path.strip())
        if p.is_absolute() or root is None:
            return str(p)
        return str(root / p)

    rows_by_path = {resolve(r["file_path"]): r for r in rows if (r.get("file_path") or "").strip()}
    print(f"{len(rows)} songs in {args.csv.name}; {len(rows_by_path)} have a file_path.", flush=True)

    available = [p for p in rows_by_path if os.path.isfile(p)]
    if len(available) < len(rows_by_path):
        first_missing = next(p for p in rows_by_path if not os.path.isfile(p))
        print(f"  {len(rows_by_path) - len(available)} file(s) not found, skipping them (is the share "
              f"mounted? relative paths need the music folder argument). First missing: {first_missing}",
              flush=True)

    todo = [p for p in available if args.force or rows_by_path[p].get("mood", "unknown") in ("", "unknown")]
    print(f"{len(available) - len(todo)} already analyzed, {len(todo)} to analyze.", flush=True)
    if args.limit:
        todo = todo[: args.limit]
        print(f"  (limited to first {len(todo)})", flush=True)

    if dropped:
        print(f"Removing column(s) from the CSV: {', '.join(dropped)}", flush=True)
    if not todo and not (dropped and not args.dry_run):
        return 0

    if not args.dry_run and args.csv.exists():
        shutil.copy2(args.csv, args.csv.with_suffix(".csv.bak"))

    done = failed = 0
    executor = ProcessPoolExecutor(max_workers=max(1, args.workers))
    try:
        futures = {executor.submit(analyze_file, p): p for p in todo}
        for finished, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            row = rows_by_path[path]
            label = f"{row.get('artist', '')} / {row.get('album', '')} / {row.get('title', '')}"
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"  [{finished}/{len(todo)}] FAILED {label}: {exc}", flush=True)
                continue
            row.update(result)
            done += 1
            print(
                f"  [{finished}/{len(todo)}] {label} -> {result['mood']} "
                f"(energy {result['energy']:.2f}, valence {result['valence']:.2f}, "
                f"{result['tempo_bpm']} bpm, acoustic {result['acousticness']:.2f})",
                flush=True,
            )
            if not args.dry_run and done % CHECKPOINT_EVERY == 0:
                save_csv(args.csv, columns, rows)
                print(f"  checkpoint saved ({done} analyzed)", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; saving what has been analyzed so far...", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        if not args.dry_run:
            save_csv(args.csv, columns, rows)
        return 130
    executor.shutdown()

    print(f"Analyzed {done} song(s), {failed} failed.", flush=True)
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0
    save_csv(args.csv, columns, rows)
    print(f"Wrote {len(rows)} songs to {args.csv} (backup: {args.csv.with_suffix('.csv.bak').name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
