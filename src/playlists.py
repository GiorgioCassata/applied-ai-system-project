"""Playlist storage and automatic playlist generation (no Streamlit in here).

Every playlist is a fixed, ordered list of song ids plus a recommender "profile"
(a user_prefs dict for recommender.score_song). The profile scores each song in the
playlist and lets the player keep going with fitting songs after the last one.

Stored in data/playlists.json:
    {"playlists": {id: playlist}, "recent": [id, ...]}   # recent: newest first

playlist = {
    "id", "name", "kind": "user" | "auto", "group": "time" | "activity" | "genre" | None,
    "song_ids": [...], "profile": {...}, "created": iso, "slot": str | None,
}
Auto playlists are temporary: each has a stable id and a "slot" (e.g. the day);
when the slot changes the playlist is regenerated in place, so "recent" keeps
pointing at it.
"""

import json
import os
import random
import uuid
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    from recommender import recommend_songs, score_song, song_key
except ImportError:  # imported as part of the "src" package (tests)
    from src.recommender import recommend_songs, score_song, song_key

PLAYLISTS_PATH = Path(__file__).resolve().parent.parent / "data" / "playlists.json"

TARGET_SECONDS = 90 * 60  # auto playlists span about 90 minutes
POOL_SIZE = 150  # best-fitting songs an auto playlist is built from
SEED_CHOICES = 15  # the first song is picked at random from this many top fits
MAX_RECENT_STORED = 50
DAY_STARTS_HOUR = 5  # the auto-playlist "day" (and the morning) begins at 5am

# Time-of-day playlists, in display order: (id, title, first hour, profile).
# Hours are [start, next start); "night" runs from 21:00 past midnight to 05:00.
DAYPARTS = [
    ("morning", "Morning", 5, {"energy": 0.30, "valence": 0.35, "tempo_bpm": 90, "likes_acoustic": True}),
    ("midday", "Midday", 11, {"energy": 0.90, "valence": 0.85, "tempo_bpm": 130, "likes_acoustic": False}),
    ("afternoon", "Afternoon", 16, {"energy": 0.65, "valence": 0.75, "tempo_bpm": 110, "likes_acoustic": False, "mood": "happy"}),
    ("evening", "Evening", 19, {"energy": 0.45, "valence": 0.55, "tempo_bpm": 95, "likes_acoustic": True}),
    ("night", "Night", 21, {"energy": 0.25, "valence": 0.10, "tempo_bpm": 75, "likes_acoustic": True}),
]
# (Mood is left out of everything except the afternoon on purpose: it is the heaviest
# weight in the recommender, and e.g. "intense" or "chill" songs would pull valence
# away from the energy/valence targets those parts of the day are defined by.)

# Activity playlists: (id, title, profile)
ACTIVITIES = [
    ("work", "Work Mix", {"energy": 0.55, "valence": 0.60, "tempo_bpm": 105, "likes_acoustic": False}),
    ("study", "Study Mix", {"energy": 0.40, "valence": 0.50, "tempo_bpm": 95, "likes_acoustic": True, "mood": "focused"}),
    ("workout", "Workout Mix", {"energy": 0.92, "valence": 0.70, "tempo_bpm": 145, "likes_acoustic": False, "mood": "intense"}),
    ("meditation", "Meditation Mix", {"energy": 0.10, "valence": 0.30, "tempo_bpm": 65, "likes_acoustic": True, "mood": "relaxed"}),
    ("party", "Party Mix", {"energy": 0.88, "valence": 0.88, "tempo_bpm": 125, "likes_acoustic": False, "mood": "happy"}),
    ("relax", "Chill Mix", {"energy": 0.30, "valence": 0.55, "tempo_bpm": 85, "likes_acoustic": True, "mood": "chill"}),
    ("roadtrip", "Road Trip Mix", {"energy": 0.70, "valence": 0.75, "tempo_bpm": 115, "likes_acoustic": False}),
]


# --- storage -------------------------------------------------------------------

def _empty_store() -> Dict:
    return {"playlists": {}, "recent": []}


def load_store(path: Path = PLAYLISTS_PATH) -> Dict:
    """Reads the playlist file; a missing or unreadable file gives an empty store."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = _empty_store()
        store["playlists"] = dict(data.get("playlists", {}))
        store["recent"] = [pid for pid in data.get("recent", []) if pid in store["playlists"]]
        return store
    except (OSError, ValueError, AttributeError, TypeError):
        return _empty_store()


def save_store(store: Dict, path: Path = PLAYLISTS_PATH) -> None:
    """Writes the store atomically (temp file, then rename) so a crash cannot
    leave half a file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(store, indent=1), encoding="utf-8")
    os.replace(temp, path)


# --- profiles ------------------------------------------------------------------

def profile_from_songs(songs: List[Dict]) -> Dict:
    """A recommender profile describing a set of songs: their average energy,
    valence and tempo, whether they lean acoustic, and the most common genre/mood."""
    if not songs:
        return {}
    n = len(songs)
    profile = {
        "energy": sum(s["energy"] for s in songs) / n,
        "valence": sum(s["valence"] for s in songs) / n,
        "tempo_bpm": sum(s["tempo_bpm"] for s in songs) / n,
        "likes_acoustic": sum(s["acousticness"] for s in songs) / n >= 0.5,
    }
    for key in ("genre", "mood"):
        counts: Dict[str, int] = {}
        for s in songs:
            counts[s[key]] = counts.get(s[key], 0) + 1
        top, hits = max(counts.items(), key=lambda kv: kv[1])
        if hits / n >= 0.5:  # only when one value clearly dominates
            profile[key] = top
    return profile


def song_score(playlist: Dict, song: Dict) -> float:
    """How well one song fits the playlist's profile (0-1)."""
    return score_song(playlist.get("profile") or {}, song)[0]


# --- user playlists ------------------------------------------------------------

def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def create_playlist(
    store: Dict,
    name: str,
    song_ids: List[int],
    songs: List[Dict],
    profile: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> str:
    """Adds a user playlist and returns its id. profile defaults to the average of
    the songs (see profile_from_songs)."""
    by_id = {s["id"]: s for s in songs}
    members = [by_id[i] for i in song_ids if i in by_id]
    pid = uuid.uuid4().hex[:8]
    store["playlists"][pid] = {
        "id": pid,
        "name": name.strip(),
        "kind": "user",
        "group": None,
        "song_ids": [s["id"] for s in members],
        "profile": profile if profile is not None else profile_from_songs(members),
        "created": _now_iso(now),
        "slot": None,
    }
    return pid


def next_playlist_name(store: Dict) -> str:
    """The suggested name for a new playlist: "Playlist 1", or the lowest number not
    already taken by one of your playlists ("Playlist 2" if you have 1 and 3)."""
    taken = {p["name"].strip().lower() for p in store["playlists"].values() if p.get("kind") == "user"}
    n = 1
    while f"playlist {n}" in taken:
        n += 1
    return f"Playlist {n}"


def unique_playlist_name(store: Dict, base: str) -> str:
    """`base` if no playlist has that name yet (ignoring case), otherwise "base 2",
    "base 3", ... the first one that is free."""
    taken = {p["name"].strip().lower() for p in store["playlists"].values()}
    name, n = base.strip(), 2
    while name.lower() in taken:
        name = f"{base.strip()} {n}"
        n += 1
    return name


def add_song(store: Dict, pid: str, song_id: int) -> None:
    ids = store["playlists"][pid]["song_ids"]
    if song_id not in ids:
        ids.append(song_id)


def remove_song(store: Dict, pid: str, song_id: int) -> None:
    ids = store["playlists"][pid]["song_ids"]
    if song_id in ids:
        ids.remove(song_id)


def update_playlist(
    store: Dict, pid: str, name: str, song_ids: List[int], songs: List[Dict]
) -> None:
    """Saves an edited user playlist: its name and its songs in the given order (ids
    that are not in the library are dropped). The recommender profile that keeps it
    playing after the last song is worked out again from the new songs; with no songs
    left the old profile is kept."""
    by_id = {s["id"]: s for s in songs}
    members = [by_id[i] for i in dict.fromkeys(song_ids) if i in by_id]
    playlist = store["playlists"][pid]
    playlist["name"] = name.strip()
    playlist["song_ids"] = [s["id"] for s in members]
    if members:
        playlist["profile"] = profile_from_songs(members)


def delete_playlist(store: Dict, pid: str) -> None:
    store["playlists"].pop(pid, None)
    store["recent"] = [r for r in store["recent"] if r != pid]


def mark_played(store: Dict, pid: str) -> None:
    """Moves a playlist to the front of the recently played list."""
    if pid not in store["playlists"]:
        return
    store["recent"] = ([pid] + [r for r in store["recent"] if r != pid])[:MAX_RECENT_STORED]


def recent_playlists(store: Dict, limit: int = 6) -> List[Dict]:
    return [store["playlists"][pid] for pid in store["recent"][:limit] if pid in store["playlists"]]


# --- auto playlists ------------------------------------------------------------

def current_daypart(now: datetime) -> str:
    """Id of the daypart that `now` falls in."""
    current = DAYPARTS[-1][0]  # before 05:00 it is still night
    for pid, _title, start, _profile in DAYPARTS:
        if now.hour >= start:
            current = pid
    return current


def auto_day(now: datetime) -> str:
    """The date the auto playlists belong to; it changes at 05:00, so the night
    playlist stays the same through midnight."""
    return (now - timedelta(hours=DAY_STARTS_HOUR)).date().isoformat()


def build_mix(
    profile: Dict,
    songs: List[Dict],
    rng: random.Random,
    target_seconds: int = TARGET_SECONDS,
) -> List[int]:
    """Picks about target_seconds of songs that fit `profile`, in a flowing order.

    Candidates are the POOL_SIZE best fits (ties shuffled, so a profile with only
    a genre is not biased toward the top of the CSV). The first song is a random
    pick from the best few; each next song is the best fit given the previous one
    (recommend_songs with current_song), which keeps energy jumps and repeat
    artists down."""
    if not songs:
        return []
    shuffled = list(songs)
    rng.shuffle(shuffled)
    scored = sorted(shuffled, key=lambda s: score_song(profile, s)[0], reverse=True)
    pool = _unique_songs(scored)[:POOL_SIZE]  # one copy of each song

    current = rng.choice(pool[:SEED_CHOICES])
    picked = [current]
    total = _seconds(current)
    while total < target_seconds and len(picked) < len(pool):
        remaining = [s for s in pool if s["id"] not in {p["id"] for p in picked}]
        best = recommend_songs(profile, remaining, k=1, current_song=current)
        if not best:
            break
        current = best[0][0]
        picked.append(current)
        total += _seconds(current)
    return [s["id"] for s in picked]


ARTIST_BOOST = 0.15  # a mix from one song scores the same artist's songs 15% higher
SONG_MIX_VARIETY = 0.15  # how much "regenerate" shakes a song mix's scores


def build_song_mix(
    seed: Dict,
    songs: List[Dict],
    rng: Optional[random.Random] = None,
    target_seconds: int = TARGET_SECONDS,
    artist_boost: float = ARTIST_BOOST,
    variety: float = 0.0,
) -> List[int]:
    """A mix built entirely from one song ("create a mix with this song"): the song comes
    first, then about target_seconds of the songs that fit it best, in a flowing order.

    The song's own profile (profile_from_songs) is what everything is scored against, like
    build_mix. Songs by the same artist get a slight bias: their scores are multiplied by
    1 + artist_boost, so they tend to be picked a little earlier without taking over.
    variety (0 for none) shakes every score by up to that fraction, up or down, so that
    building the mix again gives a different one."""
    rng = rng or random.Random()
    profile = profile_from_songs([seed])

    shake = {}  # each song's random nudge, fixed for this mix so the ranking is consistent

    def biased(song: Dict, score: float) -> float:
        # (a negative score is left alone: multiplying it would make it worse, not better)
        if song["artist"] == seed["artist"] and score > 0:
            score *= 1 + artist_boost
        if variety:
            score *= 1 + shake.setdefault(song["id"], rng.uniform(-variety, variety))
        return score

    seed_key = song_key(seed)  # other copies of the seed song (other albums) are left out
    candidates = [s for s in songs if s["id"] != seed["id"] and song_key(s) != seed_key]
    rng.shuffle(candidates)  # ties are shuffled, not decided by the order of the library
    candidates.sort(key=lambda s: biased(s, score_song(profile, s)[0]), reverse=True)
    pool = _unique_songs(candidates)[:POOL_SIZE]

    picked, current, total = [seed], seed, _seconds(seed)
    while total < target_seconds and pool:
        ranked = recommend_songs(profile, pool, k=len(pool), current_song=current)
        best = max(ranked, key=lambda r: biased(r[0], r[1]))[0]
        picked.append(best)
        pool = [s for s in pool if s["id"] != best["id"]]
        current, total = best, total + _seconds(best)
    return [s["id"] for s in picked]


def remove_from_mix(
    store: Dict, pid: str, songs: List[Dict], song_id: Optional[int] = None,
    artist: Optional[str] = None, rng: Optional[random.Random] = None,
) -> int:
    """Takes one song (song_id) or every song by an artist (artist) out of an auto mix.
    The rest close up to fill the gap, and fresh songs that fit the mix's profile are
    added at the end until the mix is as long (in time) as it was.

    What was removed is remembered on the mix (excluded_song_ids / excluded_artists), so
    neither the top-up nor "regenerate" brings it back. Returns how many songs were
    removed (0 for anything that is not an auto mix, or a song/artist not in it)."""
    playlist = store["playlists"].get(pid)
    if not playlist or playlist.get("kind") != "auto":
        return 0
    by_id = {s["id"]: s for s in songs}
    ids = playlist["song_ids"]
    if artist is not None:
        gone = [i for i in ids if i in by_id and by_id[i]["artist"] == artist]
        playlist.setdefault("excluded_artists", [])
        if artist not in playlist["excluded_artists"]:
            playlist["excluded_artists"].append(artist)
    else:
        gone = [i for i in ids if i == song_id]
        if song_id is not None and song_id not in playlist.setdefault("excluded_song_ids", []):
            playlist["excluded_song_ids"].append(song_id)
    if not gone:
        return 0
    removed_seconds = sum(_seconds(by_id[i]) for i in gone if i in by_id)
    playlist["song_ids"] = [i for i in ids if i not in gone]  # the rest close up
    playlist["song_ids"] += _top_up(playlist, songs, removed_seconds, rng or random.Random())
    return len(gone)


def _allowed_songs(playlist: Dict, songs: List[Dict]) -> List[Dict]:
    """The songs a mix may use: not removed from it, and not by an artist removed from it."""
    bad_ids = set(playlist.get("excluded_song_ids", []))
    bad_artists = set(playlist.get("excluded_artists", []))
    return [s for s in songs if s["id"] not in bad_ids and s["artist"] not in bad_artists]


def _top_up(playlist: Dict, songs: List[Dict], seconds: int, rng: random.Random) -> List[int]:
    """Ids of songs that fit the playlist's profile and are not in it yet, worth about
    `seconds` in all, each the best fit after the one before (like build_mix)."""
    by_id = {s["id"]: s for s in songs}
    profile = playlist.get("profile") or {}
    have = {song_key(by_id[i]) for i in playlist["song_ids"] if i in by_id}
    candidates = [s for s in _allowed_songs(playlist, songs) if song_key(s) not in have]
    rng.shuffle(candidates)  # ties are shuffled, not decided by the order of the library
    candidates.sort(key=lambda s: score_song(profile, s)[0], reverse=True)
    pool = _unique_songs(candidates)[:POOL_SIZE]
    last = next((by_id[i] for i in reversed(playlist["song_ids"]) if i in by_id), None)
    added, total = [], 0
    while total < seconds and pool:
        best = recommend_songs(profile, pool, k=1, current_song=last)
        if not best:
            break
        last = best[0][0]
        added.append(last["id"])
        total += _seconds(last)
        pool = [s for s in pool if s["id"] != last["id"]]
    return added


def regenerate_auto_playlist(
    store: Dict, pid: str, songs: List[Dict], rng: Optional[random.Random] = None
) -> bool:
    """Builds a new set and order of songs for an auto playlist, from the same profile
    (the "regenerate" menu item). Its id, name and day stay as they are, so the new mix
    lasts until the day changes like the one it replaces. A random start is used each
    time (pass `rng` to fix it), and a few tries are made so the result differs from
    the old one. Returns False for anything that is not an auto playlist."""
    playlist = store["playlists"].get(pid)
    if not playlist or playlist.get("kind") != "auto":
        return False
    rng = rng or random.Random()
    songs = _allowed_songs(playlist, songs)  # what the listener removed stays out
    by_id = {s["id"]: s for s in songs}
    seed = by_id.get(playlist.get("seed_id"))  # a song mix is rebuilt around its own song
    song_ids = playlist["song_ids"]
    for _ in range(5):
        if seed is not None:
            song_ids = build_song_mix(seed, songs, rng, variety=SONG_MIX_VARIETY)
        else:
            song_ids = build_mix(playlist["profile"], songs, rng)
        if song_ids != playlist["song_ids"]:
            break
    playlist["song_ids"] = song_ids
    return True


def _unique_songs(songs: List[Dict]) -> List[Dict]:
    """The songs in the same order with only the first copy of each song kept (copies
    are the same title and artist, for example on a compilation album)."""
    seen = set()
    unique = []
    for song in songs:
        key = song_key(song)
        if key not in seen:
            seen.add(key)
            unique.append(song)
    return unique


def _seconds(song: Dict) -> int:
    try:
        return int(float(song.get("duration_seconds") or 0)) or 210  # unknown: ~3.5 min
    except (TypeError, ValueError):
        return 210


YOU_MIX_ID = "auto-you"  # the mix built from the listener's own profile


def _auto_specs(songs: List[Dict], genre_tiles: int, user_prefs: Optional[Dict] = None):
    """Every auto playlist as (id, name, group, profile), in display order. With
    user_prefs (the listener's saved profile) a "Made for You" mix comes first."""
    specs = [(YOU_MIX_ID, "Made for You", "you", user_prefs)] if user_prefs else []
    specs += [(f"auto-time-{pid}", title, "time", profile) for pid, title, _s, profile in DAYPARTS]
    specs += [(f"auto-act-{pid}", title, "activity", profile) for pid, title, profile in ACTIVITIES]
    counts: Dict[str, int] = {}
    for s in songs:
        counts[s["genre"]] = counts.get(s["genre"], 0) + 1
    top = sorted(counts, key=lambda g: (-counts[g], g))[:genre_tiles]
    specs += [(f"auto-genre-{g}", f"{g.title()} Mix", "genre", {"genre": g}) for g in top]
    return specs


def refresh_auto_playlists(
    store: Dict, songs: List[Dict], now: Optional[datetime] = None, genre_tiles: int = 6,
    user_prefs: Optional[Dict] = None,
) -> bool:
    """Creates auto playlists that are missing and regenerates ones whose slot is
    out of date (a new day). With user_prefs, the "Made for You" mix is kept in step with
    the listener's profile (and removed when there is none). Returns True when the store
    changed (caller saves)."""
    now = now or datetime.now()
    slot = auto_day(now)
    changed = False
    wanted = set()
    for pid, name, group, profile in _auto_specs(songs, genre_tiles, user_prefs):
        wanted.add(pid)
        existing = store["playlists"].get(pid)
        # Up to date: same day, same name and profile (so editing a profile above
        # takes effect right away instead of waiting for tomorrow).
        if (
            existing and existing.get("slot") == slot and existing.get("song_ids")
            and existing.get("profile") == profile and existing.get("name") == name
        ):
            continue
        rng = random.Random(f"{pid}|{slot}")
        store["playlists"][pid] = {
            "id": pid,
            "name": name,
            "kind": "auto",
            "group": group,
            "song_ids": build_mix(profile, songs, rng),
            "profile": profile,
            "created": _now_iso(now),
            "slot": slot,
        }
        changed = True
    # Drop auto playlists that are no longer offered (e.g. a genre fell out of the
    # top few); they are temporary, so they leave the recent list too.
    # (Song mixes are kept: they leave the recommended list on their own, see create_song_mix.)
    for pid in [
        p for p, pl in store["playlists"].items()
        if pl.get("kind") == "auto" and pl.get("group") != "song"
    ]:
        if pid not in wanted:
            delete_playlist(store, pid)
            changed = True
    return changed


# --- gradients -------------------------------------------------------------------

ACCENT_HUE = 160  # the app's teal-green accent, #17A173, is hsl(160, 75%, 36%)
ACCENT_AVOID = 45  # keep the bottom right colour at least this many degrees from it


def _hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues around the colour wheel (0-180)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def generate_gradient(seed) -> str:
    """Makes a vivid two-colour gradient from any text: the same one every time for the
    same text, a different one for different text.

    The seed picks the first hue (top left) and a second hue 60-160 degrees round the
    colour wheel (bottom right), so the colours are clearly different. The second hue is
    never within ACCENT_AVOID degrees of the app's green accent, because the green play
    button sits over that corner and would vanish into a similar colour. If the second
    hue lands in that green zone, the other side of the first hue is tried, and failing
    that it steps round the wheel to the nearest clear colour. Saturation and lightness
    vary a little between gradients."""
    h = zlib.crc32(str(seed).encode("utf-8"))
    hue1 = h % 360
    offset = 60 + (h >> 9) % 100

    def fits(hue: float) -> bool:
        return (
            _hue_distance(hue, ACCENT_HUE) > ACCENT_AVOID
            and _hue_distance(hue, hue1) >= 50
        )

    hue2 = (hue1 + offset) % 360
    if not fits(hue2):
        hue2 = (hue1 - offset) % 360
    step = 0
    while not fits(hue2) and step < 360:
        hue2 = (hue2 + 5) % 360
        step += 5

    sat = 70 + (h >> 16) % 20  # 70-89%
    light1, light2 = 52 + (h >> 20) % 8, 46 + (h >> 24) % 8
    return f"linear-gradient(135deg, hsl({hue1}, {sat}%, {light1}%), hsl({hue2}, {sat}%, {light2}%))"


# Which activity mix suits each part of the day (the second recommended playlist):
# work through the day, a workout in the afternoon, studying in the evening, and
# something chill at night.
DAYPART_ACTIVITY = {
    "morning": "work",
    "midday": "work",
    "afternoon": "workout",
    "evening": "study",
    "night": "relax",
}


def _recommended_base(store: Dict, now: Optional[datetime]) -> List[Dict]:
    """The recommended playlists that are always first: "Made for You" (if the listener has
    set up a profile), the auto mix for this time of day, and the activity mix that suits
    it (DAYPART_ACTIVITY). Anything missing from the store is skipped."""
    daypart = current_daypart(now or datetime.now())
    wanted = [YOU_MIX_ID, f"auto-time-{daypart}", f"auto-act-{DAYPART_ACTIVITY[daypart]}"]
    return [store["playlists"][pid] for pid in wanted if pid in store["playlists"]]


def song_mixes(store: Dict) -> List[Dict]:
    """The temporary mixes made with "create a mix with this song", newest first."""
    return sorted(
        (p for p in store["playlists"].values() if p.get("group") == "song"),
        key=lambda p: p.get("created", ""),
        reverse=True,
    )


def create_song_mix(
    store: Dict,
    seed: Dict,
    songs: List[Dict],
    now: Optional[datetime] = None,
    rng: Optional[random.Random] = None,
    limit: int = 6,
) -> str:
    """A temporary auto-generated mix built from one song (see build_song_mix), shown in the
    recommended playlists for as long as there is room for it. Returns its id.

    The recommended list has `limit` places. Song mixes fill the ones left after the
    always-present mixes (_recommended_base), newest first; making one more than fits
    drops the oldest. Making a mix from a song that already has one rebuilds that mix and
    moves it to the front instead of adding a second."""
    now = now or datetime.now()
    pid = f"auto-song-{seed['id']}"
    store["playlists"][pid] = {
        "id": pid,
        "name": f"{seed['title']} Mix",
        "kind": "auto",
        "group": "song",
        "song_ids": build_song_mix(seed, songs, rng),
        "profile": profile_from_songs([seed]),
        "created": _now_iso(now),
        "slot": None,
        "seed_id": seed["id"],
    }
    room = max(1, limit - len(_recommended_base(store, now)))  # the newest one always stays
    for old in song_mixes(store)[room:]:
        delete_playlist(store, old["id"])
    return pid


def recommended_playlists(store: Dict, now: Optional[datetime] = None, limit: int = 6) -> List[Dict]:
    """The home page's recommended playlists, in order:
    0. "Made for You", if the listener has set up a profile,
    1. the auto mix for the current time of day,
    2. the activity mix that suits this time of day (DAYPART_ACTIVITY),
    3. the temporary mixes made with "create a mix with this song", newest first,
    4. then the playlists you created, newest first (by creation time, whether or not
       you have played them), to fill the places up to `limit`.
    Anything missing from the store is skipped."""
    picks = _recommended_base(store, now) + song_mixes(store)
    mine = sorted(
        (p for p in store["playlists"].values() if p.get("kind") == "user"),
        key=lambda p: p.get("created", ""),
        reverse=True,
    )
    return (picks + mine)[:limit]


def auto_playlists(store: Dict, group: str) -> List[Dict]:
    """The offered auto playlists of one group, in display order."""
    order = {"time": [f"auto-time-{d[0]}" for d in DAYPARTS],
             "activity": [f"auto-act-{a[0]}" for a in ACTIVITIES]}
    if group in order:
        return [store["playlists"][p] for p in order[group] if p in store["playlists"]]
    return [p for p in store["playlists"].values() if p.get("group") == group]
