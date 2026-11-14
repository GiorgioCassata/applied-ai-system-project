import csv
from operator import itemgetter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, fields

@dataclass
class Song:
    """
    Represents a song and the attributes the recommender scores it on.
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    acousticness: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Song":
        """A Song from a library row (a dict as load_songs returns); the row's other
        columns (album, duration, file path...) are ignored."""
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    def to_dict(self) -> Dict:
        return dict(vars(self))

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.

    Every field is optional: None means "no preference", and the recommender then ignores
    that feature (the other features count for more). likes_acoustic is three-way: True
    (prefers acoustic songs), False (prefers non-acoustic songs) or None (does not mind).
    """
    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: Optional[float] = None
    likes_acoustic: Optional[bool] = None
    target_valence: Optional[float] = None
    target_tempo_bpm: Optional[float] = None

    def to_prefs(self) -> Dict:
        """The profile as the preferences dict score_song expects. Fields left at None
        (no preference) are left out of it."""
        prefs = {
            "genre": self.favorite_genre,
            "mood": self.favorite_mood,
            "energy": self.target_energy,
            "valence": self.target_valence,
            "tempo_bpm": self.target_tempo_bpm,
            "likes_acoustic": self.likes_acoustic,
        }
        return {key: value for key, value in prefs.items() if value is not None}

    @classmethod
    def from_prefs(cls, prefs: Dict) -> "UserProfile":
        """A profile from a preferences dict (the shape data/user_profile.json is saved
        in); anything it does not mention is "no preference"."""
        return cls(
            favorite_genre=prefs.get("genre"),
            favorite_mood=prefs.get("mood"),
            target_energy=prefs.get("energy"),
            likes_acoustic=prefs.get("likes_acoustic"),
            target_valence=prefs.get("valence"),
            target_tempo_bpm=prefs.get("tempo_bpm"),
        )

class Recommender:
    """
    Object-oriented front end to the recommendation logic.

    It holds a list of Song objects and answers questions for a UserProfile, using the
    same scoring as the dict-based functions below (score_song / recommend_songs), so
    the two interfaces always agree.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _prefs(user: UserProfile) -> Dict:
        """The UserProfile as the preferences dict score_song expects."""
        return user.to_prefs()

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """The k songs that best fit the user's taste, best first."""
        by_id = {song.id: song for song in self.songs}
        ranked = recommend_songs(self._prefs(user), [s.to_dict() for s in self.songs], k=k)
        return [by_id[song["id"]] for song, _score, _why in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Why this song fits (or doesn't fit) the user: the features that match well."""
        _score, reasons = score_song(self._prefs(user), song.to_dict())
        return "; ".join(reasons) if reasons else "no strong matches"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    int_fields = ("id",)
    float_fields = ("energy", "tempo_bpm", "valence", "acousticness")

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)

    print (f"Loaded songs: {len(songs)}")
    return songs

WEIGHTS = {
    "mood": 0.25,
    "energy": 0.22,
    "valence": 0.22,
    "genre": 0.15,
    "tempo_bpm": 0.10,
    "acousticness": 0.06,
}

# Moods that earn partial credit when they are not an exact match.
RELATED_MOODS = [
    {"chill", "relaxed"},
    {"chill", "focused"},
    {"happy", "euphoric"},
    {"intense", "angry"},
    {"sad", "melancholic"},
]

TEMPO_MIN_BPM = 60
TEMPO_RANGE_BPM = 120
STRONG_MATCH = 0.85


def song_key(song: Dict) -> Tuple[str, str]:
    """What makes two library rows the same song: title and artist, ignoring case. The
    same song often appears on several albums (a studio album and a compilation) under
    different ids, and the listener should not hear it twice in a mix."""
    return (str(song["title"]).strip().lower(), str(song["artist"]).strip().lower())


def _normalize_tempo(bpm: float) -> float:
    """Scales tempo to 0-1 so it is comparable with the other features."""
    return min(max((bpm - TEMPO_MIN_BPM) / TEMPO_RANGE_BPM, 0.0), 1.0)


def _mood_similarity(target: str, actual: str) -> float:
    """Returns 1.0 for an exact mood match, 0.5 for a related mood, else 0.0."""
    if target == actual:
        return 1.0
    if {target, actual} in RELATED_MOODS:
        return 0.5
    return 0.0


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Returns (score, reasons) where score is 0-1. Features missing from
    user_prefs (or set to None, meaning "no preference") are skipped and the remaining
    weights are renormalized. With no preferences at all every song scores 0.
    """
    user_prefs = {key: value for key, value in user_prefs.items() if value is not None}
    similarities: Dict[str, float] = {}
    reasons: List[str] = []

    if "genre" in user_prefs:
        similarities["genre"] = 1.0 if user_prefs["genre"] == song["genre"] else 0.0
        if similarities["genre"]:
            reasons.append(f"genre matches ({song['genre']})")

    if "mood" in user_prefs:
        similarities["mood"] = _mood_similarity(user_prefs["mood"], song["mood"])
        if similarities["mood"] == 1.0:
            reasons.append(f"mood matches ({song['mood']})")
        elif similarities["mood"] > 0:
            reasons.append(f"mood is related ({song['mood']} ~ {user_prefs['mood']})")

    for feature in ("energy", "valence"):
        if feature in user_prefs:
            similarities[feature] = 1 - abs(user_prefs[feature] - song[feature])
            if similarities[feature] >= STRONG_MATCH:
                reasons.append(
                    f"{feature} {song[feature]:.2f} is close to target {user_prefs[feature]:.2f}"
                )

    if "tempo_bpm" in user_prefs:
        similarities["tempo_bpm"] = 1 - abs(
            _normalize_tempo(user_prefs["tempo_bpm"]) - _normalize_tempo(song["tempo_bpm"])
        )
        if similarities["tempo_bpm"] >= STRONG_MATCH:
            reasons.append(f"tempo {song['tempo_bpm']:.0f} bpm is close to target")

    if "likes_acoustic" in user_prefs:
        target = 0.8 if user_prefs["likes_acoustic"] else 0.2
        similarities["acousticness"] = 1 - abs(target - song["acousticness"])
        if similarities["acousticness"] >= STRONG_MATCH:
            reasons.append("acousticness fits preference")

    total_weight = sum(WEIGHTS[f] for f in similarities)
    if total_weight == 0:
        return 0.0, reasons
    score = sum(WEIGHTS[f] * s for f, s in similarities.items()) / total_weight
    return score, reasons

FLOW_ALPHA_SCALE = 0.6
ENERGY_JUMP_LIMIT = 0.4
ENERGY_JUMP_PENALTY = 0.10
SAME_ARTIST_PENALTY = 0.05


def _flow_score(current: Dict, candidate: Dict) -> float:
    """Weighted similarity between two songs across all six features."""
    sims = {
        "genre": 1.0 if current["genre"] == candidate["genre"] else 0.0,
        "mood": _mood_similarity(current["mood"], candidate["mood"]),
        "tempo_bpm": 1 - abs(
            _normalize_tempo(current["tempo_bpm"]) - _normalize_tempo(candidate["tempo_bpm"])
        ),
    }
    for feature in ("energy", "valence", "acousticness"):
        sims[feature] = 1 - abs(current[feature] - candidate[feature])
    return sum(WEIGHTS[f] * s for f, s in sims.items())


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    current_song: Optional[Dict] = None,
    recently_played: Optional[List[int]] = None,
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Without current_song, songs are ranked purely by fit to user_prefs.
    With current_song, each candidate's profile score is blended with its
    similarity to the current song (Part 2 of the algorithm), and the song
    ids in recently_played are excluded.
    A song is also excluded when another copy of it (same title and artist, e.g. on a
    different album) is excluded, and each song appears at most once in the result.
    Returns (song_dict, score, explanation), best first.
    """
    excluded = set(recently_played or [])
    alpha = 0.0
    if current_song is not None:
        excluded.add(current_song["id"])
        seed_score, _ = score_song(user_prefs, current_song)
        alpha = FLOW_ALPHA_SCALE * seed_score

    excluded_keys = {song_key(s) for s in songs if s["id"] in excluded}
    if current_song is not None:
        excluded_keys.add(song_key(current_song))

    ranked = []
    for song in songs:
        if song["id"] in excluded or song_key(song) in excluded_keys:
            continue

        profile_score, reasons = score_song(user_prefs, song)
        score = profile_score

        if current_song is not None:
            flow = _flow_score(current_song, song)
            score = alpha * flow + (1 - alpha) * profile_score
            reasons.append(f"flow with current song {flow:.2f}")

            if abs(song["energy"] - current_song["energy"]) > ENERGY_JUMP_LIMIT:
                score -= ENERGY_JUMP_PENALTY
                reasons.append("penalty: large energy jump")
            if song["artist"] == current_song["artist"]:
                score -= SAME_ARTIST_PENALTY
                reasons.append("penalty: same artist as current song")

        ranked.append((song, score, "; ".join(reasons)))

    ranked.sort(key=itemgetter(1), reverse=True)
    seen = set()
    unique = []
    for entry in ranked:
        key = song_key(entry[0])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique[:k]
