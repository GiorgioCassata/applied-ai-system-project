"""The tools the chat agent (music_agent.py) can call. No Streamlit in here.

Each tool is a plain function decorated with @tool. The decorator builds the JSON schema
the model is shown (name, description, parameters) from the function itself:

    * the first line(s) of the docstring are the description the model reads
    * the "Args:" section of the docstring describes each parameter
    * type hints give the parameter types; a parameter with a default (or Optional[...])
      is not required
    * the first parameter, `ctx`, is the AgentContext and is never shown to the model

So a tool is defined once, and its schema can never drift away from its code.

Tools use the same library functions as the buttons in the app (playlists.py and the
PlayQueue class), not the app.py callbacks, which also show toasts and change the page.
Whatever a tool does is therefore exactly what the same button would do.

Every tool returns a small JSON-friendly dict, and never raises: a problem comes back as
{"ok": False, "error": "..."} so the model can read it and try something else. Positions
(in a playlist or in up next) count from 1, as a listener would say them.
"""

import inspect
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

try:
    import playlists as plib
    import search
    from play_queue import PlayQueue, REPEAT_MODES
    from recommender import song_key
except ImportError:  # imported as part of the "src" package (tests)
    from src import playlists as plib
    from src import search
    from src.play_queue import PlayQueue, REPEAT_MODES
    from src.recommender import song_key

MAX_RESULTS = 25  # most songs one tool call returns, so a reply stays small for the model
MAX_QUEUE_SHOWN = 25


# --- what the tools work on --------------------------------------------------------

@dataclass
class AgentContext:
    """Everything the tools need from the running app, passed in by app.py.

    songs:       the library (every song as a dict)
    get_queue:   returns the session's PlayQueue, or None when nothing has been played
    playable:    True for a song whose audio file exists (see album_art.resolve_audio_path)
    load_store:  returns the playlist store (app.get_store, which also refreshes the mixes)
    save_store:  writes the store back
    start_playback: starts a new play queue on a song id and plays it (app.play_song)
    start_album:    plays an album by name, from track 1 (app.play_album)
    start_playlist: plays a playlist by id, from its first playable song (app.play_playlist)
                 (each of these is None when the app cannot start playback that way)
    set_repeat:  sets the repeat mode for the playing queue and the next ones (app.set_repeat_mode)
    set_shuffle: turns shuffle on or off the same way (app.set_shuffle_mode)
                 (when None, the tools change only the queue that is playing)
    """

    songs: List[Dict]
    get_queue: Callable[[], Optional[PlayQueue]]
    playable: Callable[[Dict], bool] = lambda _song: True
    load_store: Callable[[], Dict] = plib.load_store
    save_store: Callable[[Dict], None] = plib.save_store
    start_playback: Optional[Callable[[int], None]] = None
    start_album: Optional[Callable[[str], None]] = None
    start_playlist: Optional[Callable[[str], None]] = None
    set_repeat: Optional[Callable[[str], None]] = None
    set_shuffle: Optional[Callable[[bool], None]] = None

    def song(self, song_id: int) -> Optional[Dict]:
        return next((s for s in self.songs if s["id"] == song_id), None)


# --- the @tool decorator -----------------------------------------------------------

_JSON_TYPES = {int: "integer", float: "number", bool: "boolean", str: "string"}


def _json_schema(tp: Any) -> Dict:
    """The JSON schema of a type hint (int, str, Optional[int], List[int], ...)."""
    origin = get_origin(tp)
    if origin is Union:
        inner = [a for a in get_args(tp) if a is not type(None)]
        return _json_schema(inner[0]) if inner else {"type": "string"}
    if origin in (list, List):
        args = get_args(tp)
        return {"type": "array", "items": _json_schema(args[0]) if args else {"type": "string"}}
    return {"type": _JSON_TYPES.get(tp, "string")}


def _split_docstring(doc: str):
    """(description, {param: description}) from a docstring with an "Args:" section."""
    description, params, in_args = [], {}, False
    for line in inspect.cleandoc(doc or "").splitlines():
        if line.strip().lower() in ("args:", "arguments:"):
            in_args = True
        elif in_args:
            match = re.match(r"\s*(\w+)\s*:\s*(.+)", line)
            if match:
                params[match.group(1)] = match.group(2).strip()
        else:
            description.append(line.strip())
    return " ".join(part for part in description if part), params


def _coerce(value: Any, schema: Dict) -> Any:
    """Local models often send numbers as strings and lists as JSON text; fix those."""
    kind = schema.get("type")
    if kind == "integer" and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
            return int(value)
    elif kind == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif kind == "boolean" and isinstance(value, str):
        if value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    elif kind == "array":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = [v for v in re.split(r"[,\s]+", value.strip("[] ")) if v]
        if isinstance(value, list):
            return [_coerce(v, schema.get("items", {})) for v in value]
    return value


def _matches(value: Any, schema: Dict) -> bool:
    kind = schema.get("type")
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list) and all(_matches(v, schema.get("items", {})) for v in value)
    return True


@dataclass
class Tool:
    """A function the model may call, with the schema it is described to the model by."""

    name: str
    description: str
    func: Callable
    parameters: Dict  # JSON schema of the arguments (without ctx)

    def schema(self) -> Dict:
        """The tool as the model is shown it (the format Ollama and Gemini both accept)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, ctx: AgentContext, arguments: Optional[Dict]) -> Dict:
        """Calls the tool with the model's arguments, after checking them. Never raises."""
        arguments = dict(arguments or {})
        props = self.parameters["properties"]
        unknown = sorted(set(arguments) - set(props))
        if unknown:
            return _error(f"Unknown argument(s) {unknown}. Valid arguments: {sorted(props)}.")
        missing = [p for p in self.parameters["required"] if p not in arguments]
        if missing:
            return _error(f"Missing required argument(s): {missing}.")
        for name in list(arguments):
            if arguments[name] is None:  # a model sending null means "not given"
                del arguments[name]
                continue
            arguments[name] = _coerce(arguments[name], props[name])
            if not _matches(arguments[name], props[name]):
                return _error(
                    f"Argument '{name}' must be of type {props[name]['type']}, "
                    f"got {arguments[name]!r}."
                )
        try:
            return self.func(ctx, **arguments)
        except Exception as exc:  # a tool must never take the chat down
            return _error(f"{type(exc).__name__}: {exc}")


TOOLS: Dict[str, Tool] = {}  # every tool, by name, in the order they are defined

# The tools that only look things up. Every other tool does something (plays, queues,
# changes a playlist): the agent uses the difference to tell a reply that claims an
# action from one the tools actually backed up.
READ_ONLY_TOOLS = frozenset({
    "find_song", "search_songs", "find_album", "find_songs_by_feel", "list_genres_and_moods", "read_song",
    "list_playlists", "read_playlist", "find_index", "read_queue",
})


def is_action(name: str) -> bool:
    """Whether the named tool changes something (as opposed to only looking)."""
    return name in TOOLS and name not in READ_ONLY_TOOLS


def tool(func: Callable) -> Tool:
    """Turns a function into a Tool (see the top of this file) and registers it."""
    description, param_docs = _split_docstring(func.__doc__)
    hints = get_type_hints(func)
    properties, required = {}, []
    for name, param in list(inspect.signature(func).parameters.items())[1:]:  # skip ctx
        schema = _json_schema(hints.get(name, str))
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty and get_origin(hints.get(name)) is not Union:
            required.append(name)
    built = Tool(
        name=func.__name__,
        description=description or "No description provided.",
        func=func,
        parameters={"type": "object", "properties": properties, "required": required},
    )
    TOOLS[built.name] = built
    return built


def tool_schemas() -> List[Dict]:
    """Every tool's schema, for the model."""
    return [t.schema() for t in TOOLS.values()]


def run_tool(ctx: AgentContext, name: str, arguments: Optional[Dict]) -> Dict:
    """Runs the named tool; an unknown name comes back as an error, not an exception."""
    found = TOOLS.get(name)
    if found is None:
        return _error(f"There is no tool called '{name}'. Available tools: {sorted(TOOLS)}.")
    return found.run(ctx, arguments)


# --- small helpers -------------------------------------------------------------------

def _error(message: str) -> Dict:
    return {"ok": False, "error": message}


def _brief(song: Dict) -> Dict:
    """A song as the model sees it in lists: enough to tell versions apart, nothing more."""
    return {"id": song["id"], "title": song["title"], "artist": song["artist"],
            "album": song.get("album") or ""}


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _playlist(ctx: AgentContext, store: Dict, playlist_id: str):
    """(playlist, None) or (None, error dict)."""
    found = store["playlists"].get(playlist_id)
    if found is None:
        return None, _error(
            f"No playlist with id '{playlist_id}'. Use list_playlists to see the real ids."
        )
    return found, None


def _user_playlist(ctx: AgentContext, store: Dict, playlist_id: str):
    """Like _playlist, but only playlists of the listener's own can be edited: mixes are
    regenerated by the app, so edits to them would be lost."""
    found, problem = _playlist(ctx, store, playlist_id)
    if problem:
        return None, problem
    if found.get("kind") != "user":
        return None, _error(
            f'"{found["name"]}" is an auto-generated mix and cannot be edited. '
            "Use save_mix_as_playlist to make an editable copy first."
        )
    return found, None


def _save_songs(ctx: AgentContext, store: Dict, playlist: Dict, song_ids: List[int]) -> None:
    """Gives a user playlist these songs, in this order, and saves (the playlist's
    recommender profile is worked out again, like in the playlist editor)."""
    plib.update_playlist(store, playlist["id"], playlist["name"], song_ids, ctx.songs)
    ctx.save_store(store)


def _active_queue(ctx: AgentContext):
    """(queue, None), or (None, error dict) when nothing has been played yet."""
    queue = ctx.get_queue()
    if queue is None:
        return None, _error("Nothing is playing, so there is no queue. Use play_song to start a song first, or ask the listener what to play.")
    return queue, None


# --- looking things up ---------------------------------------------------------------

@tool
def find_song(
    ctx: AgentContext,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    release_year: Optional[int] = None,
) -> Dict:
    """Searches the library for songs and returns their ids. Give at least one of the
    filters; the more you give, the fewer (better) results. The same song can exist on
    several albums with different ids, so check the album when it matters. It matches
    text anywhere (and falls back to fuzzy matching), so for "starts with", "ends with" or
    year ranges use search_songs instead. Always use
    this to get a song_id; never guess one.
    Args:
        title: part of the song title, e.g. "hello"
        artist: part of the artist name, e.g. "adele"
        album: part of the album name
        release_year: the exact release year, e.g. 1995
    """
    if not any([title, artist, album, release_year]):
        return _error("Give at least one of title, artist, album or release_year.")

    def contains(song: Dict, field: str, wanted: Optional[str]) -> bool:
        return not wanted or wanted.strip().lower() in str(song.get(field) or "").lower()

    def in_year(song: Dict) -> bool:
        try:
            return not release_year or int(float(song.get("release_year") or 0)) == release_year
        except (TypeError, ValueError):
            return False

    found = [
        s for s in ctx.songs
        if contains(s, "title", title) and contains(s, "artist", artist)
        and contains(s, "album", album) and in_year(s)
    ]
    fuzzy = False
    if not found and not release_year:  # no exact hit: fall back to the app's search
        query = " ".join(x for x in (title, artist, album) if x)
        found = search.search_songs(query, ctx.songs)
        fuzzy = bool(found)
    found.sort(key=lambda s: (s["artist"].lower(), s["title"].lower(), s.get("album") or ""))
    result = {
        "ok": True,
        "count": len(found),
        "songs": [dict(_brief(s), release_year=s.get("release_year")) for s in found[:MAX_RESULTS]],
    }
    if len(found) > MAX_RESULTS:
        result["note"] = f"Only the first {MAX_RESULTS} of {len(found)} are shown; narrow the search."
    if fuzzy:
        result["note"] = "No exact match; these are the closest matches."
    return result


def _text_of(song: Dict, field: str) -> str:
    return str(song.get(field) or "")


def _starts(text: str, prefix: str) -> bool:
    """Whether text starts with prefix, ignoring case and any punctuation in front, so
    "'Til Tuesday" and "(Don't Fear) The Reaper" start with T and D."""
    return text.lower().lstrip(_NOT_ALNUM).startswith(prefix.strip().lower().lstrip(_NOT_ALNUM))


_NOT_ALNUM = "".join(chr(c) for c in range(128) if not chr(c).isalnum())


@tool
def search_songs(
    ctx: AgentContext,
    title_starts_with: Optional[str] = None,
    title_ends_with: Optional[str] = None,
    title_contains: Optional[str] = None,
    title_regex: Optional[str] = None,
    artist_starts_with: Optional[str] = None,
    artist_contains: Optional[str] = None,
    album_starts_with: Optional[str] = None,
    album_contains: Optional[str] = None,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    sort_by: str = "title",
    limit: int = 10,
    offset: int = 0,
    playable_only: bool = True,
) -> Dict:
    """A precise search of the library, for requests find_song cannot express: songs whose
    title STARTS with a letter or word ("a song that starts with Q"), ends with one, an
    artist whose name starts with a letter, a range of years ("from the 80s"), and any mix
    of these. Every filter given has to match (case ignored). Unlike find_song there is no
    fuzzy fallback: an empty result means nothing matches. One copy of each song is
    returned. Starts-with ignores punctuation in front of the text, so "'Til Tuesday"
    starts with T, but a leading "The" counts ("The Reaper" starts with T, not R).
    Args:
        title_starts_with: what the title begins with, a letter or more, e.g. "q" or "love"
        title_ends_with: what the title ends with
        title_contains: text anywhere in the title
        title_regex: a regular expression the title must match, for anything else, e.g. "^\\w+$" for one-word titles
        artist_starts_with: what the artist's name begins with
        artist_contains: text anywhere in the artist's name
        album_starts_with: what the album's name begins with
        album_contains: text anywhere in the album's name
        genre: part of a genre name, e.g. "rock" also finds "alternative rock"
        mood: an exact mood (see list_genres_and_moods)
        min_year: earliest release year
        max_year: latest release year (1989 with min_year 1980 for "the 80s")
        sort_by: "title" (the default), "artist", "year", "newest" or "random"
        limit: how many songs to return, up to 25
        offset: how many matches to skip, to see the next page of a long result
        playable_only: leave out songs with no audio file (default true)
    """
    filters = [title_starts_with, title_ends_with, title_contains, title_regex, artist_starts_with,
               artist_contains, album_starts_with, album_contains, genre, mood, min_year, max_year]
    if all(f in (None, "") for f in filters):
        return _error("Give at least one filter, e.g. title_starts_with or artist_starts_with.")
    sorts = ("title", "artist", "year", "newest", "random")
    if sort_by not in sorts:
        return _error(f"sort_by must be one of {list(sorts)}.")
    if min_year is not None and max_year is not None and min_year > max_year:
        return _error("min_year is higher than max_year, so nothing can match.")
    pattern = None
    if title_regex:
        try:
            pattern = re.compile(title_regex, re.IGNORECASE)
        except re.error as problem:
            return _error(f"title_regex is not a valid regular expression: {problem}.")

    def year(song: Dict) -> int:
        try:
            return int(float(song.get("release_year") or 0))
        except (TypeError, ValueError):
            return 0

    def fits(song: Dict) -> bool:
        title, artist = _text_of(song, "title"), _text_of(song, "artist")
        album = _text_of(song, "album") or _text_of(song, "genre").title()
        if playable_only and not ctx.playable(song):
            return False
        if title_starts_with and not _starts(title, title_starts_with):
            return False
        if title_ends_with and not title.strip().lower().rstrip(_NOT_ALNUM).endswith(
                title_ends_with.strip().lower().rstrip(_NOT_ALNUM)):
            return False
        if title_contains and title_contains.strip().lower() not in title.lower():
            return False
        if pattern and not pattern.search(title):
            return False
        if artist_starts_with and not _starts(artist, artist_starts_with):
            return False
        if artist_contains and artist_contains.strip().lower() not in artist.lower():
            return False
        if album_starts_with and not _starts(album, album_starts_with):
            return False
        if album_contains and album_contains.strip().lower() not in album.lower():
            return False
        if genre and _norm(genre) not in _norm(song.get("genre") or ""):
            return False
        if mood and not _same(mood, song.get("mood") or ""):
            return False
        if min_year is not None or max_year is not None:
            y = year(song)
            if not y or (min_year is not None and y < min_year) or (max_year is not None and y > max_year):
                return False
        return True

    found, seen = [], set()
    for song in ctx.songs:  # one copy of each song (the same song can be on several albums)
        if fits(song) and song_key(song) not in seen:
            seen.add(song_key(song))
            found.append(song)
    if sort_by == "random":
        random.shuffle(found)
    else:
        keys = {
            "title": lambda s: (s["title"].lower(), s["artist"].lower()),
            "artist": lambda s: (s["artist"].lower(), s["title"].lower()),
            "year": lambda s: (year(s) or 9999, s["title"].lower()),
            "newest": lambda s: (-year(s), s["title"].lower()),
        }
        found.sort(key=keys[sort_by])
    start = max(0, offset)
    shown = found[start:start + max(1, min(limit, MAX_RESULTS))]
    result = {
        "ok": True,
        "count": len(found),
        "songs": [dict(_brief(s), release_year=s.get("release_year")) for s in shown],
    }
    if start + len(shown) < len(found):
        result["note"] = (f"Showing {start + 1}-{start + len(shown)} of {len(found)}; "
                          f"use offset={start + len(shown)} for more, or add filters.")
    return result


def _norm(text: str) -> str:
    """Letters and digits only, lowercase, so "Hip Hop" and "hip-hop" compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _seconds(song: Dict) -> int:
    try:
        return int(float(song.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        return 0


FEEL_SORTS = {
    "random": None,
    "valence_high": lambda s: -s["valence"],
    "valence_low": lambda s: s["valence"],
    "shortest": lambda s: _seconds(s),
    "longest": lambda s: -_seconds(s),
}


@tool
def list_genres_and_moods(ctx: AgentContext) -> Dict:
    """Lists the genres and moods the library really has, with how many songs each has,
    most common first. Use it before find_songs_by_feel when unsure what to ask for.
    """
    def counted(field: str) -> List[Dict]:
        counts = Counter(s[field] for s in ctx.songs if s.get(field))
        return [{"name": n, "song_count": c} for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {"ok": True, "genres": counted("genre"), "moods": counted("mood")}


@tool
def find_songs_by_feel(
    ctx: AgentContext,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    min_valence: Optional[float] = None,
    max_valence: Optional[float] = None,
    min_minutes: Optional[float] = None,
    max_minutes: Optional[float] = None,
    sort_by: str = "random",
    limit: int = 10,
) -> Dict:
    """Finds songs that can be played by how they feel, for vague requests like "something
    sad", "an upbeat rock song" or "a short one". Give at least one filter; they all have
    to match. Valence is how positive a song sounds, from 0 (sad, dark) to 1 (happy,
    upbeat). Only songs with an audio file are returned, one copy of each song.
    Args:
        genre: part of a genre name, e.g. "rock" also finds "alternative rock"
        mood: an exact mood, e.g. "sad", "chill", "euphoric" (see list_genres_and_moods)
        min_valence: lowest valence wanted, 0 to 1 (0.7 and up is upbeat)
        max_valence: highest valence wanted, 0 to 1 (0.3 and down is sad or dark)
        min_minutes: shortest length wanted, in minutes
        max_minutes: longest length wanted, in minutes (3 for "under three minutes")
        sort_by: "random" (the default, so repeat requests vary), "valence_high", "valence_low", "shortest" or "longest"
        limit: how many songs to return, up to 25
    """
    if all(x is None for x in (genre, mood, min_valence, max_valence, min_minutes, max_minutes)):
        return _error("Give at least one filter: genre, mood, a valence range or a length range.")
    if sort_by not in FEEL_SORTS:
        return _error(f"sort_by must be one of {sorted(FEEL_SORTS)}.")
    for low, high, what in ((min_valence, max_valence, "valence"), (min_minutes, max_minutes, "minutes")):
        if low is not None and high is not None and low > high:
            return _error(f"min_{what} is higher than max_{what}, so nothing can match.")

    if genre and not any(_norm(genre) in _norm(s["genre"]) for s in ctx.songs):
        names = [g["name"] for g in list_genres_and_moods.func(ctx)["genres"][:12]]
        return _error(f'No genre like "{genre}". Common genres: {names}. See list_genres_and_moods.')
    if mood and not any(_same(mood, s["mood"]) for s in ctx.songs):
        names = [m["name"] for m in list_genres_and_moods.func(ctx)["moods"]]
        return _error(f'No mood called "{mood}". Moods: {names}.')

    def fits(song: Dict) -> bool:
        if not ctx.playable(song):
            return False
        if genre and _norm(genre) not in _norm(song["genre"]):
            return False
        if mood and not _same(mood, song["mood"]):
            return False
        if min_valence is not None and song["valence"] < min_valence:
            return False
        if max_valence is not None and song["valence"] > max_valence:
            return False
        if min_minutes is not None or max_minutes is not None:
            seconds = _seconds(song)
            if not seconds:  # an unknown length cannot satisfy a length filter
                return False
            if min_minutes is not None and seconds < min_minutes * 60:
                return False
            if max_minutes is not None and seconds > max_minutes * 60:
                return False
        return True

    found, seen = [], set()
    for song in ctx.songs:  # one copy of each song (the same song can be on several albums)
        if fits(song) and song_key(song) not in seen:
            seen.add(song_key(song))
            found.append(song)
    if FEEL_SORTS[sort_by]:
        found.sort(key=FEEL_SORTS[sort_by])
    else:
        random.shuffle(found)
    shown = found[:max(1, min(limit, MAX_RESULTS))]
    return {
        "ok": True,
        "count": len(found),
        "songs": [
            dict(_brief(s), genre=s["genre"], mood=s["mood"], valence=round(s["valence"], 2),
                 duration_seconds=_seconds(s))
            for s in shown
        ],
    }


@tool
def read_song(ctx: AgentContext, song_id: int) -> Dict:
    """Returns everything known about one song: title, artist, album, release year, track
    and disc number, duration, genre, mood and its audio features (energy, tempo_bpm,
    valence, acousticness, each 0-1 except tempo).
    Args:
        song_id: the id of the song
    """
    song = ctx.song(song_id)
    if song is None:
        return _error(f"No song with id {song_id}. Use find_song to look songs up.")
    keep = ("id", "title", "artist", "album", "release_year", "track_number", "disc_number",
            "duration_seconds", "genre", "mood", "energy", "tempo_bpm", "valence", "acousticness")
    return {"ok": True, "song": {k: song.get(k) for k in keep}}


def _album_key(song: Dict) -> str:
    """The album a song is filed under, the way the app does it (a song with no album is
    filed under its genre)."""
    return song.get("album") or song["genre"].title()


def _albums(ctx: AgentContext) -> Dict[str, List[Dict]]:
    albums: Dict[str, List[Dict]] = {}
    for song in ctx.songs:
        albums.setdefault(_album_key(song), []).append(song)
    return albums


@tool
def find_album(ctx: AgentContext, name: Optional[str] = None, artist: Optional[str] = None) -> Dict:
    """Searches the library for albums and returns their exact names (what play_album
    needs). Give at least one of the filters. An album's songs can be listed with
    find_song(album=...).
    Args:
        name: part of the album name
        artist: part of an artist's name; albums with a song by them are returned
    """
    if not (name or artist):
        return _error("Give at least one of name or artist.")
    found = []
    for album, songs in _albums(ctx).items():
        if name and name.strip().lower() not in album.lower():
            continue
        if artist and not any(artist.strip().lower() in s["artist"].lower() for s in songs):
            continue
        artists = {s["artist"] for s in songs}
        years = []
        for s in songs:
            try:
                year = int(float(s.get("release_year") or 0))
            except (TypeError, ValueError):
                year = 0
            if year:
                years.append(year)
        found.append({
            "album": album,
            "artist": next(iter(artists)) if len(artists) == 1 else "Various artists",
            "song_count": len(songs),
            "release_year": min(years) if years else None,
        })
    found.sort(key=lambda a: (a["artist"].lower(), a["album"].lower()))
    result = {"ok": True, "count": len(found), "albums": found[:MAX_RESULTS]}
    if len(found) > MAX_RESULTS:
        result["note"] = f"Only the first {MAX_RESULTS} of {len(found)} are shown; narrow the search."
    return result


@tool
def list_playlists(ctx: AgentContext, kind: Optional[str] = None) -> Dict:
    """Lists playlists with their ids, names and sizes. Kinds: "user" (the listener's own,
    editable) and "auto" (mixes the app generates, which refresh through the day).
    Args:
        kind: "user" or "auto"; leave out for both
    """
    if kind not in (None, "user", "auto"):
        return _error('kind must be "user" or "auto".')
    store = ctx.load_store()
    rows = [
        {"id": p["id"], "name": p["name"], "kind": p.get("kind"), "song_count": len(p["song_ids"])}
        for p in store["playlists"].values()
        if kind is None or p.get("kind") == kind
    ]
    return {"ok": True, "count": len(rows), "playlists": rows}


@tool
def read_playlist(ctx: AgentContext, playlist_id: str) -> Dict:
    """Returns a playlist's name, kind and its songs in order, each with its 1-based
    position.
    Args:
        playlist_id: the playlist's id (from list_playlists)
    """
    playlist, problem = _playlist(ctx, ctx.load_store(), playlist_id)
    if problem:
        return problem
    songs = []
    for position, song_id in enumerate(playlist["song_ids"], start=1):
        song = ctx.song(song_id)
        if song:
            songs.append(dict(_brief(song), position=position))
    return {"ok": True, "id": playlist["id"], "name": playlist["name"],
            "kind": playlist.get("kind"), "song_count": len(songs), "songs": songs}


@tool
def find_index(ctx: AgentContext, song_id: int, playlist_id: str) -> Dict:
    """Finds where a song is in a playlist: its 1-based position, or null if it is not in it.
    Args:
        song_id: the id of the song
        playlist_id: the playlist's id
    """
    playlist, problem = _playlist(ctx, ctx.load_store(), playlist_id)
    if problem:
        return problem
    ids = playlist["song_ids"]
    return {"ok": True, "position": ids.index(song_id) + 1 if song_id in ids else None,
            "playlist_size": len(ids)}


# --- changing playlists --------------------------------------------------------------

@tool
def create_playlist(ctx: AgentContext, name: str, song_ids: List[int]) -> Dict:
    """Creates a new playlist of the listener's own from songs, in the given order. If the
    name is taken a number is added to it; the name actually used is returned.
    Args:
        name: the playlist's name
        song_ids: song ids in play order (from find_song); duplicates are dropped
    """
    if not name.strip():
        return _error("The playlist needs a name.")
    ids = list(dict.fromkeys(song_ids))
    unknown = [i for i in ids if ctx.song(i) is None]
    if unknown:
        return _error(f"Unknown song id(s) {unknown}; nothing was created. Use find_song for real ids.")
    store = ctx.load_store()
    final_name = plib.unique_playlist_name(store, name)
    pid = plib.create_playlist(store, final_name, ids, ctx.songs)
    ctx.save_store(store)
    return {"ok": True, "playlist_id": pid, "name": final_name, "song_count": len(ids)}


@tool
def save_mix_as_playlist(ctx: AgentContext, playlist_id: str) -> Dict:
    """Saves an auto-generated mix as a playlist of the listener's own (named "My <mix
    name>"), with the same songs. The mix itself carries on and still refreshes.
    Args:
        playlist_id: the id of the mix (kind "auto")
    """
    store = ctx.load_store()
    mix, problem = _playlist(ctx, store, playlist_id)
    if problem:
        return problem
    if mix.get("kind") != "auto":
        return _error(f'"{mix["name"]}" is already one of the listener\'s own playlists.')
    name = plib.unique_playlist_name(store, f"My {mix['name']}")
    pid = plib.create_playlist(store, name, mix["song_ids"], ctx.songs, profile=mix.get("profile"))
    ctx.save_store(store)
    return {"ok": True, "playlist_id": pid, "name": name, "song_count": len(mix["song_ids"])}


@tool
def regenerate_mix(ctx: AgentContext, playlist_id: str) -> Dict:
    """Builds a new set and order of songs for an auto-generated mix, from the same
    profile. Only works on mixes (kind "auto"), not on the listener's own playlists.
    Args:
        playlist_id: the id of the mix
    """
    store = ctx.load_store()
    mix, problem = _playlist(ctx, store, playlist_id)
    if problem:
        return problem
    if not plib.regenerate_auto_playlist(store, playlist_id, ctx.songs):
        return _error(f'"{mix["name"]}" is not a mix, so it cannot be regenerated.')
    ctx.save_store(store)
    return {"ok": True, "name": mix["name"], "song_count": len(mix["song_ids"])}


@tool
def delete_playlist(ctx: AgentContext, playlist_id: str) -> Dict:
    """Permanently deletes one of the listener's own playlists. Cannot be undone, and
    mixes cannot be deleted.
    Args:
        playlist_id: the id of the playlist
    """
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    plib.delete_playlist(store, playlist_id)
    ctx.save_store(store)
    return {"ok": True, "deleted": playlist["name"]}


@tool
def rename_playlist(ctx: AgentContext, playlist_id: str, name: str) -> Dict:
    """Renames one of the listener's own playlists. Two playlists cannot share a name.
    Args:
        playlist_id: the id of the playlist
        name: the new name
    """
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    name = name.strip()
    if not name:
        return _error("The playlist needs a name.")
    if any(_same(p["name"], name) for pid, p in store["playlists"].items() if pid != playlist_id):
        return _error(f'Another playlist is already called "{name}". Pick a different name.')
    playlist["name"] = name
    ctx.save_store(store)
    return {"ok": True, "name": name}


@tool
def add_song_to_playlist(ctx: AgentContext, song_id: int, playlist_id: str) -> Dict:
    """Adds a song to the end of one of the listener's own playlists.
    Args:
        song_id: the id of the song
        playlist_id: the id of the playlist
    """
    song = ctx.song(song_id)
    if song is None:
        return _error(f"No song with id {song_id}. Use find_song for real ids.")
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    ids = playlist["song_ids"]
    if song_id in ids:
        return _error(f'"{song["title"]}" is already in "{playlist["name"]}".')
    # The same song on another album counts as already there
    twin = next((i for i in ids if ctx.song(i) and song_key(ctx.song(i)) == song_key(song)), None)
    if twin is not None:
        return _error(f'"{song["title"]}" is already in "{playlist["name"]}" (id {twin}, another album).')
    _save_songs(ctx, store, playlist, ids + [song_id])
    return {"ok": True, "playlist": playlist["name"], "song_count": len(ids) + 1}


@tool
def remove_song_from_playlist(ctx: AgentContext, song_id: int, playlist_id: str) -> Dict:
    """Removes a song from one of the listener's own playlists.
    Args:
        song_id: the id of the song
        playlist_id: the id of the playlist
    """
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    ids = playlist["song_ids"]
    if song_id not in ids:
        return _error(f'Song {song_id} is not in "{playlist["name"]}".')
    _save_songs(ctx, store, playlist, [i for i in ids if i != song_id])
    return {"ok": True, "playlist": playlist["name"], "song_count": len(ids) - 1}


@tool
def remove_artist_from_playlist(ctx: AgentContext, artist: str, playlist_id: str) -> Dict:
    """Removes every song by an artist from one of the listener's own playlists. The
    artist name must match exactly (ignoring case).
    Args:
        artist: the artist's name
        playlist_id: the id of the playlist
    """
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    ids = playlist["song_ids"]
    keep = [i for i in ids if not (ctx.song(i) and _same(ctx.song(i)["artist"], artist))]
    if len(keep) == len(ids):
        return _error(f'No songs by "{artist}" in "{playlist["name"]}".')
    _save_songs(ctx, store, playlist, keep)
    return {"ok": True, "playlist": playlist["name"], "removed": len(ids) - len(keep),
            "song_count": len(keep)}


@tool
def remove_song_from_mix(ctx: AgentContext, song_id: int, playlist_id: str) -> Dict:
    """Removes a song from an auto-generated mix. The other songs close up, and a new
    song that fits the mix is added at the end. Only works on mixes (kind "auto"); use
    remove_song_from_playlist for the listener's own playlists.
    Args:
        song_id: the id of the song
        playlist_id: the id of the mix
    """
    store = ctx.load_store()
    mix, problem = _playlist(ctx, store, playlist_id)
    if problem:
        return problem
    if mix.get("kind") != "auto":
        return _error(f'"{mix["name"]}" is not a mix. Use remove_song_from_playlist for it.')
    if not plib.remove_from_mix(store, playlist_id, ctx.songs, song_id=song_id):
        return _error(f'Song {song_id} is not in "{mix["name"]}".')
    ctx.save_store(store)
    return {"ok": True, "mix": mix["name"], "song_count": len(mix["song_ids"])}


@tool
def remove_artist_from_mix(ctx: AgentContext, artist: str, playlist_id: str) -> Dict:
    """Removes every song by an artist from an auto-generated mix. The other songs close
    up, and new songs that fit the mix (by other artists) are added at the end. The artist
    name must match exactly (ignoring case). Only works on mixes (kind "auto").
    Args:
        artist: the artist's name
        playlist_id: the id of the mix
    """
    store = ctx.load_store()
    mix, problem = _playlist(ctx, store, playlist_id)
    if problem:
        return problem
    if mix.get("kind") != "auto":
        return _error(f'"{mix["name"]}" is not a mix. Use remove_artist_from_playlist for it.')
    names = {ctx.song(i)["artist"] for i in mix["song_ids"] if ctx.song(i) and _same(ctx.song(i)["artist"], artist)}
    if not names:
        return _error(f'No songs by "{artist}" in "{mix["name"]}".')
    removed = plib.remove_from_mix(store, playlist_id, ctx.songs, artist=next(iter(names)))
    ctx.save_store(store)
    return {"ok": True, "mix": mix["name"], "removed": removed, "song_count": len(mix["song_ids"])}


@tool
def move_song_in_playlist(ctx: AgentContext, song_id: int, playlist_id: str, position: int) -> Dict:
    """Moves a song to a new 1-based position in one of the listener's own playlists
    (1 = first). The songs around it shift to fill the gap; a position past the end
    puts it last.
    Args:
        song_id: the id of the song
        playlist_id: the id of the playlist
        position: the new position, counting from 1
    """
    store = ctx.load_store()
    playlist, problem = _user_playlist(ctx, store, playlist_id)
    if problem:
        return problem
    ids = list(playlist["song_ids"])
    if song_id not in ids:
        return _error(f'Song {song_id} is not in "{playlist["name"]}".')
    ids.remove(song_id)
    new_index = max(0, min(position - 1, len(ids)))
    ids.insert(new_index, song_id)
    _save_songs(ctx, store, playlist, ids)
    return {"ok": True, "playlist": playlist["name"], "position": new_index + 1}


# --- the queue -----------------------------------------------------------------------

@tool
def read_queue(ctx: AgentContext) -> Dict:
    """Returns what is playing now and the songs up next in order, each with its 1-based
    position in up next (1 plays next). Also says the repeat mode and whether shuffle is on.
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    current = ctx.song(queue.current_id)
    up_next = []
    for position, song_id in enumerate(queue.up_next[:MAX_QUEUE_SHOWN], start=1):
        song = ctx.song(song_id)
        if song:
            up_next.append(dict(_brief(song), position=position))
    result = {
        "ok": True,
        "now_playing": _brief(current) if current else None,
        "up_next": up_next,
        "up_next_count": len(queue.up_next),
        "repeat": queue.repeat,
        "shuffle": queue.shuffle,
    }
    if len(queue.up_next) > MAX_QUEUE_SHOWN:
        result["note"] = f"Only the first {MAX_QUEUE_SHOWN} up next songs are shown."
    return result


def _queueable(ctx: AgentContext, song_id: int):
    """(song, None), or (None, error dict) for an unknown song or one with no audio file."""
    song = ctx.song(song_id)
    if song is None:
        return None, _error(f"No song with id {song_id}. Use find_song for real ids.")
    if not ctx.playable(song):
        return None, _error(f'The audio file for "{song["title"]}" is missing, so it cannot be played.')
    return song, None


@tool
def play_song(ctx: AgentContext, song_id: int) -> Dict:
    """Starts playing a song right now. This replaces the current queue with a new one
    that starts on this song (the app then keeps adding similar songs). If the listener
    wants to keep what is playing, use play_song_next or add_song_to_queue instead.
    Args:
        song_id: the id of the song (from find_song)
    """
    if ctx.start_playback is None:
        return _error("Playback cannot be started from here.")
    song, problem = _queueable(ctx, song_id)
    if problem:
        return problem
    ctx.start_playback(song_id)
    queue = ctx.get_queue()
    if queue is None or queue.current_id != song_id:
        return _error(f'"{song["title"]}" could not be started.')
    return {"ok": True, "now_playing": _brief(song)}


@tool
def play_album(ctx: AgentContext, album: str) -> Dict:
    """Plays a whole album from its first track, replacing the current queue with it (the
    app keeps adding similar songs after the last track). The album name must match
    exactly (ignoring case): get it from find_album.
    Args:
        album: the album's exact name
    """
    if ctx.start_album is None:
        return _error("Playback cannot be started from here.")
    albums = _albums(ctx)
    name = next((a for a in albums if _same(a, album)), None)
    if name is None:
        return _error(f'No album called "{album}". Use find_album to look albums up.')
    members = albums[name]
    if not any(ctx.playable(s) for s in members):
        return _error(f'None of the songs on "{name}" have an audio file, so it cannot be played.')
    ctx.start_album(name)
    queue = ctx.get_queue()
    current = ctx.song(queue.current_id) if queue else None
    if current is None or _album_key(current) != name:
        return _error(f'"{name}" could not be started.')
    return {"ok": True, "album": name, "song_count": len(members), "now_playing": _brief(current)}


@tool
def play_playlist(ctx: AgentContext, playlist_id: str) -> Dict:
    """Plays a playlist or mix from its first playable song, replacing the current queue
    with it (in a shuffled queue it starts anywhere, and the app keeps adding similar
    songs after the last one).
    Args:
        playlist_id: the playlist's id (from list_playlists)
    """
    if ctx.start_playlist is None:
        return _error("Playback cannot be started from here.")
    playlist, problem = _playlist(ctx, ctx.load_store(), playlist_id)
    if problem:
        return problem
    members = [ctx.song(i) for i in playlist["song_ids"]]
    if not any(s and ctx.playable(s) for s in members):
        return _error(f'"{playlist["name"]}" has no playable songs.')
    ctx.start_playlist(playlist_id)
    queue = ctx.get_queue()
    current = ctx.song(queue.current_id) if queue else None
    if current is None or current["id"] not in playlist["song_ids"]:
        return _error(f'"{playlist["name"]}" could not be started.')
    return {"ok": True, "playlist": playlist["name"], "now_playing": _brief(current)}


@tool
def add_song_to_queue(ctx: AgentContext, song_id: int) -> Dict:
    """Adds a song to the very end of the queue (a song already queued is moved there).
    Needs something to be playing already (use play_song to start one).
    Args:
        song_id: the id of the song
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    song, problem = _queueable(ctx, song_id)
    if problem:
        return problem
    if not queue.add(song_id):
        return _error(f'"{song["title"]}" is the song playing right now.')
    return {"ok": True, "added": _brief(song), "position": len(queue.up_next)}


@tool
def play_song_next(ctx: AgentContext, song_id: int) -> Dict:
    """Puts a song right after the one playing now, so it plays when this song ends (a song
    already queued is moved there). Needs something to be playing already.
    Args:
        song_id: the id of the song
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    song, problem = _queueable(ctx, song_id)
    if problem:
        return problem
    if not queue.play_next(song_id):
        return _error(f'"{song["title"]}" is the song playing right now.')
    return {"ok": True, "will_play_next": _brief(song)}


@tool
def remove_song_from_queue(ctx: AgentContext, song_id: int) -> Dict:
    """Removes a song from up next (every copy of it). The song playing now cannot be removed.
    Args:
        song_id: the id of the song
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    indexes = [i for i in range(len(queue.songs) - 1, queue.pos, -1) if queue.songs[i] == song_id]
    if not indexes:
        return _error(f"Song {song_id} is not in up next.")
    for index in indexes:  # last to first, so the earlier indexes stay valid
        queue.remove(index)
    queue.top_up(ctx.songs, ctx.playable)
    return {"ok": True, "removed": len(indexes)}


@tool
def remove_artist_from_queue(ctx: AgentContext, artist: str) -> Dict:
    """Removes every song by an artist from up next. The artist name must match exactly
    (ignoring case). The song playing now stays.
    Args:
        artist: the artist's name
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    indexes = [
        i for i in range(len(queue.songs) - 1, queue.pos, -1)
        if ctx.song(queue.songs[i]) and _same(ctx.song(queue.songs[i])["artist"], artist)
    ]
    if not indexes:
        return _error(f'No songs by "{artist}" in up next.')
    for index in indexes:
        queue.remove(index)
    queue.top_up(ctx.songs, ctx.playable)
    return {"ok": True, "removed": len(indexes)}


# What the listener says for each repeat mode, mapped onto the queue's modes
_REPEAT_NAMES = {
    "off": "off", "none": "off", "no repeat": "off",
    "all": "all", "repeat": "all", "playlist": "all", "album": "all", "mix": "all",
    "one": "one", "song": "one", "repeat song": "one", "track": "one",
    "autoplay": "autoplay", "auto": "autoplay",
}


@tool
def set_play_mode(ctx: AgentContext, mode: str) -> Dict:
    """Changes what happens when the songs in up next run out (the repeat button). It
    applies to what is playing now and to anything played afterwards.
    Args:
        mode: "off" (just stop after the last song), "all" (repeat the album, playlist or
              mix), "one" (repeat the current song) or "autoplay" (keep going with
              recommended songs, the default)
    """
    key = _REPEAT_NAMES.get(str(mode).strip().lower())
    if key is None:
        return _error(f'Unknown mode "{mode}". Use one of: {", ".join(REPEAT_MODES)}.')
    if ctx.set_repeat is not None:
        ctx.set_repeat(key)
    else:
        queue, problem = _active_queue(ctx)
        if problem:
            return problem
        queue.set_repeat(key, ctx.songs, ctx.playable)
    return {"ok": True, "repeat": key}


@tool
def set_shuffle(ctx: AgentContext, on: bool) -> Dict:
    """Turns shuffle on or off. On, the rest of the album, playlist or mix that is playing
    continues in a random order (the current song keeps playing); off, in its own order
    again. It applies to anything played afterwards too.
    Args:
        on: true for shuffle on, false for off
    """
    if ctx.set_shuffle is not None:
        ctx.set_shuffle(bool(on))
    else:
        queue, problem = _active_queue(ctx)
        if problem:
            return problem
        queue.set_shuffle(bool(on), ctx.songs, ctx.playable)
    return {"ok": True, "shuffle": bool(on)}


@tool
def move_song_in_queue(ctx: AgentContext, song_id: int, position: int) -> Dict:
    """Moves a song that is already in up next to a new 1-based position (1 plays next).
    The songs around it shift to fill the gap; a position past the end puts it last.
    Args:
        song_id: the id of the song
        position: the new position in up next, counting from 1
    """
    queue, problem = _active_queue(ctx)
    if problem:
        return problem
    if not queue.move_up_next(song_id, position):
        return _error(f"Song {song_id} is not in up next. Use add_song_to_queue to add it first.")
    new_position = queue.up_next.index(song_id) + 1
    return {"ok": True, "position": new_position}
