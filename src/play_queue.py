"""The play queue: a list of song ids with a position (no Streamlit here).

Songs before `pos` are history (so Previous works); songs after it are up next.

What happens when the up next list runs low is decided by the repeat mode:

    "off"       nothing is added: playback stops when the songs (the album, playlist or
                mix) have played, and the song if it was played alone
    "all"       the album / playlist / mix is queued again, from the start, whenever the
                queue gets low (a song played alone repeats itself)
    "one"       the song that is playing fills the queue with copies of itself, forever
    "autoplay"  the recommender appends songs, whenever the queue gets low: the best fit
                for the queue's profile that also flows from the queue's last song
                (recommend_songs with current_song), never repeating a song that is
                already in the queue. This is the default

Shuffle plays the album / playlist / mix in a random order instead of its own.

Every entry remembers where it came from (see `origin`), so that changing the mode can
take out exactly what the old mode added and leave what the listener chose.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    from playlists import profile_from_songs
    from recommender import recommend_songs
except ImportError:  # imported as part of the "src" package (tests)
    from src.playlists import profile_from_songs
    from src.recommender import recommend_songs

QUEUE_AHEAD = 20  # songs always lined up after the current one
QUEUE_HISTORY = 50  # played songs kept so Previous can go back
CANDIDATES = 25  # best recommendations looked through for a playable file

# The repeat button cycles through these, in this order: none, repeat the album / playlist /
# mix, repeat the song, then autoplay (the recommender keeps going).
REPEAT_MODES = ("off", "all", "one", "autoplay")

# Where a queue entry came from
BASE = "base"      # the album / playlist / mix being played (or the single song)
MANUAL = "manual"  # put there by the listener: Play Next, Add to Queue
AUTO = "auto"      # recommended by autoplay
LOOP = "loop"      # a repeated copy: a later pass of the collection, or a copy of the song


def _always(_song: Dict) -> bool:
    return True


def next_repeat_mode(mode: str) -> str:
    """The mode the repeat button moves to when pressed."""
    if mode not in REPEAT_MODES:
        return REPEAT_MODES[0]
    return REPEAT_MODES[(REPEAT_MODES.index(mode) + 1) % len(REPEAT_MODES)]


@dataclass
class PlayQueue:
    """What is playing and what comes next.

    songs:   song ids, in play order (history, the current song, then up next). With
             repeat on, a song can be in the queue more than once.
    pos:     index in `songs` of the current song
    profile: the recommender preferences used to extend the queue
    source:  id of the playlist (or album/artist collection) it was started from, if any
    removed: ids the listener took out of the queue; they are not recommended again
    repeat:  one of REPEAT_MODES
    shuffle: whether the collection plays in a random order
    base:    the collection (album, playlist, mix) in its own order; just the song if it
             was started alone. It is what "repeat all" plays again and what turning
             shuffle off puts back in order.
    origin:  for each entry of `songs`, where it came from (BASE, MANUAL, AUTO or LOOP)

    Methods that may add songs take the library (`songs`, every song as a dict) and a
    `playable` check, so a song whose file is missing is skipped."""

    songs: List[int]
    pos: int = 0
    profile: Dict = field(default_factory=dict)
    source: Optional[str] = None
    removed: List[int] = field(default_factory=list)
    repeat: str = "autoplay"
    shuffle: bool = False
    base: List[int] = field(default_factory=list)
    origin: List[str] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.base:
            self.base = list(self.songs)
        if len(self.origin) != len(self.songs):
            self.origin = [BASE] * len(self.songs)

    @classmethod
    def start(
        cls,
        song_id: int,
        library: List[Dict],
        playlist: Optional[Dict] = None,
        playable: Callable[[Dict], bool] = _always,
        repeat: str = "autoplay",
        shuffle: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Optional["PlayQueue"]:
        """A queue that starts on song_id.

        Without a playlist it is just that song. With one, every song of the playlist is
        queued in order (starting from song_id), or, with shuffle, song_id first and all
        the others in a random order. What comes after is up to the repeat mode (by
        default the recommender fills it, from the song's or the playlist's profile).
        Returns None if song_id is unknown."""
        by_id = {s["id"]: s for s in library}
        if song_id not in by_id:
            return None
        ids = [i for i in (playlist or {}).get("song_ids", []) if i in by_id]
        if playlist and song_id in ids:
            profile = playlist.get("profile") or profile_from_songs([by_id[i] for i in ids])
            queue = cls(ids, ids.index(song_id), profile, playlist["id"], base=list(ids))
        else:
            queue = cls([song_id], 0, profile_from_songs([by_id[song_id]]), None, base=[song_id])
        queue.repeat = repeat if repeat in REPEAT_MODES else "autoplay"
        if rng is not None:
            queue.rng = rng
        if shuffle:
            # It starts on the chosen song; set_shuffle lines up all the others after it.
            del queue.songs[:queue.pos]
            del queue.origin[:queue.pos]
            queue.pos = 0
            queue.set_shuffle(True, library, playable)
        else:
            queue.top_up(library, playable)
        return queue

    @property
    def current_id(self) -> int:
        return self.songs[self.pos]

    @property
    def history(self) -> List[int]:
        return self.songs[:self.pos]

    @property
    def up_next(self) -> List[int]:
        return self.songs[self.pos + 1:]

    # --- editing the lists (they always change together) ---

    def _insert(self, index: int, song_id: int, origin: str) -> None:
        self.songs.insert(index, song_id)
        self.origin.insert(index, origin)

    def _append(self, song_id: int, origin: str) -> None:
        self.songs.append(song_id)
        self.origin.append(origin)

    def _delete(self, index: int) -> None:
        del self.songs[index]
        del self.origin[index]
        if index < self.pos:
            self.pos -= 1

    def _drop(self, *origins: str) -> None:
        """Takes the entries of those origins out of up next."""
        for index in reversed(range(self.pos + 1, len(self.songs))):
            if self.origin[index] in origins:
                self._delete(index)

    # --- keeping the queue stocked ---

    def top_up(self, library: List[Dict], playable: Callable[[Dict], bool] = _always) -> None:
        """Stocks up next according to the repeat mode (see the top of this file), then
        trims old history."""
        by_id = {s["id"]: s for s in library}
        if self.repeat == "autoplay":
            while len(self.up_next) < QUEUE_AHEAD:
                last = by_id.get(self.songs[-1])
                ranked = recommend_songs(
                    self.profile, library, k=CANDIDATES, current_song=last,
                    recently_played=self.songs + self.removed,
                )
                choice = next((s for s, _score, _why in ranked if playable(s)), None)
                if choice is None:
                    break  # nothing playable left to add
                self._append(choice["id"], AUTO)
        elif self.repeat == "all":
            while len(self.up_next) < QUEUE_AHEAD:
                again = [i for i in self.base if i in by_id and playable(by_id[i])]
                if not again:
                    break
                if self.shuffle:
                    self.rng.shuffle(again)
                for song_id in again:
                    self._append(song_id, LOOP)
        elif self.repeat == "one":
            self._drop(LOOP)  # the copies of whatever played before
            for _ in range(QUEUE_AHEAD):
                self._insert(self.pos + 1, self.current_id, LOOP)
        # "off": nothing is added
        extra = self.pos - QUEUE_HISTORY
        if extra > 0:
            del self.songs[:extra]
            del self.origin[:extra]
            self.pos -= extra

    def set_repeat(
        self, mode: str, library: List[Dict], playable: Callable[[Dict], bool] = _always
    ) -> None:
        """Switches the repeat mode. What the old mode had added to up next is taken out
        (repeated copies always; recommendations unless the new mode is autoplay), what the
        listener put there stays, and the new mode stocks the queue."""
        if mode not in REPEAT_MODES or mode == self.repeat:
            return
        self.repeat = mode
        self._drop(LOOP)
        if mode != "autoplay":
            self._drop(AUTO)
        self.top_up(library, playable)

    def set_shuffle(
        self, on: bool, library: List[Dict], playable: Callable[[Dict], bool] = _always
    ) -> None:
        """Turns shuffle on or off. The song playing stays; the rest of the collection in up
        next is lined up again: with shuffle on, every other song of it in a random order
        (also ones already played); with it off, in its own order from after the song
        playing. Songs the listener added, and recommendations, stay where they are."""
        if on == self.shuffle:
            return
        self.shuffle = on
        by_id = {s["id"]: s for s in library}
        first = next(
            (i for i in range(self.pos + 1, len(self.songs)) if self.origin[i] in (BASE, LOOP)),
            None,
        )
        kept_before = None if first is None else first - self.pos - 1
        self._drop(BASE, LOOP)
        at = len(self.songs) if kept_before is None else self.pos + 1 + kept_before

        current = self.current_id
        if on:
            items = [i for i in self.base if i != current]
            self.rng.shuffle(items)
        elif current in self.base:
            items = self.base[self.base.index(current) + 1:]
        else:
            items = list(self.base)
        for offset, song_id in enumerate(i for i in items if i in by_id and playable(by_id[i])):
            self._insert(at + offset, song_id, BASE)
        self.top_up(library, playable)

    # --- moving around and editing ---

    def jump(
        self, index: int, library: List[Dict], playable: Callable[[Dict], bool] = _always
    ) -> Optional[int]:
        """Makes the song at `index` the current one (clicking a row in the queue panel)
        and tops the queue up. Returns its id, or None if out of range."""
        if not 0 <= index < len(self.songs):
            return None
        self.pos = index
        song_id = self.songs[index]
        self.top_up(library, playable)
        return song_id

    def remove(
        self, index: int, library: Optional[List[Dict]] = None,
        playable: Callable[[Dict], bool] = _always,
    ) -> bool:
        """Takes the song at `index` out of the queue. The song that is playing now cannot
        be removed. If `library` is given the queue is topped up again afterwards. Returns
        whether anything was removed."""
        if not 0 <= index < len(self.songs) or index == self.pos:
            return False
        self.removed.append(self.songs[index])
        self._delete(index)
        if library is not None:
            self.top_up(library, playable)
        return True

    def _detach(self, song_id: int) -> None:
        """Takes the song out of its place in the queue, if it is in it (the first copy
        after the current song, otherwise the first before it), keeping the playing song
        playing. It is no longer counted as removed: adding it was the listener's choice."""
        index = next(
            (i for i in range(self.pos + 1, len(self.songs)) if self.songs[i] == song_id), None
        )
        if index is None:
            index = next((i for i in range(self.pos) if self.songs[i] == song_id), None)
        if index is not None:
            self._delete(index)
        if song_id in self.removed:
            self.removed.remove(song_id)

    def play_next(self, song_id: int) -> bool:
        """Puts the song right after the current one, so it plays when this song ends. A
        song already in the queue is moved there. Returns False for the song that is
        playing now (it is already playing)."""
        if song_id == self.current_id:
            return False
        self._detach(song_id)
        self._insert(self.pos + 1, song_id, MANUAL)
        return True

    def add(self, song_id: int) -> bool:
        """Adds the song at the very end of the queue (a song already in it is moved
        there). Returns False for the song that is playing now. Songs added later to keep
        the queue topped up go after it."""
        if song_id == self.current_id:
            return False
        self._detach(song_id)
        self._append(song_id, MANUAL)
        return True

    def advance(
        self, delta: int, library: List[Dict], playable: Callable[[Dict], bool] = _always
    ) -> Optional[int]:
        """Moves to the next (+1) or previous (-1) playable song and returns its id.

        Next keeps the queue topped up. Previous at the start of the history stays put
        (returns the current song, so the caller restarts it). Returns None only when
        there is nothing to move to (with repeat off, after the last song)."""
        by_id = {s["id"]: s for s in library}
        step = 1 if delta > 0 else -1
        pos = self.pos + step
        while 0 <= pos < len(self.songs):
            song = by_id.get(self.songs[pos])
            if song is not None and playable(song):
                self.pos = pos
                self.top_up(library, playable)
                return song["id"]
            pos += step
        if step < 0:
            return self.current_id
        return None
