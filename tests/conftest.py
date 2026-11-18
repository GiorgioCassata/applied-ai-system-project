"""Shared fixtures for the agent tests: a small made-up library (so they do not need
data/songs.csv), an in-memory playlist store, and an AgentContext over them."""

import pytest

from src import playlists as plib
from src.agent_tools import AgentContext
from src.play_queue import PlayQueue

UNPLAYABLE_ID = 13  # a song whose audio file is "missing"


def make_song(song_id, title, artist, album="Album 0", year="2000"):
    return {
        "id": song_id, "title": title, "artist": artist, "album": album,
        "release_year": year, "track_number": "1", "disc_number": "1",
        "duration_seconds": "200", "genre": "pop", "mood": "happy",
        "energy": (song_id % 10) / 10, "tempo_bpm": 100.0, "valence": 0.5,
        "acousticness": 0.2, "file_path": f"/music/{song_id}.mp3",
    }


@pytest.fixture
def library():
    """Songs 1-30 ("Song N" by "Artist N%4" on "Album N%3"), plus song 100: the same song
    as song 1 on another album. Songs 7 and 8 are from 1995."""
    songs = [
        make_song(i, f"Song {i}", f"Artist {i % 4}", f"Album {i % 3}",
                  year="1995" if i in (7, 8) else "2000")
        for i in range(1, 31)
    ]
    songs.append(make_song(100, "Song 1", "Artist 1", "Compilation"))
    return songs


@pytest.fixture
def store():
    return {"playlists": {}, "recent": []}


@pytest.fixture
def saves():
    """One entry for every time the tools saved the store."""
    return []


@pytest.fixture
def queue_box():
    """Holds the session's play queue: tests put a PlayQueue in box["queue"]."""
    return {"queue": None}


@pytest.fixture
def played():
    """The song ids the tools started playing."""
    return []


@pytest.fixture
def played_albums():
    """The album names the tools started playing."""
    return []


@pytest.fixture
def played_playlists():
    """The playlist ids the tools started playing."""
    return []


@pytest.fixture
def ctx(library, store, saves, queue_box, played, played_albums, played_playlists):
    def start_playback(song_id):
        played.append(song_id)
        queue_box["queue"] = PlayQueue([song_id], 0, repeat="off")  # like app.play_song

    def start_album(name):
        played_albums.append(name)
        ids = [s["id"] for s in library if s["album"] == name and s["id"] != UNPLAYABLE_ID]
        queue_box["queue"] = PlayQueue(ids, 0, repeat="off")

    def start_playlist(playlist_id):
        played_playlists.append(playlist_id)
        ids = [i for i in store["playlists"][playlist_id]["song_ids"] if i != UNPLAYABLE_ID]
        queue_box["queue"] = PlayQueue(ids, 0, repeat="off")

    return AgentContext(
        songs=library,
        get_queue=lambda: queue_box["queue"],
        playable=lambda song: song["id"] != UNPLAYABLE_ID,
        load_store=lambda: store,
        save_store=lambda s: saves.append(1),
        start_playback=start_playback,
        start_album=start_album,
        start_playlist=start_playlist,
    )


@pytest.fixture
def user_playlist(store, library):
    """The listener's playlist "Faves": songs 1, 2, 3. Returns its id."""
    return plib.create_playlist(store, "Faves", [1, 2, 3], library)


@pytest.fixture
def auto_mix(store):
    """An auto-generated mix "Work Mix" (songs 1, 2, 3). Returns its id."""
    store["playlists"]["auto-work"] = {
        "id": "auto-work", "name": "Work Mix", "kind": "auto", "group": "activity",
        "song_ids": [1, 2, 3],
        "profile": {"energy": 0.5, "valence": 0.5, "tempo_bpm": 100, "likes_acoustic": True},
        "created": "2026-01-01T00:00:00", "slot": None,
    }
    return "auto-work"
