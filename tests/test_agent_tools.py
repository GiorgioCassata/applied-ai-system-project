import pytest

from src import agent_tools as at
from src.agent_tools import MAX_RESULTS, run_tool
from src.play_queue import PlayQueue

EXPECTED_TOOLS = {
    "find_song", "read_song", "list_playlists", "read_playlist", "find_index",
    "create_playlist", "save_mix_as_playlist", "regenerate_mix", "delete_playlist",
    "rename_playlist", "add_song_to_playlist", "remove_song_from_playlist",
    "search_songs", "set_play_mode", "set_shuffle", "remove_artist_from_playlist", "remove_duplicates_from_playlist", "remove_song_from_mix", "remove_artist_from_mix",
    "move_song_in_playlist", "read_queue",
    "find_songs_by_feel", "list_genres_and_moods", "find_album", "play_album", "play_playlist", "play_song", "add_song_to_queue", "play_song_next", "remove_song_from_queue",
    "remove_artist_from_queue", "move_song_in_queue",
}


def ids_of(result):
    return [s["id"] for s in result["songs"]]


def set_queue(queue_box, songs, pos=0):
    queue_box["queue"] = PlayQueue(list(songs), pos, repeat="off")
    return queue_box["queue"]


# --- the registry and the schemas ---------------------------------------------------

def test_every_expected_tool_is_registered():
    assert set(at.TOOLS) == EXPECTED_TOOLS
    assert len(at.tool_schemas()) == len(EXPECTED_TOOLS)


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_each_schema_is_well_formed(name):
    tool = at.TOOLS[name]
    assert tool.name == name and tool.description
    params = tool.parameters
    assert params["type"] == "object"
    assert set(params["required"]) <= set(params["properties"])
    assert "ctx" not in params["properties"]  # the context is never shown to the model
    for prop, schema in params["properties"].items():
        assert schema["type"] in {"string", "integer", "number", "boolean", "array"}, prop
        assert schema.get("description"), f"{name}.{prop} has no description in the docstring"
    shown = tool.schema()
    assert shown["type"] == "function" and shown["function"]["name"] == name


def test_schema_types_and_required_come_from_the_signature():
    create = at.TOOLS["create_playlist"].parameters
    assert create["properties"]["name"]["type"] == "string"
    assert create["properties"]["song_ids"] == {
        "type": "array", "items": {"type": "integer"},
        "description": create["properties"]["song_ids"]["description"],
    }
    assert create["required"] == ["name", "song_ids"]
    find = at.TOOLS["find_song"].parameters
    assert find["required"] == []  # every filter is optional
    assert find["properties"]["release_year"]["type"] == "integer"


def test_read_only_tools_are_the_ones_that_only_look():
    assert at.READ_ONLY_TOOLS <= set(at.TOOLS)
    for name in at.TOOLS:
        assert at.is_action(name) == (name not in at.READ_ONLY_TOOLS)
    for name in ("play_song", "play_album", "play_playlist", "add_song_to_queue", "create_playlist",
                 "delete_playlist", "move_song_in_queue", "regenerate_mix"):
        assert at.is_action(name), name
    assert not at.is_action("no_such_tool")


# --- checking what the model sends --------------------------------------------------

def test_unknown_tool_is_an_error_not_an_exception(ctx):
    result = run_tool(ctx, "make_coffee", {})
    assert result["ok"] is False and "make_coffee" in result["error"]


def test_unknown_and_missing_arguments_are_reported(ctx):
    assert "Unknown argument" in run_tool(ctx, "read_song", {"song_id": 1, "volume": 3})["error"]
    assert "Missing required" in run_tool(ctx, "read_song", {})["error"]


def test_wrong_type_is_reported(ctx):
    result = run_tool(ctx, "read_song", {"song_id": "abc"})
    assert result["ok"] is False and "song_id" in result["error"]


def test_numbers_sent_as_text_are_accepted(ctx):
    assert run_tool(ctx, "read_song", {"song_id": "5"})["ok"] is True
    assert run_tool(ctx, "read_song", {"song_id": 5.0})["ok"] is True


def test_lists_sent_as_text_are_accepted(ctx):
    for sent in ("[1, 2]", "1,2", "1 2"):
        result = run_tool(ctx, "create_playlist", {"name": f"P {sent}", "song_ids": sent})
        assert result["ok"] is True and result["song_count"] == 2, sent


def test_null_arguments_count_as_not_given(ctx):
    result = run_tool(ctx, "find_song", {"title": "song 17", "artist": None})
    assert result["ok"] is True and ids_of(result) == [17]


def test_a_crashing_tool_comes_back_as_an_error(library, saves, queue_box):
    def boom():
        raise ZeroDivisionError("nope")

    broken = at.AgentContext(songs=library, get_queue=lambda: None, load_store=boom,
                             save_store=lambda s: None)
    result = run_tool(broken, "list_playlists", {})
    assert result["ok"] is False and "ZeroDivisionError" in result["error"]


# --- looking things up ---------------------------------------------------------------

def test_find_song_by_title_and_artist(ctx):
    result = run_tool(ctx, "find_song", {"title": "song 17", "artist": "artist 1"})
    assert result["ok"] and ids_of(result) == [17]
    assert set(result["songs"][0]) == {"id", "title", "artist", "album", "release_year"}


def test_find_song_shows_the_same_song_on_several_albums(ctx):
    both = run_tool(ctx, "find_song", {"title": "Song 1", "artist": "Artist 1"})
    assert {1, 100} <= set(ids_of(both))
    only = run_tool(ctx, "find_song", {"title": "Song 1", "album": "compilation"})
    assert ids_of(only) == [100]


def test_find_song_by_year_alone(ctx):
    assert sorted(ids_of(run_tool(ctx, "find_song", {"release_year": 1995}))) == [7, 8]


def test_find_song_needs_a_filter(ctx):
    assert run_tool(ctx, "find_song", {})["ok"] is False


def test_find_song_falls_back_to_the_apps_search(ctx):
    result = run_tool(ctx, "find_song", {"title": "song 99 artist"})  # no exact match
    assert result["ok"] and result["count"] > 0 and "closest" in result["note"]


def test_find_song_with_nothing_found(ctx):
    result = run_tool(ctx, "find_song", {"title": "zzz qqq"})
    assert result == {"ok": True, "count": 0, "songs": []}


def test_find_song_caps_the_results(ctx):
    result = run_tool(ctx, "find_song", {"title": "song"})
    assert result["count"] == 31 and len(result["songs"]) == MAX_RESULTS
    assert "narrow" in result["note"]


def test_read_song(ctx):
    result = run_tool(ctx, "read_song", {"song_id": 5})
    assert result["ok"] and result["song"]["title"] == "Song 5"
    assert "file_path" not in result["song"]  # the model does not need to see file paths
    assert run_tool(ctx, "read_song", {"song_id": 999})["ok"] is False


def test_list_playlists_and_its_kind_filter(ctx, user_playlist, auto_mix):
    everything = run_tool(ctx, "list_playlists", {})
    assert everything["count"] == 2
    mine = run_tool(ctx, "list_playlists", {"kind": "user"})
    assert [p["name"] for p in mine["playlists"]] == ["Faves"]
    assert mine["playlists"][0]["song_count"] == 3
    assert [p["id"] for p in run_tool(ctx, "list_playlists", {"kind": "auto"})["playlists"]] == [auto_mix]
    assert run_tool(ctx, "list_playlists", {"kind": "other"})["ok"] is False


def test_read_playlist_numbers_the_songs_from_one(ctx, user_playlist):
    result = run_tool(ctx, "read_playlist", {"playlist_id": user_playlist})
    assert result["name"] == "Faves" and result["kind"] == "user"
    assert [(s["id"], s["position"]) for s in result["songs"]] == [(1, 1), (2, 2), (3, 3)]


@pytest.mark.parametrize("tool,extra", [("read_playlist", {}), ("find_index", {"song_id": 1})])
def test_an_unknown_playlist_points_to_list_playlists(ctx, tool, extra):
    result = run_tool(ctx, tool, {"playlist_id": "nope", **extra})
    assert result["ok"] is False and "list_playlists" in result["error"]


def test_find_index(ctx, user_playlist):
    assert run_tool(ctx, "find_index", {"song_id": 3, "playlist_id": user_playlist})["position"] == 3
    missing = run_tool(ctx, "find_index", {"song_id": 20, "playlist_id": user_playlist})
    assert missing["ok"] and missing["position"] is None and missing["playlist_size"] == 3


# --- creating, saving and deleting playlists ----------------------------------------

def test_create_playlist(ctx, store, saves):
    result = run_tool(ctx, "create_playlist", {"name": " Road ", "song_ids": [3, 1, 3, 2]})
    assert result["ok"] and result["name"] == "Road" and result["song_count"] == 3
    made = store["playlists"][result["playlist_id"]]
    assert made["kind"] == "user" and made["song_ids"] == [3, 1, 2]  # in order, no repeats
    assert saves == [1]


def test_create_playlist_makes_the_name_unique(ctx):
    run_tool(ctx, "create_playlist", {"name": "Mix", "song_ids": [1]})
    second = run_tool(ctx, "create_playlist", {"name": "mix", "song_ids": [2]})
    assert second["name"] == "mix 2"


def test_create_playlist_with_unknown_songs_creates_nothing(ctx, store, saves):
    result = run_tool(ctx, "create_playlist", {"name": "Bad", "song_ids": [1, 999]})
    assert result["ok"] is False and "999" in result["error"]
    assert store["playlists"] == {} and saves == []


def test_create_playlist_needs_a_name(ctx):
    assert run_tool(ctx, "create_playlist", {"name": "  ", "song_ids": [1]})["ok"] is False


def test_save_mix_as_playlist(ctx, store, auto_mix):
    first = run_tool(ctx, "save_mix_as_playlist", {"playlist_id": auto_mix})
    assert first["ok"] and first["name"] == "My Work Mix"
    copy = store["playlists"][first["playlist_id"]]
    assert copy["kind"] == "user" and copy["song_ids"] == [1, 2, 3]
    assert store["playlists"][auto_mix]["kind"] == "auto"  # the mix carries on
    assert run_tool(ctx, "save_mix_as_playlist", {"playlist_id": auto_mix})["name"] == "My Work Mix 2"


def test_save_mix_as_playlist_refuses_your_own_playlist(ctx, user_playlist):
    assert run_tool(ctx, "save_mix_as_playlist", {"playlist_id": user_playlist})["ok"] is False


def test_regenerate_mix(ctx, store, auto_mix, saves):
    result = run_tool(ctx, "regenerate_mix", {"playlist_id": auto_mix})
    assert result["ok"] and result["song_count"] == len(store["playlists"][auto_mix]["song_ids"]) > 0
    assert saves == [1]


def test_regenerate_mix_refuses_your_own_playlist(ctx, user_playlist, saves):
    result = run_tool(ctx, "regenerate_mix", {"playlist_id": user_playlist})
    assert result["ok"] is False and saves == []


def test_delete_playlist(ctx, store, user_playlist):
    assert run_tool(ctx, "delete_playlist", {"playlist_id": user_playlist}) == {"ok": True, "deleted": "Faves"}
    assert user_playlist not in store["playlists"]


def test_rename_playlist(ctx, store, user_playlist):
    assert run_tool(ctx, "rename_playlist", {"playlist_id": user_playlist, "name": " Best "})["name"] == "Best"
    assert store["playlists"][user_playlist]["name"] == "Best"
    # keeping its own name (maybe in other letters) is fine
    assert run_tool(ctx, "rename_playlist", {"playlist_id": user_playlist, "name": "BEST"})["ok"]


def test_rename_playlist_refuses_a_taken_or_empty_name(ctx, store, user_playlist):
    run_tool(ctx, "create_playlist", {"name": "Other", "song_ids": [4]})
    taken = run_tool(ctx, "rename_playlist", {"playlist_id": user_playlist, "name": "other"})
    assert taken["ok"] is False and store["playlists"][user_playlist]["name"] == "Faves"
    assert run_tool(ctx, "rename_playlist", {"playlist_id": user_playlist, "name": " "})["ok"] is False


# --- mixes cannot be edited ----------------------------------------------------------

@pytest.mark.parametrize("tool,args", [
    ("add_song_to_playlist", {"song_id": 5}),
    ("remove_song_from_playlist", {"song_id": 1}),
    ("remove_artist_from_playlist", {"artist": "Artist 1"}),
    ("move_song_in_playlist", {"song_id": 1, "position": 3}),
    ("rename_playlist", {"name": "Renamed"}),
    ("delete_playlist", {}),
])
def test_mixes_cannot_be_edited_and_the_error_says_how_to_copy_one(ctx, store, auto_mix, saves, tool, args):
    result = run_tool(ctx, tool, {"playlist_id": auto_mix, **args})
    assert result["ok"] is False and "save_mix_as_playlist" in result["error"]
    assert store["playlists"][auto_mix]["song_ids"] == [1, 2, 3]
    assert store["playlists"][auto_mix]["name"] == "Work Mix"
    assert saves == []


# --- editing your own playlists ------------------------------------------------------

def test_add_song_to_playlist(ctx, store, user_playlist, saves):
    result = run_tool(ctx, "add_song_to_playlist", {"song_id": 5, "playlist_id": user_playlist})
    assert result == {"ok": True, "playlist": "Faves", "song_count": 4}
    assert store["playlists"][user_playlist]["song_ids"] == [1, 2, 3, 5]
    assert saves == [1]


def test_add_song_to_playlist_refuses_a_song_already_in_it(ctx, user_playlist):
    again = run_tool(ctx, "add_song_to_playlist", {"song_id": 1, "playlist_id": user_playlist})
    assert again["ok"] is False and "already" in again["error"]


def test_add_song_to_playlist_refuses_the_same_song_from_another_album(ctx, store, user_playlist):
    twin = run_tool(ctx, "add_song_to_playlist", {"song_id": 100, "playlist_id": user_playlist})
    assert twin["ok"] is False and "another album" in twin["error"]
    assert store["playlists"][user_playlist]["song_ids"] == [1, 2, 3]


def test_add_song_to_playlist_with_an_unknown_song(ctx, user_playlist):
    assert run_tool(ctx, "add_song_to_playlist", {"song_id": 999, "playlist_id": user_playlist})["ok"] is False


def test_remove_song_from_playlist(ctx, store, user_playlist):
    result = run_tool(ctx, "remove_song_from_playlist", {"song_id": 2, "playlist_id": user_playlist})
    assert result["ok"] and result["song_count"] == 2
    assert store["playlists"][user_playlist]["song_ids"] == [1, 3]
    assert run_tool(ctx, "remove_song_from_playlist", {"song_id": 2, "playlist_id": user_playlist})["ok"] is False


def test_remove_artist_from_playlist(ctx, store, user_playlist):
    run_tool(ctx, "add_song_to_playlist", {"song_id": 5, "playlist_id": user_playlist})  # Artist 1, like song 1
    result = run_tool(ctx, "remove_artist_from_playlist", {"artist": "artist 1", "playlist_id": user_playlist})
    assert result["ok"] and result["removed"] == 2
    assert store["playlists"][user_playlist]["song_ids"] == [2, 3]
    nothing = run_tool(ctx, "remove_artist_from_playlist", {"artist": "Artist 1", "playlist_id": user_playlist})
    assert nothing["ok"] is False


def test_remove_duplicates_from_playlist(ctx, store, user_playlist):
    nothing = run_tool(ctx, "remove_duplicates_from_playlist", {"playlist_id": user_playlist})
    assert nothing["ok"] is False
    store["playlists"][user_playlist]["song_ids"] = [1, 2, 1, 3, 2]
    result = run_tool(ctx, "remove_duplicates_from_playlist", {"playlist_id": user_playlist})
    assert result["ok"] and result["removed"] == 2 and result["song_count"] == 3
    assert store["playlists"][user_playlist]["song_ids"] == [1, 2, 3]


def test_remove_duplicates_from_mix(ctx, store, auto_mix):
    store["playlists"][auto_mix]["song_ids"].append(1)  # a second copy of song 1
    result = run_tool(ctx, "remove_duplicates_from_playlist", {"playlist_id": auto_mix})
    ids = store["playlists"][auto_mix]["song_ids"]
    assert result["ok"] and result["removed"] == 1
    assert ids[:3] == [1, 2, 3] and len(ids) == len(set(ids))  # new songs only go at the end
    assert result["song_count"] == len(ids)
    assert run_tool(ctx, "remove_duplicates_from_playlist", {"playlist_id": auto_mix})["ok"] is False


def test_a_playlist_can_be_given_by_name_instead_of_id(ctx, store, user_playlist):
    name = store["playlists"][user_playlist]["name"]
    result = run_tool(ctx, "remove_song_from_playlist", {"song_id": 2, "playlist_id": name.upper()})
    assert result["ok"] and store["playlists"][user_playlist]["song_ids"] == [1, 3]
    assert run_tool(ctx, "remove_song_from_playlist", {"song_id": 2, "playlist_id": "No Such"})["ok"] is False


def test_move_song_in_playlist(ctx, store, user_playlist):
    run_tool(ctx, "add_song_to_playlist", {"song_id": 4, "playlist_id": user_playlist})

    def ids():  # read fresh each time: an edit replaces the playlist's list
        return store["playlists"][user_playlist]["song_ids"]

    moved = run_tool(ctx, "move_song_in_playlist", {"song_id": 4, "playlist_id": user_playlist, "position": 1})
    assert moved["position"] == 1 and ids() == [4, 1, 2, 3]
    run_tool(ctx, "move_song_in_playlist", {"song_id": 4, "playlist_id": user_playlist, "position": 99})
    assert ids() == [1, 2, 3, 4]  # past the end puts it last
    run_tool(ctx, "move_song_in_playlist", {"song_id": 3, "playlist_id": user_playlist, "position": 0})
    assert ids() == [3, 1, 2, 4]  # before the start puts it first
    assert run_tool(ctx, "move_song_in_playlist",
                    {"song_id": 20, "playlist_id": user_playlist, "position": 1})["ok"] is False


def test_editing_a_playlist_works_out_its_profile_again(ctx, store, user_playlist):
    before = dict(store["playlists"][user_playlist]["profile"])
    run_tool(ctx, "add_song_to_playlist", {"song_id": 9, "playlist_id": user_playlist})  # energy 0.9
    assert store["playlists"][user_playlist]["profile"]["energy"] > before["energy"]


# --- finding songs by feel -----------------------------------------------------------

@pytest.fixture
def feel(library):
    """A context over a few songs with different genres, moods, valence and lengths:
    id: genre / mood / valence / seconds. Song 6 has no audio file."""
    rows = [
        (1, "rock", "sad", 0.2, 150), (2, "rock", "happy", 0.9, 240), (3, "pop", "sad", 0.1, 300),
        (4, "pop", "happy", 0.8, 120), (5, "Rap & Hip-Hop", "energetic", 0.7, 200),
        (6, "jazz", "chill", 0.5, 400), (7, "Alternative Rock", "intense", 0.4, 180),
    ]
    songs = [
        dict(library[0], id=i, title=f"S{i}", artist=f"A{i}", genre=g, mood=m, valence=v,
             duration_seconds=str(sec))
        for i, g, m, v, sec in rows
    ]
    return at.AgentContext(songs=songs, get_queue=lambda: None, playable=lambda s: s["id"] != 6)


def feel_ids(ctx, **filters):
    result = run_tool(ctx, "find_songs_by_feel", filters)
    assert result["ok"], result
    return sorted(ids_of(result))


def test_feel_by_genre_matches_part_of_the_name_ignoring_case_and_punctuation(feel):
    assert feel_ids(feel, genre="rock") == [1, 2, 7]  # also "Alternative Rock"
    assert feel_ids(feel, genre="ROCK") == [1, 2, 7]
    assert feel_ids(feel, genre="hip hop") == [5]  # "Rap & Hip-Hop"


def test_feel_by_mood_is_exact(feel):
    assert feel_ids(feel, mood="SAD") == [1, 3]  # (a part of a mood, like "sa", is an error: see below)


def test_feel_by_valence(feel):
    assert feel_ids(feel, max_valence=0.3) == [1, 3]  # sad
    assert feel_ids(feel, min_valence=0.75) == [2, 4]  # upbeat
    assert feel_ids(feel, min_valence=0.35, max_valence=0.5) == [7]


def test_feel_by_length_in_minutes(feel):
    assert feel_ids(feel, max_minutes=2.5) == [1, 4]  # up to 150 seconds
    assert feel_ids(feel, min_minutes=5) == [3]  # song 6 is longer but cannot be played
    assert feel_ids(feel, min_minutes=3, max_minutes=4) == [2, 5, 7]  # 180 to 240 seconds


def test_feel_filters_all_have_to_match(feel):
    assert feel_ids(feel, genre="pop", mood="sad", max_valence=0.2) == [3]
    assert feel_ids(feel, genre="rock", min_valence=0.8, max_minutes=5) == [2]
    assert feel_ids(feel, genre="pop", mood="intense") == []


def test_feel_results_carry_what_the_model_needs_to_judge(feel):
    song = run_tool(feel, "find_songs_by_feel", {"mood": "sad", "genre": "pop"})["songs"][0]
    assert song == {"id": 3, "title": "S3", "artist": "A3", "album": "Album 1",
                    "genre": "pop", "mood": "sad", "valence": 0.1, "duration_seconds": 300}


@pytest.mark.parametrize("sort_by,expected", [
    ("shortest", [4, 1, 7, 5, 2, 3]),
    ("longest", [3, 2, 5, 7, 1, 4]),
    ("valence_low", [3, 1, 7, 5, 4, 2]),
    ("valence_high", [2, 4, 5, 7, 1, 3]),
])
def test_feel_sorting(feel, sort_by, expected):
    result = run_tool(feel, "find_songs_by_feel", {"min_valence": 0, "sort_by": sort_by})
    assert ids_of(result) == expected


def test_feel_random_order_still_returns_exactly_the_matches(feel):
    for _ in range(5):
        assert feel_ids(feel, genre="rock") == [1, 2, 7]


def test_feel_limit_and_count(feel):
    result = run_tool(feel, "find_songs_by_feel", {"min_valence": 0, "limit": 2})
    assert result["count"] == 6 and len(result["songs"]) == 2
    assert len(run_tool(feel, "find_songs_by_feel", {"min_valence": 0, "limit": 0})["songs"]) == 1


def test_feel_shows_one_copy_of_a_song_on_several_albums(feel):
    twin = dict(feel.songs[0], id=8, album="Other album")  # the same song as 1
    feel.songs.append(twin)
    assert feel_ids(feel, mood="sad", genre="rock") == [1]


def test_feel_skips_songs_without_a_length_when_asked_for_one(feel):
    feel.songs.append(dict(feel.songs[0], id=9, title="Unknown length", duration_seconds=""))
    assert 9 not in feel_ids(feel, max_minutes=60)
    assert 9 in feel_ids(feel, mood="sad")  # fine when length is not asked about


def test_feel_with_a_real_genre_but_nothing_playable_is_just_empty(feel):
    assert run_tool(feel, "find_songs_by_feel", {"genre": "jazz"}) == {"ok": True, "count": 0, "songs": []}


def test_feel_errors_say_what_is_valid(feel):
    no_filter = run_tool(feel, "find_songs_by_feel", {})
    assert no_filter["ok"] is False and "at least one filter" in no_filter["error"]
    genre = run_tool(feel, "find_songs_by_feel", {"genre": "zydeco"})
    assert genre["ok"] is False and "rock" in genre["error"] and "list_genres_and_moods" in genre["error"]
    mood = run_tool(feel, "find_songs_by_feel", {"mood": "sa"})
    assert mood["ok"] is False and "sad" in mood["error"]
    assert run_tool(feel, "find_songs_by_feel", {"min_valence": 0.9, "max_valence": 0.1})["ok"] is False
    assert run_tool(feel, "find_songs_by_feel", {"min_minutes": 5, "max_minutes": 1})["ok"] is False
    assert run_tool(feel, "find_songs_by_feel", {"mood": "sad", "sort_by": "loudest"})["ok"] is False


def test_feel_accepts_numbers_sent_as_text(feel):
    assert feel_ids(feel, max_valence="0.3", limit="5") == [1, 3]


def test_list_genres_and_moods(feel):
    result = run_tool(feel, "list_genres_and_moods", {})
    genres = {g["name"]: g["song_count"] for g in result["genres"]}
    assert genres == {"rock": 2, "pop": 2, "Rap & Hip-Hop": 1, "jazz": 1, "Alternative Rock": 1}
    assert result["genres"][0]["song_count"] == 2  # most common first
    assert {m["name"] for m in result["moods"]} == {"sad", "happy", "energetic", "chill", "intense"}


# --- the queue -----------------------------------------------------------------------

QUEUE_TOOLS = [
    ("read_queue", {}),
    ("add_song_to_queue", {"song_id": 5}),
    ("play_song_next", {"song_id": 5}),
    ("remove_song_from_queue", {"song_id": 5}),
    ("remove_artist_from_queue", {"artist": "Artist 1"}),
    ("move_song_in_queue", {"song_id": 5, "position": 1}),
]


@pytest.mark.parametrize("tool,args", QUEUE_TOOLS)
def test_queue_tools_need_something_playing(ctx, tool, args):
    result = run_tool(ctx, tool, args)
    assert result["ok"] is False and "no queue" in result["error"] and "play_song" in result["error"]


def test_read_queue(ctx, queue_box):
    set_queue(queue_box, [1, 2, 3])
    result = run_tool(ctx, "read_queue", {})
    assert result["now_playing"]["id"] == 1
    assert [(s["id"], s["position"]) for s in result["up_next"]] == [(2, 1), (3, 2)]
    assert result["up_next_count"] == 2 and result["repeat"] == "off" and result["shuffle"] is False


def test_read_queue_says_which_playlist_is_playing(ctx, store, user_playlist, queue_box):
    queue_box["queue"] = PlayQueue([1, 2, 3], 0, source=user_playlist, repeat="off")
    playing_from = run_tool(ctx, "read_queue", {})["playing_from"]
    assert playing_from["id"] == user_playlist and playing_from["kind"] == "user"
    queue_box["queue"] = PlayQueue([1, 2, 3], 0, repeat="off")
    assert "playing_from" not in run_tool(ctx, "read_queue", {})


def test_read_queue_shows_a_limited_number_of_songs(ctx, queue_box):
    set_queue(queue_box, list(range(1, 31)))
    result = run_tool(ctx, "read_queue", {})
    assert len(result["up_next"]) == at.MAX_QUEUE_SHOWN and result["up_next_count"] == 29
    assert "note" in result


def test_play_song_starts_a_new_queue_on_the_song(ctx, queue_box, played):
    set_queue(queue_box, [1, 2, 3])
    result = run_tool(ctx, "play_song", {"song_id": "7"})
    assert result == {"ok": True, "now_playing": {"id": 7, "title": "Song 7", "artist": "Artist 3", "album": "Album 1"}}
    assert played == [7] and queue_box["queue"].current_id == 7


def test_play_song_works_with_nothing_playing(ctx, queue_box, played):
    assert queue_box["queue"] is None
    assert run_tool(ctx, "play_song", {"song_id": 5})["ok"] is True
    assert played == [5]
    assert run_tool(ctx, "read_queue", {})["now_playing"]["id"] == 5  # the queue tools work now


def test_play_song_refusals(ctx, played):
    assert "find_song" in run_tool(ctx, "play_song", {"song_id": 999})["error"]
    assert "missing" in run_tool(ctx, "play_song", {"song_id": 13})["error"]  # no audio file
    assert played == []


def test_play_song_when_playback_cannot_start(library, queue_box):
    no_playback = at.AgentContext(songs=library, get_queue=lambda: None)
    assert run_tool(no_playback, "play_song", {"song_id": 5})["ok"] is False


def test_play_song_reports_a_song_that_did_not_start(library):
    stuck = at.AgentContext(songs=library, get_queue=lambda: None, start_playback=lambda song_id: None)
    result = run_tool(stuck, "play_song", {"song_id": 5})
    assert result["ok"] is False and "could not be started" in result["error"]


def test_find_album_by_name(ctx):
    result = run_tool(ctx, "find_album", {"name": "album 1"})
    assert result["ok"] and result["count"] == 1
    album = result["albums"][0]
    assert album == {"album": "Album 1", "artist": "Various artists", "song_count": 10, "release_year": 1995}


def test_find_album_by_artist_lists_every_album_with_one_of_their_songs(ctx):
    result = run_tool(ctx, "find_album", {"artist": "artist 1"})
    assert {a["album"] for a in result["albums"]} == {"Album 0", "Album 1", "Album 2", "Compilation"}
    compilation = next(a for a in result["albums"] if a["album"] == "Compilation")
    assert compilation["artist"] == "Artist 1" and compilation["song_count"] == 1


def test_find_album_needs_a_filter_and_can_find_nothing(ctx):
    assert run_tool(ctx, "find_album", {})["ok"] is False
    assert run_tool(ctx, "find_album", {"name": "zzz"}) == {"ok": True, "count": 0, "albums": []}


def test_a_song_without_an_album_is_filed_under_its_genre(library, queue_box):
    loose = dict(library[0], id=500, album="", genre="jazz")
    bare = at.AgentContext(songs=[loose], get_queue=lambda: None)
    assert run_tool(bare, "find_album", {"name": "jazz"})["albums"][0]["album"] == "Jazz"


def test_play_album_plays_it_under_its_exact_name(ctx, played_albums, queue_box):
    result = run_tool(ctx, "play_album", {"album": "album 0"})  # any letter case
    assert result["ok"] and result["album"] == "Album 0" and result["song_count"] == 10
    assert played_albums == ["Album 0"]
    assert result["now_playing"]["album"] == "Album 0"
    assert queue_box["queue"].current_id == result["now_playing"]["id"]


def test_play_album_refusals(library, ctx, played_albums):
    unknown = run_tool(ctx, "play_album", {"album": "No Such Album"})
    assert unknown["ok"] is False and "find_album" in unknown["error"]
    silent = at.AgentContext(songs=library, get_queue=lambda: None, playable=lambda s: False,
                             start_album=lambda name: None)
    assert "audio file" in run_tool(silent, "play_album", {"album": "Album 0"})["error"]
    assert played_albums == []


def test_play_album_when_playback_cannot_start_or_did_not_start(library):
    assert run_tool(at.AgentContext(songs=library, get_queue=lambda: None),
                    "play_album", {"album": "Album 0"})["ok"] is False
    stuck = at.AgentContext(songs=library, get_queue=lambda: None, start_album=lambda name: None)
    assert "could not be started" in run_tool(stuck, "play_album", {"album": "Album 0"})["error"]


def test_play_playlist_plays_a_playlist_or_a_mix(ctx, user_playlist, auto_mix, played_playlists, queue_box):
    mine = run_tool(ctx, "play_playlist", {"playlist_id": user_playlist})
    assert mine["ok"] and mine["playlist"] == "Faves" and mine["now_playing"]["id"] == 1
    mix = run_tool(ctx, "play_playlist", {"playlist_id": auto_mix})
    assert mix["ok"] and mix["playlist"] == "Work Mix"
    assert played_playlists == [user_playlist, auto_mix]
    assert queue_box["queue"].current_id in (1, 2, 3)


def test_play_playlist_refusals(library, ctx, store, played_playlists):
    unknown = run_tool(ctx, "play_playlist", {"playlist_id": "nope"})
    assert unknown["ok"] is False and "list_playlists" in unknown["error"]
    store["playlists"]["empty"] = {"id": "empty", "name": "Empty", "kind": "user", "song_ids": []}
    assert "no playable" in run_tool(ctx, "play_playlist", {"playlist_id": "empty"})["error"]
    store["playlists"]["gone"] = {"id": "gone", "name": "Gone", "kind": "user", "song_ids": [13]}
    assert "no playable" in run_tool(ctx, "play_playlist", {"playlist_id": "gone"})["error"]  # no audio file
    assert played_playlists == []


def test_play_playlist_when_playback_cannot_start_or_did_not_start(library, store, user_playlist):
    load = lambda: store
    assert run_tool(at.AgentContext(songs=library, get_queue=lambda: None, load_store=load),
                    "play_playlist", {"playlist_id": user_playlist})["ok"] is False
    stuck = at.AgentContext(songs=library, get_queue=lambda: None, load_store=load,
                            start_playlist=lambda pid: None)
    assert "could not be started" in run_tool(stuck, "play_playlist", {"playlist_id": user_playlist})["error"]


def test_add_song_to_queue(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3])
    result = run_tool(ctx, "add_song_to_queue", {"song_id": "5"})
    assert result["ok"] and result["position"] == 3 and result["added"]["id"] == 5
    assert queue.up_next == [2, 3, 5]
    assert run_tool(ctx, "add_song_to_queue", {"song_id": 2})["position"] == 3  # moved to the end
    assert queue.up_next == [3, 5, 2]


def test_add_song_to_queue_refusals(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3])
    assert "playing right now" in run_tool(ctx, "add_song_to_queue", {"song_id": 1})["error"]
    assert "missing" in run_tool(ctx, "add_song_to_queue", {"song_id": 13})["error"]
    assert "find_song" in run_tool(ctx, "add_song_to_queue", {"song_id": 999})["error"]
    assert queue.songs == [1, 2, 3]


def test_play_song_next(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3])
    assert run_tool(ctx, "play_song_next", {"song_id": 3})["will_play_next"]["id"] == 3
    assert queue.songs == [1, 3, 2]
    assert run_tool(ctx, "play_song_next", {"song_id": 1})["ok"] is False  # it is playing
    assert run_tool(ctx, "play_song_next", {"song_id": 13})["ok"] is False  # no audio file


def test_remove_song_from_queue_removes_every_copy_in_up_next(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3, 2, 4])
    result = run_tool(ctx, "remove_song_from_queue", {"song_id": 2})
    assert result == {"ok": True, "removed": 2}
    assert queue.songs == [1, 3, 4] and queue.removed == [2, 2]


def test_remove_song_from_queue_leaves_what_is_playing_and_history(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3], pos=1)  # 2 is playing, 1 is history
    assert run_tool(ctx, "remove_song_from_queue", {"song_id": 2})["ok"] is False
    assert run_tool(ctx, "remove_song_from_queue", {"song_id": 1})["ok"] is False
    assert queue.songs == [1, 2, 3] and queue.pos == 1


def test_remove_artist_from_queue(ctx, queue_box):
    queue = set_queue(queue_box, [4, 1, 5, 8, 9])  # 1, 5 and 9 are Artist 1
    assert run_tool(ctx, "remove_artist_from_queue", {"artist": "artist 1"}) == {"ok": True, "removed": 3}
    assert queue.songs == [4, 8]
    assert run_tool(ctx, "remove_artist_from_queue", {"artist": "Artist 1"})["ok"] is False


def test_remove_artist_from_queue_keeps_the_song_playing(ctx, queue_box):
    queue = set_queue(queue_box, [4, 1, 5, 8], pos=1)  # song 1 (Artist 1) is playing
    assert run_tool(ctx, "remove_artist_from_queue", {"artist": "Artist 1"})["removed"] == 1
    assert queue.songs == [4, 1, 8] and queue.current_id == 1


def test_move_song_in_queue(ctx, queue_box):
    queue = set_queue(queue_box, [1, 2, 3, 4, 5])
    assert run_tool(ctx, "move_song_in_queue", {"song_id": 5, "position": 1})["position"] == 1
    assert queue.up_next == [5, 2, 3, 4]
    assert run_tool(ctx, "move_song_in_queue", {"song_id": 5, "position": 99})["position"] == 4
    assert queue.up_next == [2, 3, 4, 5]
    run_tool(ctx, "move_song_in_queue", {"song_id": 4, "position": 0})
    assert queue.up_next == [4, 2, 3, 5]
    assert queue.current_id == 1


def test_move_song_in_queue_needs_the_song_to_be_in_up_next(ctx, queue_box):
    set_queue(queue_box, [1, 2, 3], pos=1)
    assert "add_song_to_queue" in run_tool(ctx, "move_song_in_queue", {"song_id": 1, "position": 1})["error"]
    assert run_tool(ctx, "move_song_in_queue", {"song_id": 2, "position": 1})["ok"] is False
