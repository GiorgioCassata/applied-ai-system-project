import random
from datetime import datetime
from pathlib import Path

from src import playlists as plib
from src.recommender import load_songs

SONGS = load_songs(str(Path(__file__).resolve().parent.parent / "data" / "songs.csv"))


def test_store_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "playlists.json"
    assert plib.load_store(path) == {"playlists": {}, "recent": []}
    store = plib.load_store(path)
    pid = plib.create_playlist(store, " Mine ", [SONGS[0]["id"], SONGS[1]["id"], -5], SONGS)
    plib.mark_played(store, pid)
    plib.save_store(store, path)
    loaded = plib.load_store(path)
    assert loaded["playlists"][pid]["name"] == "Mine"
    assert loaded["playlists"][pid]["song_ids"] == [SONGS[0]["id"], SONGS[1]["id"]]  # unknown id dropped
    assert loaded["recent"] == [pid]
    assert not path.with_name(path.name + ".tmp").exists()


def test_corrupt_file_gives_empty_store(tmp_path):
    path = tmp_path / "playlists.json"
    path.write_text("{not json")
    assert plib.load_store(path) == {"playlists": {}, "recent": []}


def test_recent_is_newest_first_and_limited():
    store = {"playlists": {}, "recent": []}
    ids = [plib.create_playlist(store, f"p{i}", [SONGS[0]["id"]], SONGS) for i in range(8)]
    for pid in ids:
        plib.mark_played(store, pid)
    plib.mark_played(store, ids[0])
    recent = plib.recent_playlists(store, 6)
    assert len(recent) == 6
    assert recent[0]["id"] == ids[0] and recent[1]["id"] == ids[7]


def test_edit_playlist():
    store = {"playlists": {}, "recent": []}
    pid = plib.create_playlist(store, "x", [SONGS[0]["id"]], SONGS)
    plib.add_song(store, pid, SONGS[1]["id"])
    plib.add_song(store, pid, SONGS[1]["id"])  # no duplicates
    plib.remove_song(store, pid, SONGS[0]["id"])
    assert store["playlists"][pid]["song_ids"] == [SONGS[1]["id"]]
    plib.mark_played(store, pid)
    plib.delete_playlist(store, pid)
    assert store == {"playlists": {}, "recent": []}


def test_daypart_boundaries():
    at = lambda h: plib.current_daypart(datetime(2026, 1, 1, h))
    assert [at(h) for h in (0, 4, 5, 10, 11, 15, 16, 18, 19, 20, 21, 23)] == [
        "night", "night", "morning", "morning", "midday", "midday",
        "afternoon", "afternoon", "evening", "evening", "night", "night",
    ]
    # night keeps the same auto day through midnight
    assert plib.auto_day(datetime(2026, 1, 1, 23)) == plib.auto_day(datetime(2026, 1, 2, 3))
    assert plib.auto_day(datetime(2026, 1, 2, 5)) != plib.auto_day(datetime(2026, 1, 2, 4))


def test_build_mix_length_and_no_repeats():
    ids = plib.build_mix({"energy": 0.9, "valence": 0.9}, SONGS, random.Random(1))
    assert len(ids) == len(set(ids))
    total = sum(int(float(s["duration_seconds"])) for s in SONGS if s["id"] in set(ids))
    assert plib.TARGET_SECONDS <= total < plib.TARGET_SECONDS + 15 * 60


def test_time_profiles_differ_in_the_requested_direction():
    by_part = {}
    for pid, _title, _start, profile in plib.DAYPARTS:
        ids = set(plib.build_mix(profile, SONGS, random.Random(2)))
        chosen = [s for s in SONGS if s["id"] in ids]
        by_part[pid] = (
            sum(s["energy"] for s in chosen) / len(chosen),
            sum(s["valence"] for s in chosen) / len(chosen),
        )
    assert by_part["midday"][0] > by_part["afternoon"][0] > by_part["evening"][0]
    assert by_part["evening"][0] > by_part["night"][0]
    assert by_part["midday"][0] > by_part["morning"][0]
    assert by_part["night"][1] == min(v for _e, v in by_part.values())


def test_refresh_creates_then_only_regenerates_on_a_new_day():
    store = {"playlists": {}, "recent": []}
    day1 = datetime(2026, 3, 1, 9)
    assert plib.refresh_auto_playlists(store, SONGS, day1)
    assert len(plib.auto_playlists(store, "time")) == 5
    assert len(plib.auto_playlists(store, "activity")) == len(plib.ACTIVITIES) == 7
    assert len(plib.auto_playlists(store, "genre")) == 6
    snapshot = {k: list(v["song_ids"]) for k, v in store["playlists"].items()}
    assert not plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 1, 22))  # same auto day
    assert not plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 2, 3))  # still night of Mar 1
    assert plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 2, 6))
    assert store["playlists"]["auto-time-morning"]["song_ids"] != snapshot["auto-time-morning"]
    assert set(store["playlists"]) == set(snapshot)  # same ids, so "recent" stays valid


def test_changed_profile_regenerates_a_playlist_the_same_day():
    store = {"playlists": {}, "recent": []}
    now = datetime(2026, 3, 1, 9)
    plib.refresh_auto_playlists(store, SONGS, now)
    store["playlists"]["auto-time-evening"]["profile"] = {"energy": 0.99}  # stale profile
    store["playlists"]["auto-time-evening"]["name"] = "Afternoon & Evening"
    assert plib.refresh_auto_playlists(store, SONGS, now)
    assert store["playlists"]["auto-time-evening"]["name"] == "Evening"
    assert not plib.refresh_auto_playlists(store, SONGS, now)


def test_recommended_playlists_order_and_activity_per_daypart():
    store = {"playlists": {}, "recent": []}
    plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 1, 9))
    # one activity mix per part of the day, as asked: work, work, workout, study, chill
    expected = {9: "Work Mix", 13: "Work Mix", 17: "Workout Mix", 20: "Study Mix", 23: "Chill Mix"}
    for hour, mix in expected.items():
        recs = plib.recommended_playlists(store, datetime(2026, 3, 1, hour))
        assert recs[0]["id"] == f"auto-time-{plib.current_daypart(datetime(2026, 3, 1, hour))}"
        assert recs[1]["name"] == mix
        assert len(recs) == 2  # no playlists of your own yet

    # then the playlists you created fill the rest, newest first by creation time (not by
    # when they were played), up to 6 in all
    when = datetime(2026, 3, 1, 9)
    ids = [
        plib.create_playlist(store, f"mine{n}", [SONGS[0]["id"]], SONGS, now=when.replace(minute=n))
        for n in range(6)
    ]
    plib.mark_played(store, ids[0])  # playing the oldest one must not move it up
    recs = plib.recommended_playlists(store, when)
    assert [p["id"] for p in recs[:2]] == ["auto-time-morning", "auto-act-work"]
    assert [p["id"] for p in recs[2:]] == [ids[5], ids[4], ids[3], ids[2]]
    assert len(recs) == 6


def _hues(gradient):
    import re
    return [int(h) for h in re.findall(r"hsl\((\d+),", gradient)]


def test_gradient_bottom_right_is_never_near_the_green_accent():
    seeds = [f"seed-{n}" for n in range(5000)] + [f"{n:08x}" for n in range(5000)]
    seeds += ["auto-genre-r&b", "auto-genre-easy listening", "auto-time-morning", ""]
    for seed in seeds:
        first, second = _hues(plib.generate_gradient(seed))
        assert plib._hue_distance(second, plib.ACCENT_HUE) > plib.ACCENT_AVOID, (seed, second)
        assert plib._hue_distance(first, second) >= 50, (seed, first, second)  # still two colours


def test_gradient_is_stable_and_varied():
    assert plib.generate_gradient("abc") == plib.generate_gradient("abc")
    many = {plib.generate_gradient(f"playlist-{n}") for n in range(500)}
    assert len(many) > 450  # different playlists nearly always look different
    # the second colour still spreads over most of the allowed colour wheel
    seconds = {_hues(g)[1] // 30 for g in many}
    assert len(seconds) >= 8


def test_next_playlist_name_uses_the_lowest_free_number():
    store = {"playlists": {}, "recent": []}
    assert plib.next_playlist_name(store) == "Playlist 1"
    plib.create_playlist(store, "Playlist 1", [SONGS[0]["id"]], SONGS)
    assert plib.next_playlist_name(store) == "Playlist 2"
    pid3 = plib.create_playlist(store, " playlist 3 ", [SONGS[0]["id"]], SONGS)  # case/space ignored
    assert plib.next_playlist_name(store) == "Playlist 2"  # the gap is filled first
    plib.create_playlist(store, "Playlist 2", [SONGS[0]["id"]], SONGS)
    assert plib.next_playlist_name(store) == "Playlist 4"
    plib.delete_playlist(store, pid3)
    assert plib.next_playlist_name(store) == "Playlist 3"
    # automatic mixes never take a number, even with a matching name
    store["playlists"]["auto-x"] = {"id": "auto-x", "name": "Playlist 3", "kind": "auto", "song_ids": []}
    assert plib.next_playlist_name(store) == "Playlist 3"


def test_profile_from_songs_and_score():
    chosen = SONGS[:5]
    profile = plib.profile_from_songs(chosen)
    assert 0 <= profile["energy"] <= 1
    playlist = {"profile": profile}
    assert 0 <= plib.song_score(playlist, chosen[0]) <= 1
    assert plib.profile_from_songs([]) == {}


def test_update_playlist_changes_name_songs_order_and_profile():
    store = {"playlists": {}, "recent": []}
    ids = [s["id"] for s in SONGS[:6]]
    pid = plib.create_playlist(store, "old", ids, SONGS)
    before = store["playlists"][pid]["profile"]

    new_ids = [ids[4], ids[1], ids[1], -999, SONGS[20]["id"]]  # reordered, a repeat, an unknown id
    plib.update_playlist(store, pid, "  new name ", new_ids, SONGS)
    playlist = store["playlists"][pid]

    assert playlist["name"] == "new name"
    assert playlist["song_ids"] == [ids[4], ids[1], SONGS[20]["id"]]
    assert playlist["profile"] != before
    assert playlist["kind"] == "user" and playlist["id"] == pid


def test_update_playlist_with_no_valid_songs_keeps_the_old_profile():
    store = {"playlists": {}, "recent": []}
    pid = plib.create_playlist(store, "mine", [SONGS[0]["id"]], SONGS)
    before = store["playlists"][pid]["profile"]

    plib.update_playlist(store, pid, "mine", [-1], SONGS)

    assert store["playlists"][pid]["song_ids"] == [] and store["playlists"][pid]["profile"] == before


def test_unique_playlist_name_numbers_duplicates_from_2():
    store = {"playlists": {}, "recent": []}
    assert plib.unique_playlist_name(store, "My Work Mix") == "My Work Mix"

    names = []
    for _ in range(3):
        name = plib.unique_playlist_name(store, "My Work Mix")
        plib.create_playlist(store, name, [SONGS[0]["id"]], SONGS)
        names.append(name)
    assert names == ["My Work Mix", "My Work Mix 2", "My Work Mix 3"]

    # case does not matter, and a free number left by a deleted playlist is reused
    assert plib.unique_playlist_name(store, "my work mix") == "my work mix 4"
    pid = next(p["id"] for p in store["playlists"].values() if p["name"] == "My Work Mix 2")
    plib.delete_playlist(store, pid)
    assert plib.unique_playlist_name(store, "My Work Mix") == "My Work Mix 2"


def test_regenerate_auto_playlist_makes_a_new_mix_and_keeps_its_identity():
    store = {"playlists": {}, "recent": []}
    plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 1, 9))
    pid = "auto-act-work"
    before = dict(store["playlists"][pid], song_ids=list(store["playlists"][pid]["song_ids"]))

    assert plib.regenerate_auto_playlist(store, pid, SONGS, random.Random(7))
    after = store["playlists"][pid]

    assert after["song_ids"] != before["song_ids"] and len(set(after["song_ids"])) == len(after["song_ids"])
    for key in ("id", "name", "kind", "group", "profile", "slot"):
        assert after[key] == before[key]
    # it stays put for the rest of the day instead of being rebuilt
    assert not plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 1, 22))
    assert store["playlists"][pid]["song_ids"] == after["song_ids"]


def test_regenerate_leaves_your_own_playlists_alone():
    store = {"playlists": {}, "recent": []}
    pid = plib.create_playlist(store, "mine", [s["id"] for s in SONGS[:5]], SONGS)
    before = list(store["playlists"][pid]["song_ids"])

    assert not plib.regenerate_auto_playlist(store, pid, SONGS)
    assert not plib.regenerate_auto_playlist(store, "no-such-id", SONGS)
    assert store["playlists"][pid]["song_ids"] == before


# --- a mix built from one song ---

def _song(i, artist, **kw):
    base = {"id": i, "title": f"Song {i}", "artist": artist, "genre": "pop", "mood": "happy",
            "energy": 0.8, "tempo_bpm": 120.0, "valence": 0.8, "acousticness": 0.2,
            "duration_seconds": 200}
    base.update(kw)
    return base


def test_song_mix_starts_with_the_song_and_never_repeats():
    songs = [s for s in SONGS if s["id"] != SONGS[7]["id"]]
    mix = plib.build_song_mix(SONGS[7], SONGS, random.Random(1))

    assert mix[0] == SONGS[7]["id"]
    assert len(mix) == len(set(mix)) and len(mix) > 5
    assert all(i in {s["id"] for s in SONGS} for i in mix)


def test_song_mix_is_filled_with_songs_that_resemble_it():
    seed = _song(1, "A", genre="metal", mood="angry", energy=0.95, valence=0.2, tempo_bpm=170)
    close = [_song(10 + i, "B", genre="metal", mood="angry", energy=0.93, valence=0.2, tempo_bpm=165) for i in range(3)]
    far = [_song(20 + i, "C", genre="jazz", mood="relaxed", energy=0.2, valence=0.7, tempo_bpm=70) for i in range(3)]

    # 200 seconds a song: the seed plus three songs reach the 800 second target
    mix = plib.build_song_mix(seed, [seed] + far + close, random.Random(2), target_seconds=800)
    assert mix[0] == 1 and set(mix[1:4]) == {10, 11, 12}  # the similar ones come right after it


def test_same_artist_gets_a_slight_bias_that_is_not_a_takeover():
    seed = _song(1, "Seed Artist")
    # A song by the same artist that is a little worse a fit, and a rival that is a bit better
    same = _song(2, "Seed Artist", energy=0.70, valence=0.70)
    rival = _song(3, "Other", energy=0.76, valence=0.76)
    library = [seed, same, rival]

    without = plib.build_song_mix(seed, library, random.Random(3), 300, artist_boost=0.0)
    with_boost = plib.build_song_mix(seed, library, random.Random(3), 300, artist_boost=0.30)
    assert without[1] == 3  # on merit the rival is first
    assert with_boost[1] == 2  # the bias lifts the same artist's song above it

    # ...but a song that is a far worse fit is not rescued by being from the same artist
    bad_same = _song(4, "Seed Artist", genre="jazz", mood="relaxed", energy=0.1, valence=0.1, tempo_bpm=60)
    ordered = plib.build_song_mix(seed, [seed, bad_same, rival], random.Random(3), 300)
    assert ordered[1] == 3


def test_song_mix_length_is_about_the_target():
    mix = plib.build_song_mix(SONGS[0], SONGS, random.Random(4), target_seconds=900)
    assert 3 <= len(mix) <= 12  # unknown durations count as ~3.5 minutes


# --- temporary song mixes ---

def _store_with_auto_mixes(now=datetime(2026, 3, 1, 9), user_prefs=None):
    store = {"playlists": {}, "recent": []}
    plib.refresh_auto_playlists(store, SONGS, now, user_prefs=user_prefs)
    return store, now


def test_a_song_mix_is_a_temporary_auto_playlist_that_starts_with_its_song():
    store, now = _store_with_auto_mixes()
    pid = plib.create_song_mix(store, SONGS[3], SONGS, now, random.Random(1))
    mix = store["playlists"][pid]

    assert mix["kind"] == "auto" and mix["group"] == "song" and mix["seed_id"] == SONGS[3]["id"]
    assert mix["name"] == f"{SONGS[3]['title']} Mix"
    assert mix["song_ids"][0] == SONGS[3]["id"]
    # it is not a playlist of the user's: it is listed among the recommended ones
    assert mix not in [p for p in store["playlists"].values() if p["kind"] == "user"]
    assert mix in plib.recommended_playlists(store, now)


def test_song_mixes_sit_after_the_base_mixes_and_before_the_users_playlists():
    store, now = _store_with_auto_mixes()
    mine = plib.create_playlist(store, "mine", [SONGS[0]["id"]], SONGS, now=now)
    first = plib.create_song_mix(store, SONGS[1], SONGS, datetime(2026, 3, 1, 10), random.Random(1))
    second = plib.create_song_mix(store, SONGS[2], SONGS, datetime(2026, 3, 1, 11), random.Random(1))

    ids = [p["id"] for p in plib.recommended_playlists(store, now)]
    assert ids[:2] == ["auto-time-morning", "auto-act-work"]
    assert ids[2:5] == [second, first, mine]  # newest song mix first, then your playlist


def test_the_oldest_song_mix_leaves_once_the_recommended_list_is_full():
    store, now = _store_with_auto_mixes()  # two base mixes -> room for four song mixes
    made = []
    for i in range(6):
        made.append(plib.create_song_mix(store, SONGS[i], SONGS, datetime(2026, 3, 1, 10, i), random.Random(1)))

    kept = [p["id"] for p in plib.song_mixes(store)]
    assert kept == made[:1:-1]  # the newest four, newest first
    assert all(old not in store["playlists"] for old in made[:2])  # the oldest two are gone
    assert len(plib.recommended_playlists(store, now)) == 6

    # with a "Made for You" mix there is one place fewer
    store2, now = _store_with_auto_mixes(user_prefs={"energy": 0.5})
    for i in range(6):
        plib.create_song_mix(store2, SONGS[i], SONGS, datetime(2026, 3, 1, 10, i), random.Random(1))
    assert len(plib.song_mixes(store2)) == 3


def test_a_mix_from_the_same_song_is_rebuilt_not_duplicated():
    store, now = _store_with_auto_mixes()
    a = plib.create_song_mix(store, SONGS[5], SONGS, datetime(2026, 3, 1, 10), random.Random(1))
    plib.create_song_mix(store, SONGS[6], SONGS, datetime(2026, 3, 1, 11), random.Random(1))
    b = plib.create_song_mix(store, SONGS[5], SONGS, datetime(2026, 3, 1, 12), random.Random(1))

    assert a == b and len(plib.song_mixes(store)) == 2
    assert plib.song_mixes(store)[0]["id"] == a  # moved back to the front


def test_song_mixes_survive_the_daily_refresh_of_auto_playlists():
    store, now = _store_with_auto_mixes()
    pid = plib.create_song_mix(store, SONGS[4], SONGS, now, random.Random(1))
    before = list(store["playlists"][pid]["song_ids"])

    plib.refresh_auto_playlists(store, SONGS, datetime(2026, 3, 3, 9))  # two days later

    assert store["playlists"][pid]["song_ids"] == before


def test_regenerating_a_song_mix_keeps_its_song_first_and_changes_the_rest():
    store, now = _store_with_auto_mixes()
    pid = plib.create_song_mix(store, SONGS[8], SONGS, now, random.Random(1))
    before = list(store["playlists"][pid]["song_ids"])

    assert plib.regenerate_auto_playlist(store, pid, SONGS, random.Random(9))
    after = store["playlists"][pid]["song_ids"]

    assert after[0] == SONGS[8]["id"] and after != before
    assert len(set(after)) == len(after)
