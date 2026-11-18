from datetime import datetime
from pathlib import Path

from src import playlists as plib
from src import user_profile as up
from src.recommender import load_songs

CATALOG = load_songs(str(Path(__file__).resolve().parent / "data" / "catalog.csv"))
NOW = datetime(2026, 3, 1, 9)


def test_no_file_means_no_profile(tmp_path):
    assert up.load_profile(tmp_path / "missing.json") is None


def test_profile_round_trip(tmp_path):
    path = tmp_path / "profile.json"
    saved = up.save_profile(
        {"genre": "jazz", "mood": "relaxed", "energy": 0.37, "valence": 0.71,
         "tempo_bpm": 90, "likes_acoustic": True},
        path,
    )
    assert up.load_profile(path) == saved == {
        "genre": "jazz", "mood": "relaxed", "energy": 0.37, "valence": 0.71,
        "tempo_bpm": 90, "likes_acoustic": True,
    }


def test_anything_blank_or_missing_means_no_preference():
    prefs = up.clean_profile({"genre": "  ", "mood": None, "energy": 0.6})
    assert prefs == {"energy": 0.6}  # nothing else is invented

    assert up.clean_profile({}) == {}
    assert up.clean_profile({"valence": None, "tempo_bpm": None, "likes_acoustic": None}) == {}


def test_acoustic_is_three_way():
    assert up.clean_profile({"likes_acoustic": True}) == {"likes_acoustic": True}
    assert up.clean_profile({"likes_acoustic": False}) == {"likes_acoustic": False}  # a real answer
    assert "likes_acoustic" not in up.clean_profile({"likes_acoustic": None})
    assert "likes_acoustic" not in up.clean_profile({"likes_acoustic": "yes"})  # not a bool: ignored


def test_a_profile_with_no_preferences_is_saved_and_is_not_the_same_as_no_profile(tmp_path):
    path = tmp_path / "profile.json"
    assert up.load_profile(path) is None
    assert up.save_profile({}, path) == {}
    assert up.load_profile(path) == {}  # "I don't mind anything", not "never set up"


def test_numbers_are_pulled_into_range():
    prefs = up.clean_profile({"energy": 3, "valence": -1, "tempo_bpm": 999})
    assert prefs["energy"] == 1.0 and prefs["valence"] == 0.0 and prefs["tempo_bpm"] == up.TEMPO_MAX
    assert up.clean_profile({"tempo_bpm": 10})["tempo_bpm"] == up.TEMPO_MIN


def test_corrupt_file_gives_no_profile_and_clear_removes_it(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json", encoding="utf-8")
    assert up.load_profile(path) is None

    up.save_profile({"energy": 0.2}, path)
    assert up.load_profile(path) is not None
    up.clear_profile(path)
    assert up.load_profile(path) is None
    up.clear_profile(path)  # clearing again is fine


# --- the "For You" mix ---

PREFS = {"genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.6,
         "tempo_bpm": 72, "likes_acoustic": True}


def test_made_for_you_mix_follows_the_profile_and_comes_first_at_home():
    store = {"playlists": {}, "recent": []}
    plib.refresh_auto_playlists(store, CATALOG, NOW, user_prefs=PREFS)

    mix = store["playlists"][plib.YOU_MIX_ID]
    assert mix["name"] == "For You" and mix["profile"] == PREFS and mix["song_ids"]
    assert plib.recommended_playlists(store, NOW)[0]["id"] == plib.YOU_MIX_ID


def test_changing_the_profile_rebuilds_the_mix_and_removing_it_deletes_the_mix():
    store = {"playlists": {}, "recent": []}
    plib.refresh_auto_playlists(store, CATALOG, NOW, user_prefs=PREFS)
    assert not plib.refresh_auto_playlists(store, CATALOG, NOW, user_prefs=PREFS)  # unchanged

    changed = dict(PREFS, genre="metal", mood="angry", energy=0.95)
    assert plib.refresh_auto_playlists(store, CATALOG, NOW, user_prefs=changed)
    assert store["playlists"][plib.YOU_MIX_ID]["profile"] == changed

    assert plib.refresh_auto_playlists(store, CATALOG, NOW)  # no profile any more
    assert plib.YOU_MIX_ID not in store["playlists"]
    assert plib.recommended_playlists(store, NOW)[0]["id"] != plib.YOU_MIX_ID
