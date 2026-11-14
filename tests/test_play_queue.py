import random
from pathlib import Path

from src import playlists as plib
from src.play_queue import QUEUE_AHEAD, QUEUE_HISTORY, REPEAT_MODES, PlayQueue, next_repeat_mode
from src.recommender import load_songs

SONGS = load_songs(str(Path(__file__).resolve().parent.parent / "data" / "songs.csv"))
BY_ID = {s["id"]: s for s in SONGS}


def test_single_song_gets_recommendations_after_it():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)
    assert queue.songs[0] == SONGS[0]["id"] and queue.pos == 0
    assert queue.current_id == SONGS[0]["id"]
    assert len(queue.up_next) == QUEUE_AHEAD
    assert len(set(queue.songs)) == len(queue.songs)  # no repeats
    assert queue.source is None


def test_unknown_song_gives_no_queue():
    assert PlayQueue.start(-1, SONGS) is None


def test_playlist_is_queued_in_full_then_extended_as_it_runs_low():
    store = {"playlists": {}, "recent": []}
    ids = [s["id"] for s in SONGS[:QUEUE_AHEAD + 5]]
    pid = plib.create_playlist(store, "mine", ids, SONGS)
    playlist = store["playlists"][pid]

    queue = PlayQueue.start(ids[0], SONGS, playlist)
    assert queue.songs == ids  # all songs queued, in order; enough are up next, so no top-up yet
    assert queue.source == pid

    # Move forward: once fewer than QUEUE_AHEAD remain, one song is added at a time
    for _ in range(6):
        queue.advance(1, SONGS)
    assert queue.songs[:len(ids)] == ids
    assert len(queue.songs) > len(ids)
    assert len(queue.up_next) == QUEUE_AHEAD
    queue.advance(1, SONGS)
    assert len(queue.up_next) == QUEUE_AHEAD
    assert len(set(queue.songs)) == len(queue.songs)


def test_starting_from_a_row_inside_a_playlist():
    store = {"playlists": {}, "recent": []}
    ids = [s["id"] for s in SONGS[:12]]
    pid = plib.create_playlist(store, "mine", ids, SONGS)
    queue = PlayQueue.start(ids[4], SONGS, store["playlists"][pid])
    assert queue.pos == 4 and queue.history == ids[:4]
    assert queue.advance(-1, SONGS) == ids[3]  # earlier playlist songs stay reachable


def test_previous_at_start_returns_current_and_next_skips_unplayable():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)
    assert queue.advance(-1, SONGS) == SONGS[0]["id"] and queue.pos == 0
    skipped = queue.songs[1]
    wanted = queue.songs[2]
    assert queue.advance(1, SONGS, playable=lambda s: s["id"] != skipped) == wanted
    assert queue.pos == 2


def test_jump_makes_that_row_current_and_rejects_bad_rows():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)
    target = queue.songs[5]
    assert queue.jump(5, SONGS) == target and queue.current_id == target
    assert len(queue.up_next) == QUEUE_AHEAD
    assert queue.jump(10_000, SONGS) is None and queue.jump(-1, SONGS) is None


def test_history_is_trimmed():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)
    for _ in range(QUEUE_HISTORY + 20):
        queue.advance(1, SONGS)
    assert queue.pos <= QUEUE_HISTORY
    assert len(queue.up_next) == QUEUE_AHEAD


def test_nothing_playable_stops_without_looping_forever():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS, playable=lambda s: False)
    assert queue.up_next == []
    assert queue.advance(1, SONGS, playable=lambda s: False) is None


def _fixed_queue(n=6, pos=2):
    ids = [s["id"] for s in SONGS[:n]]
    return PlayQueue(ids[:], pos), ids


def test_remove_before_current_keeps_the_same_song_playing():
    queue, ids = _fixed_queue()
    assert queue.remove(0)
    assert queue.songs == ids[1:] and queue.pos == 1
    assert queue.current_id == ids[2]


def test_remove_after_current_and_the_playing_song_cannot_be_removed():
    queue, ids = _fixed_queue()
    assert queue.remove(4)
    assert queue.songs == ids[:4] + ids[5:] and queue.pos == 2
    assert not queue.remove(2)  # the playing song stays
    assert not queue.remove(99) and not queue.remove(-1)


def test_remove_tops_the_queue_back_up_when_given_the_library():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)
    gone = queue.songs[3]
    assert queue.remove(3, SONGS)
    assert gone not in queue.songs
    assert len(queue.up_next) == QUEUE_AHEAD


def test_play_next_puts_the_song_right_after_the_current_one():
    queue, ids = _fixed_queue(8, pos=2)
    newcomer = SONGS[40]["id"]
    assert queue.play_next(newcomer)
    assert queue.songs[:5] == ids[:3] + [newcomer, ids[3]]
    assert queue.current_id == ids[2] and queue.pos == 2  # still playing the same song


def test_play_next_moves_a_song_that_is_already_in_the_queue():
    queue, ids = _fixed_queue(8, pos=2)
    assert queue.play_next(ids[6])  # from further down
    assert queue.songs == ids[:3] + [ids[6]] + ids[3:6] + ids[7:]
    assert queue.play_next(ids[0])  # from the history: the playing song moves up one place
    assert queue.songs[queue.pos] == ids[2] and queue.songs[queue.pos + 1] == ids[0]
    assert queue.pos == 1
    assert len(set(queue.songs)) == len(queue.songs)  # never twice


def test_add_goes_to_the_very_end_and_moves_an_existing_song_there():
    queue, ids = _fixed_queue(8, pos=2)
    newcomer = SONGS[41]["id"]
    assert queue.add(newcomer) and queue.songs[-1] == newcomer
    assert queue.add(ids[3]) and queue.songs[-1] == ids[3]
    assert queue.songs.count(newcomer) == 1 and queue.current_id == ids[2]


def test_the_playing_song_cannot_be_queued_again_and_a_removed_song_can_come_back():
    queue, ids = _fixed_queue(8, pos=2)
    assert not queue.play_next(ids[2]) and not queue.add(ids[2])
    assert queue.songs == ids

    queue.remove(5)
    assert ids[5] in queue.removed  # the recommender will not bring it back...
    assert queue.add(ids[5]) and queue.songs[-1] == ids[5]  # ...but the listener can
    assert ids[5] not in queue.removed


# --- repeat modes ---

def _playlist(n=8):
    ids = [s["id"] for s in SONGS[:n]]
    return {"id": "p", "song_ids": ids, "profile": None}, ids


def test_the_repeat_button_cycles_off_all_one_autoplay_and_round_again():
    assert REPEAT_MODES == ("off", "all", "one", "autoplay")
    assert [next_repeat_mode(m) for m in REPEAT_MODES] == ["all", "one", "autoplay", "off"]
    assert next_repeat_mode("nonsense") == "off"


def test_autoplay_is_the_default_and_still_recommends():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[0], SONGS, playlist)
    assert queue.repeat == "autoplay" and not queue.shuffle
    assert len(queue.up_next) == QUEUE_AHEAD and queue.up_next[:7] == ids[1:]
    assert set(queue.up_next[7:]).isdisjoint(ids)  # the rest are recommendations


def test_repeat_off_adds_nothing_and_stops_after_the_last_song():
    playlist, ids = _playlist(5)
    queue = PlayQueue.start(ids[0], SONGS, playlist, repeat="off")
    assert queue.songs == ids
    assert [queue.advance(1, SONGS) for _ in range(4)] == ids[1:]
    assert queue.advance(1, SONGS) is None and queue.current_id == ids[4]

    alone = PlayQueue.start(SONGS[0]["id"], SONGS, repeat="off")
    assert alone.songs == [SONGS[0]["id"]] and alone.advance(1, SONGS) is None


def test_turning_repeat_off_drops_recommendations_but_keeps_what_the_listener_added():
    queue = PlayQueue.start(SONGS[0]["id"], SONGS)  # autoplay: recommendations queued
    extra = SONGS[50]["id"]
    queue.add(extra)
    assert len(queue.up_next) > 1

    queue.set_repeat("off", SONGS)

    assert queue.up_next == [extra]


def test_repeat_all_queues_the_collection_again_instead_of_recommending():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[0], SONGS, playlist, repeat="all")
    assert queue.songs[:16] == ids + ids  # a second pass, from the start
    assert len(queue.up_next) >= QUEUE_AHEAD

    for _ in range(40):  # it never runs out, and never wanders off to other songs
        assert queue.advance(1, SONGS) is not None
        assert len(queue.up_next) >= QUEUE_AHEAD
    assert set(queue.songs) <= set(ids)


def test_repeat_all_on_a_single_song_repeats_that_song():
    queue = PlayQueue.start(SONGS[3]["id"], SONGS, repeat="all")
    assert queue.up_next == [SONGS[3]["id"]] * QUEUE_AHEAD


def test_repeat_one_fills_the_queue_with_the_song_and_clears_it_when_turned_off():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[2], SONGS, playlist, repeat="one")
    assert queue.up_next[:QUEUE_AHEAD] == [ids[2]] * QUEUE_AHEAD

    assert queue.advance(1, SONGS) == ids[2]  # the song plays again...
    assert queue.up_next[:QUEUE_AHEAD] == [ids[2]] * QUEUE_AHEAD  # ...and the queue is full again

    queue.set_repeat("off", SONGS)  # the copies go; the rest of the playlist is still there
    assert queue.up_next == ids[3:]


def test_switching_between_modes_takes_out_what_the_old_mode_added():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[0], SONGS, playlist, repeat="all")

    queue.set_repeat("autoplay", SONGS)  # repeated passes go, recommendations come
    assert queue.up_next[:7] == ids[1:] and len(queue.up_next) == QUEUE_AHEAD
    assert set(queue.up_next[7:]).isdisjoint(ids)

    queue.set_repeat("all", SONGS)  # recommendations go, the playlist repeats
    assert set(queue.up_next) <= set(ids) and len(queue.up_next) >= QUEUE_AHEAD


# --- shuffle ---

def test_shuffle_keeps_the_playing_song_and_plays_the_rest_in_another_order():
    playlist, ids = _playlist(12)
    queue = PlayQueue.start(ids[3], SONGS, playlist, repeat="off", rng=random.Random(1))

    queue.set_shuffle(True, SONGS)

    assert queue.current_id == ids[3] and queue.shuffle
    assert sorted(queue.up_next) == sorted(set(ids) - {ids[3]})  # every other song, once
    assert queue.up_next != [i for i in ids if i != ids[3]]  # but not in the original order

    queue.set_shuffle(False, SONGS)  # back in order, from after the song playing
    assert queue.up_next == ids[4:]


def test_starting_a_playlist_with_shuffle_begins_on_the_chosen_song():
    playlist, ids = _playlist(10)
    queue = PlayQueue.start(ids[5], SONGS, playlist, repeat="off", shuffle=True, rng=random.Random(2))

    assert queue.songs[0] == ids[5] and queue.history == []
    assert sorted(queue.songs) == sorted(ids)


def test_shuffle_leaves_songs_the_listener_added_alone():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[0], SONGS, playlist, repeat="off", rng=random.Random(3))
    extra = SONGS[60]["id"]
    queue.add(extra)

    queue.set_shuffle(True, SONGS)

    assert queue.up_next[-1] == extra and sorted(queue.up_next[:-1]) == sorted(ids[1:])


def test_shuffle_with_repeat_all_shuffles_every_pass():
    playlist, ids = _playlist(8)
    queue = PlayQueue.start(ids[0], SONGS, playlist, repeat="all", shuffle=True, rng=random.Random(4))

    first_pass, second_pass = queue.songs[:8], queue.songs[8:16]
    assert sorted(first_pass) == sorted(ids) and first_pass[0] == ids[0]
    assert sorted(second_pass) == sorted(ids)  # the next pass has every song again
