from src import search

SONGS = [
    {"id": 1, "title": "Hello", "artist": "Adele", "album": "25", "genre": "pop"},
    {"id": 2, "title": "Lovely Day", "artist": "Bill Withers", "album": "Menagerie", "genre": "soul"},
    {"id": 3, "title": "Hello Goodbye", "artist": "The Beatles", "album": "", "genre": "rock"},
    {"id": 4, "title": "The", "artist": "Nobody", "album": "Nothing", "genre": "pop"},
]


def ids(query):
    return [s["id"] for s in search.search_songs(query, SONGS)]


def test_tokens_are_lowercase_unique_and_skip_common_words():
    assert search.tokenize("The Hello hello WITH a Day") == ["hello", "day"]


def test_a_query_of_only_common_words_keeps_them():
    assert search.tokenize("the") == ["the"]
    # The song called "The" comes first (the token is in its title); "the" is also a
    # substring of "The Beatles" and "Withers", which follow alphabetically.
    assert ids("the") == [4, 3, 2]


def test_substrings_match():
    assert ids("love") == [2]  # "love" finds "Lovely Day"


def test_tokens_match_in_any_order_across_title_artist_and_album():
    assert ids("adele hello")[0] == 1
    assert ids("hello adele")[0] == 1


def test_more_matching_tokens_rank_first_and_one_is_enough():
    # "hello goodbye" matches song 3 twice, song 1 once
    assert ids("hello goodbye") == [3, 1]


def test_no_match_gives_nothing():
    assert ids("zzzz") == []


def test_a_song_without_an_album_is_filed_under_its_genre():
    assert search.album_of(SONGS[2]) == "Rock"
    assert ids("rock") == [3]
