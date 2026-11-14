"""How text searches match songs (no Streamlit in here). Used by the top search bar and
by the song search on the Create a Playlist page, so both find things the same way.

A query is split into words ("tokens") that may match in any order, each as a substring
("love" finds "lovely"). A result's score is how many different tokens it matches.
Tokens that are common words ("the", "a", "with") are filtered out so they don't match
nearly everything; a query made only of such tokens (a song called "The", say) keeps them.
"""

from typing import Dict, List, Tuple

SEARCH_STOPWORDS = frozenset(
    "a an the and or but of in on at to for from with by as is it its be this that "
    "these those i me my you your we our he she they them his her their".split()
)


def tokenize(query: str) -> List[str]:
    """The query's search tokens: lowercase, no repeats, common words filtered out."""
    all_tokens = list(dict.fromkeys(query.lower().split()))
    return [t for t in all_tokens if t not in SEARCH_STOPWORDS] or all_tokens


def count_hits(tokens: List[str], *fields: str) -> int:
    """How many different tokens appear (as substrings) in the fields together."""
    text = " ".join(fields).lower()
    return sum(1 for t in tokens if t in text)


def album_of(song: Dict) -> str:
    """A song's album; a song without one is filed under its genre."""
    return song.get("album") or song["genre"].title()


def rank_songs(tokens: List[str], songs: List[Dict]) -> List[Tuple[int, int, Dict]]:
    """Songs matching at least one token as (score, title_hits, song), best first. Songs
    are matched on title, artist and album together (so "adele hello" finds Hello by
    Adele); ties go to the song with more tokens in its title, then alphabetical."""
    scored = [
        (count_hits(tokens, s["title"], s["artist"], album_of(s)), count_hits(tokens, s["title"]), s)
        for s in songs
    ]
    return sorted(
        (x for x in scored if x[0]), key=lambda x: (-x[0], -x[1], x[2]["title"].lower())
    )


def search_songs(query: str, songs: List[Dict]) -> List[Dict]:
    """The songs matching the query, best match first."""
    return [song for _score, _title_hits, song in rank_songs(tokenize(query), songs)]
