import json
import os
import random
import re
import shutil
from html import escape
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from play_queue import PlayQueue, next_repeat_mode
import agent_settings
import agent_tools
import llm_client
import playlists as plib
import search
import user_profile as uprofile
from album_art import ART_DIR, ART_URL_PREFIX, art_jpeg_bytes, resolve_audio_path
from music_agent import AgentJob, AgentResult, MusicAgent
from recommender import Recommender, Song, UserProfile, load_songs

SONGS_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


@st.cache_data
def get_songs():
    return load_songs(str(SONGS_PATH))


@st.cache_resource(show_spinner=False)
def get_recommender():
    """The library as a Recommender (see recommender.py), built once. It answers "which
    songs fit this UserProfile?", which is what the profile page asks."""
    return Recommender([Song.from_dict(s) for s in get_songs()])


@st.cache_data(show_spinner=False)
def get_album_art(audio_path_str, size=96):
    """Fallback for songs with no art_file in songs.csv: reads the art from the
    music share and returns it as a base64 data URI (or None)."""
    import base64

    data = art_jpeg_bytes(audio_path_str, size)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode() if data else None


NOW_DIR = ART_DIR.parent / "now"  # src/static/now: copies of the songs being played
NOW_URL_PREFIX = "/app/static/now/"
# Streamlit's static server only serves a fixed list of "safe" extensions, which
# may not include .m4a, so those copies are named .mp4 (same AAC-in-MP4 data,
# which the browser's audio player handles fine).
STATIC_AUDIO_EXT = {".m4a": ".mp4"}


def stage_song(song):
    """Copies a song into src/static/now/ and returns its URL, or None on failure.

    The browser's own audio player streams it from Streamlit's static file server,
    so Streamlit never owns the <audio> element (see PLAY_PAUSE_HTML). Only the
    few most recent copies are kept."""
    source = resolve_audio_path(song)
    if source is None:
        return None
    ext = source.suffix.lower()
    target = NOW_DIR / f"{song['id']}{STATIC_AUDIO_EXT.get(ext, ext)}"
    try:
        NOW_DIR.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            partial = target.with_name(target.name + ".part")
            shutil.copyfile(source, partial)
            partial.replace(target)
        target.touch()
        copies = sorted(NOW_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in copies[4:]:
            old.unlink(missing_ok=True)
    except OSError:  # the share dropped, or the disk is full
        return None
    return NOW_URL_PREFIX + target.name


ACTIVE_FILE = NOW_DIR.parent / "now_active.txt"  # id of the song that is current


def restore_active_song():
    """The song that was active when the app last ran: the id recorded in ACTIVE_FILE,
    or else the most recently used copy in src/static/now/. None if there is no such
    copy (or the song is no longer in the library)."""
    try:
        copies = [p for p in NOW_DIR.iterdir() if p.suffix.lower() in (".mp3", ".mp4", ".m4a", ".flac", ".wav", ".ogg")]
        wanted = ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else None
        if not any(p.stem == wanted for p in copies):
            newest = max(copies, key=lambda p: p.stat().st_mtime, default=None)
            wanted = newest.stem if newest else None
    except OSError:
        return None
    return next((s for s in get_songs() if str(s["id"]) == wanted), None) if wanted else None


def build_art_maps(songs):
    """Maps artist name, album name and song id to a thumbnail file name, using
    the art_file column filled in by scripts/populate_songs.py. No disk or
    network access: it only reads the rows already loaded from songs.csv."""
    by_artist, by_album, by_song = {}, {}, {}
    for s in songs:
        art = (s.get("art_file") or "").strip()
        if not art:
            continue
        by_song[str(s["id"])] = art
        by_artist.setdefault(s["artist"], art)
        by_album.setdefault(s.get("album") or s["genre"].title(), art)
    return {"artist": by_artist, "album": by_album, "song": by_song}


def get_queue():
    """The session's play queue, or None (also None for a queue left over from before the
    PlayQueue class existed, e.g. after the code reloaded mid-session)."""
    queue = st.session_state.get("queue")
    return queue if isinstance(queue, PlayQueue) else None


def start_queue(song_id, playlist=None):
    """A new play queue on song_id (see PlayQueue.start) that follows the player bar's
    shuffle and repeat settings, which stay as they are from one queue to the next."""
    return PlayQueue.start(
        song_id, get_songs(), playlist, resolve_audio_path,
        repeat=st.session_state.get("repeat", "autoplay"),
        shuffle=st.session_state.get("shuffle", False),
    )


def toggle_shuffle():
    """The shuffle button in the player bar: on, the album / playlist / mix that is playing
    continues in a random order; off, in its own order again."""
    st.session_state.shuffle = not st.session_state.get("shuffle", False)
    queue = get_queue()
    if queue:
        queue.set_shuffle(st.session_state.shuffle, get_songs(), resolve_audio_path)


def cycle_repeat():
    """The repeat button in the player bar: off, repeat album / playlist / mix, repeat song,
    autoplay (the default), and round again."""
    mode = next_repeat_mode(st.session_state.get("repeat", "autoplay"))
    st.session_state.repeat = mode
    queue = get_queue()
    if queue:
        queue.set_repeat(mode, get_songs(), resolve_audio_path)


def get_store():
    """The playlist file, with today's auto playlists created or refreshed first."""
    store = plib.load_store()
    if plib.refresh_auto_playlists(store, get_songs(), user_prefs=uprofile.load_profile()):
        plib.save_store(store)
    return store


def play_song(song_id, playlist_id=None):
    """Starts a new play queue at song_id (see play_queue.py).

    Alone, the song gets the 10 best-fitting recommendations queued after it. With
    playlist_id, all of that playlist's songs are queued (the playlist then counts as
    recently played) and the recommender extends the queue as it runs low."""
    store = plib.load_store() if playlist_id else None
    playlist = store["playlists"].get(playlist_id) if store else None
    queue = start_queue(song_id, playlist)
    if queue is None:
        return
    st.session_state.queue = queue
    st.session_state.now_playing = song_id
    # The player restarts a song whenever this changes, so clicking the song that is
    # already playing starts it over instead of doing nothing.
    st.session_state.play_token = st.session_state.get("play_token", 0) + 1
    if playlist:
        plib.mark_played(store, playlist_id)
        plib.save_store(store)


def play_playlist(playlist_id):
    """Starts a playlist from its first playable song."""
    playlist = plib.load_store()["playlists"].get(playlist_id)
    if not playlist or not playlist["song_ids"]:
        return
    by_id = {s["id"]: s for s in get_songs()}
    members = [by_id[i] for i in playlist["song_ids"] if i in by_id]
    if st.session_state.get("shuffle"):  # shuffled, it starts anywhere
        members_in_order = random.sample(members, len(members))
    else:
        members_in_order = members
    first = next((s for s in members_in_order if resolve_audio_path(s)), None)
    play_song((first or members[0])["id"], playlist_id)


def play_songs(members, source_id, start_id=None):
    """Plays a list of songs in order as the queue (an album, an artist's first album),
    from start_id if given (clicking a song in an album) or else from the first one that
    has a file. Like a playlist for the queue's purposes, but nothing is stored and it
    does not count as a recently played playlist."""
    first = next((s for s in members if s["id"] == start_id and resolve_audio_path(s)), None)
    if first is None:  # no song was picked: the first one, or with shuffle any of them
        order = random.sample(members, len(members)) if st.session_state.get("shuffle") else members
        first = next((s for s in order if resolve_audio_path(s)), None)
    if first is None:
        return
    collection = {"id": source_id, "song_ids": [s["id"] for s in members], "profile": None}
    queue = start_queue(first["id"], collection)
    if queue is None:
        return
    st.session_state.queue = queue
    st.session_state.now_playing = first["id"]
    st.session_state.play_token = st.session_state.get("play_token", 0) + 1


def album_key(song):
    return song.get("album") or song["genre"].title()


def album_track_order(song):
    return (_int(song.get("disc_number")), _int(song.get("track_number")), song["title"])


def play_album(name):
    """Double-click (or the bubble) on an album in the library: play it from track 1."""
    members = [s for s in get_songs() if album_key(s) == name]
    play_songs(sorted(members, key=album_track_order), f"album:{name}")


def play_artist(name):
    """Double-click (or the bubble) on an artist: play their first album, using the same
    order as the artist page (earliest release year first, then by name)."""
    albums = {}
    for s in get_songs():
        if s["artist"] == name:
            albums.setdefault(album_key(s), []).append(s)
    if not albums:
        return

    def oldest_first(item):
        years = [y for y in (_int(s.get("release_year")) for s in item[1]) if y]
        return (min(years) if years else 9999, item[0].lower())

    _album, members = min(albums.items(), key=oldest_first)
    play_songs(sorted(members, key=album_track_order), f"artist:{name}")


def open_view(kind, name, source=None, back=None):
    """Shows an artist, album or playlist in the main area.

    kind: "artist" | "album" | "playlist" (for a playlist, name is its id);
    source is unused. back: the view the Back button should
    return to (e.g. the artist page an album was opened from); None means home."""
    st.session_state.view = {"type": kind, "name": name, "source": source, "back": back}


def open_create():
    """The library's + button: open the Create a Playlist page with a fresh default name."""
    open_view("create", "new")
    st.session_state.name_reset = True


def open_player_view(kind, name):
    """The song title / artist in the player bar: open the queue or the artist page.
    Back returns to the page you were on (not to another queue/artist page opened the
    same way, so repeated clicks never build a chain)."""
    here = st.session_state.get("view", {"type": "home"})
    if kind == "queue":
        # A queue that is playing a saved playlist opens that playlist's own page.
        queue = get_queue()
        source = queue.source if queue else None
        if source and source in get_store()["playlists"]:
            kind, name = "playlist", source
    if here.get("type") == kind and here.get("name") == name:
        return  # already showing it
    back = here.get("back") if here.get("type") in ("queue", "artist", "playlist") else here
    open_view(kind, name, None, back)


def save_auto_playlist(playlist_id):
    """"Save playlist" on an auto-generated one: a copy of it becomes a playlist of your own,
    named "My <name>" (or "My <name> 2", 3, ... if that is taken), with the same songs in the same order and the same profile. The
    copy's page opens (it has the Edit pill); the auto mix itself carries on as before."""
    store = plib.load_store()
    auto = store["playlists"].get(playlist_id)
    if auto is None:
        return
    # Auto mixes change from day to day, so saving the same one twice keeps both, as
    # "My Work Mix" and "My Work Mix 2" (never two playlists with the same name)
    name = plib.unique_playlist_name(store, f"My {auto['name']}")
    new_id = plib.create_playlist(
        store, name, auto["song_ids"], get_songs(), profile=auto.get("profile")
    )
    plib.save_store(store)
    st.toast(f'Saved "{name}" to your playlists.')
    open_view("playlist", new_id, None, st.session_state.get("view", {}).get("back"))


def delete_playlist_and_leave(playlist_id):
    """"Delete" in a playlist's "..." menu: removes the playlist and goes back to the page
    before it (or home)."""
    store = plib.load_store()
    playlist = store["playlists"].get(playlist_id)
    plib.delete_playlist(store, playlist_id)
    plib.save_store(store)
    for key in (f"edit_draft_{playlist_id}", f"edit_name_{playlist_id}", f"edit_search_{playlist_id}"):
        st.session_state.pop(key, None)  # nothing left over from editing it
    if playlist:
        st.toast(f'Deleted "{playlist["name"]}".')
    go_back()


def regenerate_playlist(playlist_id):
    """"Regenerate" in an auto-generated playlist's "..." menu: a new set and order of
    songs; the page shows it straight away."""
    store = plib.load_store()
    if plib.regenerate_auto_playlist(store, playlist_id, get_songs()):
        plib.save_store(store)
        st.toast("Made a new mix.")


def open_edit(playlist_id):
    """The Edit pill on one of your playlists: open the editor, filled with that
    playlist's name and songs. Back (or saving) returns to the playlist's page."""
    playlist = plib.load_store()["playlists"].get(playlist_id)
    if playlist is None:
        return
    st.session_state[f"edit_draft_{playlist_id}"] = list(playlist["song_ids"])
    st.session_state[f"edit_name_{playlist_id}"] = playlist["name"]
    st.session_state[f"edit_search_{playlist_id}"] = ""
    here = st.session_state.get("view", {"type": "home"})
    open_view("edit", playlist_id, None, here)


def open_profile():
    """The gear button at the right of the top bar: open the user profile page. Back
    returns to the page you were on."""
    here = st.session_state.get("view", {"type": "home"})
    if here.get("type") != "profile":
        open_view("profile", "me", None, here)


def go_home():
    st.session_state.view = {"type": "home"}
    st.session_state.search = ""  # clear the top search box too


def run_search():
    """Top search box: pressing Enter shows the results page for what was typed."""
    query = st.session_state.get("search", "").strip()
    if query:
        open_view("search", query)


def clear_search(key="search"):
    """The x inside a search pill: empty that box (the current page stays as it is)."""
    st.session_state[key] = ""


def go_back():
    st.session_state.view = st.session_state.get("view", {}).get("back") or {"type": "home"}


def skip(delta):
    """Moves to the next (+1) or previous (-1) song in the play queue. Next also tops
    the queue back up to 10 songs (see play_queue.py); Previous at the very start
    restarts the current song."""
    queue = get_queue()
    current = st.session_state.get("now_playing")
    if current is None:
        return
    if not queue or queue.current_id != current:
        queue = start_queue(current)  # no queue yet
        st.session_state.queue = queue
        if queue is None:
            return
    target = queue.advance(delta, get_songs(), resolve_audio_path)
    if target is not None:
        st.session_state.now_playing = target
        st.session_state.play_token = st.session_state.get("play_token", 0) + 1




def create_song_mix(song_id):
    """"Create a mix with this song" in a song's "..." menu: a temporary, auto-generated
    mix built entirely from that song (see playlists.create_song_mix; the same artist is
    slightly favoured). It is added to the home page's recommended playlists and opens
    straight away; "Save playlist" on its page keeps a copy as a playlist of your own."""
    song = next((s for s in get_songs() if s["id"] == song_id), None)
    if song is None:
        return
    store = get_store()  # (today's auto mixes are in place, so the recommended list is complete)
    mix_id = plib.create_song_mix(store, song, get_songs())
    plib.save_store(store)
    st.toast(f'Made "{store["playlists"][mix_id]["name"]}". It is in Recommended on the home page for now.')
    here = st.session_state.get("view", {"type": "home"})
    open_view("playlist", mix_id, None, here)


def _song_by_id(song_id):
    return next((s for s in get_songs() if s["id"] == song_id), None)


def queue_song_next(song_id):
    """"Play Next" in a song's "..." menu: the song goes right after the one playing now.
    With nothing playing there is no queue to put it in, so it simply starts playing."""
    song = _song_by_id(song_id)
    queue = get_queue()
    if song is None:
        return
    if queue is None:
        play_song(song_id)
        return
    if queue.play_next(song_id):
        st.toast(f'"{song["title"]}" will play next.')


def queue_song_last(song_id):
    """"Add to Queue" in a song's "..." menu: the song goes at the very end of the queue.
    With nothing playing there is no queue, so it simply starts playing."""
    song = _song_by_id(song_id)
    queue = get_queue()
    if song is None:
        return
    if queue is None:
        play_song(song_id)
        return
    if queue.add(song_id):
        st.toast(f'Added "{song["title"]}" to the end of the queue.')


def remove_from_queue(index):
    """"Remove" in a queue row's "..." menu: takes that song out of the queue (the song
    playing now has no menu, so it cannot be removed)."""
    queue = get_queue()
    if queue:
        queue.remove(index, get_songs(), resolve_audio_path)


def jump_queue(index):
    """Clicking a row in the queue panel: play that song and carry on from there."""
    queue = get_queue()
    target = queue.jump(index, get_songs(), resolve_audio_path) if queue else None
    if target is not None:
        st.session_state.now_playing = target
        st.session_state.play_token = st.session_state.get("play_token", 0) + 1


WEB_DIR = Path(__file__).parent / "web"  # the page's HTML and JavaScript pieces


def web_fragment(name):
    """The markup in web/<name>.html (if any) followed by the script in web/<name>.js
    (if any, wrapped in <script> tags), ready for st.html(..., unsafe_allow_javascript=True)."""
    parts = []
    html_file, js_file = WEB_DIR / f"{name}.html", WEB_DIR / f"{name}.js"
    if html_file.exists():
        parts.append(html_file.read_text(encoding="utf-8").strip())
    if js_file.exists():
        parts.append("<script>\n" + js_file.read_text(encoding="utf-8") + "</script>")
    return "\n".join(parts)


st.set_page_config(page_title="Legally Distinct-ify", page_icon=":material/headphones:", layout="wide")

# The page styling lives in style.css, next to this file.
ICON_DIR = Path(__file__).parent / "assets" / "icons"  # the .svg icons


def icon_markup(name):
    """An icon's svg file as one line of markup, for putting straight into the page."""
    return " ".join((ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").split())


def inline_icons(css):
    """style.css points at icons as url("assets/icons/<name>.svg"); the browser gets each
    one embedded in the stylesheet as a data: URI (so there is nothing extra to serve)."""
    def embed(match):
        svg = icon_markup(match.group(1)).replace("> <", "><").replace("#", "%23")
        return f'url("data:image/svg+xml;utf8,{svg}")'
    return re.sub(r'url\("assets/icons/([\w-]+)\.svg"\)', embed, css)


APP_CSS = (
    "\n<style>\n"
    + inline_icons((Path(__file__).parent / "style.css").read_text(encoding="utf-8"))
    + "</style>\n"
)
# The style block, top bar, left panel and library-art script are all invisible or
# position: fixed, but each would still be a flow item that adds Streamlit's 1rem
# element gap above the page content. Group them in one zero-size fixed container
# (".st-key-chrome" in the CSS) so they add no space. It is closed again right
# after the library-art script, below.
chrome = st.container(key="chrome")
chrome.__enter__()
st.markdown(APP_CSS, unsafe_allow_html=True)

def playlist_gradient(playlist):
    """A playlist's colour: always the same gradient for the same playlist. It is used
    wherever a playlist needs a picture (home tiles, recent cells, the library panel,
    search results) instead of the album art of its first song."""
    return plib.generate_gradient(playlist["id"])  # see playlists.generate_gradient


# --- Top: Search Bar with label inside ---
with st.container(key="top_bar"):
    brand_col, home_col, search_col, profile_col = st.columns([1, 1, 2, 1], vertical_alignment="center")
    with brand_col:
        st.markdown(
            f'<div class="brand">{icon_markup("brand")}Legally Distinct<span>-ify</span></div>',
            unsafe_allow_html=True,
        )
        # Clicking the logo does the same as the home button beside it.
        st.html(web_fragment("brand_home"), unsafe_allow_javascript=True)
    with home_col:
        st.button(":material/home:", key="home_btn", on_click=go_home)
    with search_col:
        st.text_input(
            "What do you want to play?", key="search", placeholder="What do you want to play?",
            label_visibility="collapsed", on_change=run_search,
        )
        st.button(":material/close:", key="search_clear", on_click=clear_search, help="Clear search")
    with profile_col:  # the gear, in the far right corner
        st.button(":material/settings:", key="profile_btn", on_click=open_profile)  # hint: native_hints.js

# --- Left panel with Tabs (fixed between the top and bottom bars) ---
with st.container(key="left_panel"):
    with st.container(key="lib_header"):
        # Title on the left, the "+" (new playlist) at the right end of the same row. When
        # the panel is collapsed only the + stays, as a bubble pinned to the panel's top.
        lib_title_col, lib_new_col = st.columns([5, 1], vertical_alignment="center")
        lib_title_col.markdown("### :material/library_music: Your Library")
        lib_new_col.button(
            ":material/add:", key="lib_new", on_click=open_create,
        )  # its hover hint is set by LIB_ART_JS, like the bubbles' hints
        with st.container(key="lib_search_box"):
            lib_query = st.text_input(
                "Search your library",
                key="lib_search",
                placeholder="Search your library",
                label_visibility="collapsed",
            ).strip().lower()
            st.button(
                ":material/close:", key="lib_search_clear", on_click=clear_search,
                args=("lib_search",), help="Clear search",
            )

    # Create tabs
    tabs = st.tabs([":material/queue_music: Playlists", ":material/album: Albums", ":material/mic: Artists"])

    library_songs = get_songs()

    # Artists tab (third)
    with tabs[2], st.container(height=450, border=False, key="lib_list_artists"):
        artists = sorted(
            {song["artist"] for song in library_songs if lib_query in song["artist"].lower()}
        )
        for i, artist in enumerate(artists):
            st.button(
                artist, key=f"artist_btn{i}", use_container_width=True,
                on_click=open_view, args=("artist", artist),
            )
            # Hidden: clicked by the double-click / bubble handler in LIB_ART_JS
            st.button("Play", key=f"artist_play{i}", on_click=play_artist, args=(artist,))
        if not artists:
            st.caption("No matching artists.")
        st.markdown('<div class="list-spacer"></div>', unsafe_allow_html=True)

    # Albums tab: uses an "album" column if the CSV has one, otherwise
    # falls back to genres since the data has no album field yet.
    with tabs[1], st.container(height=450, border=False, key="lib_list_albums"):
        albums = sorted(
            {
                song.get("album") or song["genre"].title()
                for song in library_songs
                if lib_query in (song.get("album") or song["genre"]).lower()
            }
        )
        for i, album in enumerate(albums):
            st.button(
                album, key=f"album_btn{i}", use_container_width=True,
                on_click=open_view, args=("album", album),
            )
            st.button("Play", key=f"album_play{i}", on_click=play_album, args=(album,))
        if not albums:
            st.caption("No matching albums.")
        st.markdown('<div class="list-spacer"></div>', unsafe_allow_html=True)

    # Playlists tab (first): the playlists you created come first, then a thin divider,
    # then the automatic mixes (time of day, activity, genre), so the tab is never empty.
    lib_store = get_store()
    user_playlists = sorted(
        (p for p in lib_store["playlists"].values() if p["kind"] == "user"),
        key=lambda p: p["name"].lower(),
    )
    auto_list = (
        plib.auto_playlists(lib_store, "time")
        + plib.auto_playlists(lib_store, "activity")
        + plib.auto_playlists(lib_store, "genre")
    )
    lib_playlists = user_playlists + auto_list  # every playlist that can appear in the tab
    # Button keys use a plain number, never the playlist's id: ids such as
    # "auto-genre-r&b" or "auto-genre-easy listening" contain characters Streamlit
    # rewrites in CSS class names, which made the gradient rules miss those playlists.
    lib_index = {p["id"]: n for n, p in enumerate(lib_playlists)}
    with tabs[0], st.container(height=450, border=False, key="lib_list_playlists"):
        shown_user = [p for p in user_playlists if lib_query in p["name"].lower()]
        shown_auto = [p for p in auto_list if lib_query in p["name"].lower()]
        for group in (shown_user, shown_auto):
            if group is shown_auto and shown_user and shown_auto:
                st.markdown('<div class="lib-divider"></div>', unsafe_allow_html=True)
            for p in group:
                st.button(
                    p["name"], key=f"playlist_btn{lib_index[p['id']]}", use_container_width=True,
                    on_click=open_view, args=("playlist", p["id"]),
                )
                st.button(
                    "Play", key=f"playlist_play{lib_index[p['id']]}",
                    on_click=play_playlist, args=(p["id"],),
                )
        if not (shown_user or shown_auto):
            st.caption("No matching playlists.")
        st.markdown('<div class="list-spacer"></div>', unsafe_allow_html=True)

# Album art for the library items. Python reads nothing from disk or the music
# share here: songs.csv's art_file column already names each thumbnail (built once
# by scripts/populate_songs.py) and Streamlit serves them from src/static/art/.
# This script only watches which library buttons are on screen
# (IntersectionObserver) and sets --art for those, so the browser never even
# requests pictures for items you haven't scrolled to.
LIB_ART_JS = web_fragment("lib_art")
art_map_json = json.dumps(build_art_maps(library_songs)).replace("</", "<\\/")
# Playlists do not use album art in the library panel: each playlist's bubble is its own
# gradient (the same one as its home page tile), set through the --art variable that the
# bubble's background already reads. Plain CSS, so the art script has nothing to do for
# playlists (it finds no art for them and leaves --art alone).
st.markdown(
    "<style>"
    + "".join(
        # a class selector matches the whole class, so ".st-key-playlist_btn1" does not
        # also match "playlist_btn10"
        f'.st-key-playlist_btn{n} {{ --art: {playlist_gradient(p)}; }}'
        for n, p in enumerate(lib_playlists)
    )
    + "</style>",
    unsafe_allow_html=True,
)
st.html(
    LIB_ART_JS.replace("__ART_MAP__", art_map_json).replace("__ART_PREFIX__", ART_URL_PREFIX),
    unsafe_allow_javascript=True,
)

# The tab on the library panel's top right corner that collapses / expands it. It only
# flips the data-lib-collapsed attribute on <html> (all the visual changes are CSS, see
# "Collapsible library panel"), and remembers the choice for the next page load.
LIB_TOGGLE_HTML = web_fragment("lib_toggle")
st.html(LIB_TOGGLE_HTML, unsafe_allow_javascript=True)
chrome.__exit__(None, None, None)  # end of the "chrome" container opened above

# --- Main Content ---
# The main area has three states, kept in st.session_state.view:
#   home      the playlist grid and the playlist builder
#   album     one album with all its songs        (opened from the Albums tab)
#   playlist  one playlist with all its songs     (opened from a playlist card
#                                                  or a created playlist)
from datetime import datetime

MAX_TRACKS = 100  # rows shown per page in a playlist / queue view (a genre can hold thousands)


def page_bounds(total, default_page=0):
    """(start, end) row indexes of the current page of the open view. The page number is
    kept on the view itself, so opening another page of the site starts at the first page
    and coming Back returns to the page you were on."""
    view = st.session_state.get("view", {})
    pages = max(1, -(-total // MAX_TRACKS))
    page = min(max(view.get("page", default_page), 0), pages - 1)
    view["page"] = page
    return page * MAX_TRACKS, min((page + 1) * MAX_TRACKS, total)


def change_page(delta):
    view = st.session_state.view
    view["page"] = view.get("page", 0) + delta


def render_pager(total, where):
    """Previous / next buttons with "Page x of y" for lists longer than one page; nothing
    is drawn for a short list. where ("top" | "bottom") keeps the two copies' keys apart."""
    pages = -(-total // MAX_TRACKS)
    if pages <= 1:
        return
    page = st.session_state.view.get("page", 0)
    start, end = page * MAX_TRACKS + 1, min((page + 1) * MAX_TRACKS, total)
    prev_col, label_col, next_col = st.columns([1, 3, 1], vertical_alignment="center")
    prev_col.button(
        ":material/chevron_left: Previous", key=f"page_prev_{where}", disabled=page == 0,
        on_click=change_page, args=(-1,), use_container_width=True,
    )
    label_col.markdown(
        f'<div style="text-align:center;color:#b3b3b3;font-size:0.85rem">'
        f'Page {page + 1} of {pages} &bull; songs {start}&ndash;{end} of {total}</div>',
        unsafe_allow_html=True,
    )
    next_col.button(
        "Next :material/chevron_right:", key=f"page_next_{where}", disabled=page >= pages - 1,
        on_click=change_page, args=(1,), use_container_width=True,
    )
MAX_SEARCH_ROWS = 8  # artists / albums / playlists shown per section of the results
MAX_SEARCH_SONGS = 25  # songs shown in the results
MAX_TOP_RESULTS = 5  # best matches from any category, shown first


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_duration(seconds):
    seconds = _int(seconds, -1)
    return "" if seconds < 0 else f"{seconds // 60}:{seconds % 60:02d}"


def fmt_total(seconds):
    minutes = round(seconds / 60)
    return f"{minutes // 60} hr {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


def song_art_url(song):
    art = (song.get("art_file") or "").strip()
    return ART_URL_PREFIX + art if art else None


def art_cover_html(songs):
    """Large cover: the first song's album art, or a gray '?' if none has any."""
    art = next((url for url in map(song_art_url, songs) if url), None)
    if art:
        return f'<img class="view-cover" src="{art}" alt="">'
    return '<div class="view-cover view-cover-empty">?</div>'


def render_view_header(
    kind, title, meta, cover_html, first_song_id, playlist_id=None, collection=None,
    editable=False, savable=False,
):
    st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
    st.markdown(
        f'<div class="view-head">{cover_html}<div class="view-info">'
        f'<div class="view-kind">{kind}</div>'
        f'<div class="view-title">{escape(title)}</div>'
        f'<div class="view-meta">{meta}</div></div></div>',
        unsafe_allow_html=True,
    )
    # Play, and next to it a grey pill: "Edit" for a playlist you made, "Save playlist" for
    # an auto-generated one (which copies it into your own playlists)
    # (kept in a container of its own: its CSS stops these buttons from shrinking, so their
    # text is never abbreviated, however narrow the page gets when a side panel is open)
    with st.container(key="view_actions"):
        play_col, edit_col, more_col, _rest = st.columns([1.3, 1.7, 0.8, 5.8], vertical_alignment="center")
        if first_song_id is not None:
            if collection is not None:  # an album: queue all of it
                play = {"on_click": play_songs, "args": tuple(collection)}
            elif playlist_id:
                play = {"on_click": play_playlist, "args": (playlist_id,)}
            else:
                play = {"on_click": play_song, "args": (first_song_id,)}
            play_col.button(":material/play_arrow: Play", key="vplay", type="primary", **play)
        if editable and playlist_id:
            edit_col.button("Edit", key="vedit", on_click=open_edit, args=(playlist_id,))
        elif savable and playlist_id:
            edit_col.button(
                "Save playlist", key="vsave", on_click=save_auto_playlist, args=(playlist_id,),
                help="Keep a copy of this mix in your own playlists",
            )
        # The "..." menu: Delete for a playlist you made, Regenerate for an auto-generated one
        if playlist_id and (editable or savable):
            with more_col.popover("⋯", key="vmore"):  # its hint is set by native_hints.js
                if editable:
                    st.button(
                        "Delete", key="vdelete", on_click=delete_playlist_and_leave,
                        args=(playlist_id,),
                    )
                else:
                    st.button(
                        "Regenerate", key="vregen", on_click=regenerate_playlist,
                        args=(playlist_id,),
                        help="Build a new set and order of songs from this mix's profile",
                    )
    if first_song_id is None:
        st.caption("None of these songs has a playable file. Is the music share mounted?")


def render_track_list(
    rows, show_album, playlist_id=None, queue_positions=None, header=True, tag=None,
    collection=None,
):
    """rows is a list of (song, note); note is small text under the title or None.
    playlist_id: when the rows are a playlist, playing a row queues that playlist.
    collection: (songs, source_id) when the rows are an album: playing a row queues the
    whole album from that song, like playlist_id does for a playlist.
    queue_positions: when the rows ARE the play queue (one queue index per row), playing
    a row jumps to it in the queue instead of starting a new one."""
    def play_action(row_number, song):
        if queue_positions is not None:
            return {"on_click": jump_queue, "args": (queue_positions[row_number],)}
        if collection is not None:  # e.g. an album: queue all of it, starting at this song
            return {"on_click": play_songs, "args": (*collection, song["id"])}
        return {"on_click": play_song, "args": (song["id"], playlist_id)}

    playing_id = st.session_state.get("now_playing")
    # Same card styling as the album rows on the artist page (the "arow..." /
    # "ahead..." / "abtn..." key prefixes pick up that CSS), but a song needs a
    # DOUBLE click anywhere on its card to play; the invisible button covering the
    # card ignores single clicks (see the click listener in PLAY_PAUSE_HTML). The
    # small play button between the number and the title plays with one click.
    widths = [0.8, 0.7, 4.5, 3, 3, 1] if show_album else [0.8, 0.7, 5.5, 4, 1]
    labels = ["" if show_album else "#", "", "Title", "Artist"] + (
        ["Album"] if show_album else []
    ) + ["Time"]
    widths, labels = widths + [0.7], labels + [""]  # one more column, for the "..." menu
    if header:  # (off for a single row shown among other kinds of result)
        with st.container(key="ahead_tracks"):
            for col, label in zip(st.columns(widths, vertical_alignment="center"), labels):
                col.markdown(f'<div class="track-head">{label}</div>', unsafe_allow_html=True)
    queue = get_queue() if queue_positions is not None else None
    for position, (song, note) in enumerate(rows, 1):
        action = play_action(position - 1, song)
        if queue_positions is not None:
            # In the queue a song can appear more than once (repeat), so a row is told apart
            # by its place in the queue: the key ends with it, and only the entry that is
            # actually playing counts as "playing", not every copy of the song.
            index = queue_positions[position - 1]
            sfx = f"_{index}"
            playing = queue is not None and index == queue.pos
        else:
            sfx = ""
            playing = song["id"] == playing_id
        # Every track row (album, playlist and search results) uses the thin "arow_trk"
        # style; see the CSS next to ".st-key-arow".
        with st.container(key=f"arow_trk{song['id']}{sfx}"):
            cols = st.columns(widths, vertical_alignment="center")
            if show_album:  # playlists mix albums, so show each song's art
                art = song_art_url(song)
                cols[0].markdown(
                    f'<img class="alb-thumb" src="{art}" alt="">' if art
                    else '<div class="alb-thumb alb-thumb-empty">?</div>',
                    unsafe_allow_html=True,
                )
            else:  # one album: show the track number
                number = _int(song.get("track_number")) or position
                cols[0].markdown(
                    f'<div class="{"track-title playing" if playing else "track-sub"}">{number}</div>',
                    unsafe_allow_html=True,
                )
            # The current song's button gets a different key ("vplay_cur") so the
            # player script can turn it into a play/pause toggle for that song.
            cols[1].button(
                ":material/play_arrow:",
                key=f"vplay_{'cur' if playing else 'row'}{song['id']}{sfx}", **action,
            )
            note_html = f'<div class="track-sub">{escape(note)}</div>' if note else ""
            cols[2].markdown(
                f'<div class="track-title{" playing" if playing else ""}">'
                f'{escape(song["title"])}{result_tag(tag)}</div>{note_html}',
                unsafe_allow_html=True,
            )
            # Artist and album names are links: one click opens that page, and Back
            # returns to the page you are on. They sit above the card-covering button.
            here = st.session_state.get("view", {"type": "home"})
            cols[3].button(
                song["artist"], key=f"vlink_ar{song['id']}{sfx}",
                on_click=open_view, args=("artist", song["artist"], None, here),
            )
            if show_album:
                album = song.get("album") or song["genre"].title()
                cols[4].button(
                    album, key=f"vlink_al{song['id']}{sfx}",
                    on_click=open_view, args=("album", album, None, here),
                )
            cols[-2].markdown(
                f'<div class="track-sub">{fmt_duration(song.get("duration_seconds"))}</div>',
                unsafe_allow_html=True,
            )
            with cols[-1]:  # the "..." menu, far right
                if queue_positions is not None and not playing:  # queued songs can be removed
                    song_menu(f"qvmore_{song['id']}{sfx}", song["id"], f"qrm_v{index}", index)
                else:
                    song_menu(f"qvmore_{song['id']}{sfx}", song["id"], playing=playing)
            st.button(
                song["title"], key=f"abtn_song{song['id']}{sfx}", **action,
            )


def render_album_view(name):
    songs = [s for s in library_songs if (s.get("album") or s["genre"].title()) == name]
    if not songs:
        st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
        st.warning(f'Album "{name}" was not found.')
        return
    songs.sort(key=lambda s: (_int(s.get("disc_number")), _int(s.get("track_number")), s["title"]))
    artists = sorted({s["artist"] for s in songs})
    meta = [escape(artists[0] if len(artists) == 1 else "Various artists")]
    years = [y for y in (_int(s.get("release_year")) for s in songs) if y]
    if years:
        meta.append(str(min(years)))
    meta.append(f"{len(songs)} songs")
    total = sum(_int(s.get("duration_seconds")) for s in songs)
    if total:
        meta.append(fmt_total(total))
    first = next((s for s in songs if resolve_audio_path(s)), None)
    album_queue = (songs, f"album:{name}")  # playing any track queues the rest of the album
    render_view_header(
        "Album", name, " &bull; ".join(meta), art_cover_html(songs), first["id"] if first else None,
        collection=album_queue,
    )
    render_track_list([(s, None) for s in songs], show_album=False, collection=album_queue)


def render_artist_view(name):
    songs = [s for s in library_songs if s["artist"] == name]
    if not songs:
        st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
        st.warning(f'Artist "{name}" was not found.')
        return
    albums = {}
    for s in songs:
        albums.setdefault(s.get("album") or s["genre"].title(), []).append(s)

    def oldest_first(item):  # albums by earliest release year, then by name
        years = [y for y in (_int(s.get("release_year")) for s in item[1]) if y]
        return (min(years) if years else 9999, item[0].lower())

    ordered = sorted(albums.items(), key=oldest_first)
    meta = [f"{len(albums)} albums", f"{len(songs)} songs"]
    total = sum(_int(s.get("duration_seconds")) for s in songs)
    if total:
        meta.append(fmt_total(total))
    first = None
    for _album, album_songs in ordered:
        album_songs.sort(key=lambda s: (_int(s.get("disc_number")), _int(s.get("track_number"))))
        first = first or next((s for s in album_songs if resolve_audio_path(s)), None)
    render_view_header(
        "Artist", name, " &bull; ".join(meta), art_cover_html(songs), first["id"] if first else None
    )

    # Each album is one rectangular card holding the whole row of information. An
    # invisible button stretched over the card (see ".st-key-abtn..." in the CSS)
    # makes the entire card clickable.
    widths = [0.8, 6, 2, 2]
    with st.container(key="ahead"):
        for col, label in zip(
            st.columns(widths, vertical_alignment="center"), ["", "Album", "Year", "Songs"]
        ):
            col.markdown(f'<div class="track-head">{label}</div>', unsafe_allow_html=True)
    for i, (album, album_songs) in enumerate(ordered):
        with st.container(key=f"arow{i}"):
            cols = st.columns(widths, vertical_alignment="center")
            art = next((url for url in map(song_art_url, album_songs) if url), None)
            cols[0].markdown(
                f'<img class="alb-thumb" src="{art}" alt="">' if art
                else '<div class="alb-thumb alb-thumb-empty">?</div>',
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f'<div class="track-title">{escape(album)}</div>', unsafe_allow_html=True
            )
            years = [y for y in (_int(s.get("release_year")) for s in album_songs) if y]
            cols[2].markdown(
                f'<div class="track-sub">{min(years) if years else ""}</div>',
                unsafe_allow_html=True,
            )
            cols[3].markdown(
                f'<div class="track-sub">{len(album_songs)}</div>', unsafe_allow_html=True
            )
            st.button(
                album, key=f"abtn{i}", on_click=open_view,
                # Back from the album returns to this artist page
                args=("album", album, None, {"type": "artist", "name": name, "source": None, "back": None}),
            )


def result_tag(tag):
    """A small pill saying what kind of result a row is (used in the mixed Top results)."""
    return f'<span class="result-tag">{escape(tag)}</span>' if tag else ""


def nav_row(key, songs, title, extras, open_args, widths, gradient=None, tag=None):
    """One clickable result card (thumbnail, title, extra columns). The whole card
    opens a page: an invisible button covers it ("abtn..." CSS, like the artist page).
    gradient: a CSS gradient to use as the thumbnail instead of album art (playlists)."""
    with st.container(key=f"arow_nav{key}"):
        cols = st.columns(widths, vertical_alignment="center")
        art = next((url for url in map(song_art_url, songs) if url), None)
        if gradient:
            thumb = f'<div class="alb-thumb" style="background:{gradient}"></div>'
        elif art:
            thumb = f'<img class="alb-thumb" src="{art}" alt="">'
        else:
            thumb = '<div class="alb-thumb alb-thumb-empty">?</div>'
        cols[0].markdown(thumb, unsafe_allow_html=True)
        cols[1].markdown(
            f'<div class="track-title">{escape(title)}{result_tag(tag)}</div>',
            unsafe_allow_html=True,
        )
        for col, text in zip(cols[2:], extras):
            col.markdown(f'<div class="track-sub">{escape(str(text))}</div>', unsafe_allow_html=True)
        st.button(title, key=f"abtn_nav{key}", on_click=open_view, args=open_args)


def render_search_view(query):
    """Results for the top search box, grouped: playlists, artists, albums, songs."""
    st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
    st.markdown(
        f'<div class="view-title" style="font-size:2rem">Results for &ldquo;{escape(query)}&rdquo;</div>',
        unsafe_allow_html=True,
    )
    # Matching (tokens, scores) is shared with the Create a Playlist song search: search.py.
    tokens = search.tokenize(query)
    # Results link back to this page, so Back from an artist/album returns here
    here = {"type": "search", "name": query, "source": None, "back": None}
    widths = [0.8, 6, 2, 2]

    album_of = search.album_of

    def hits(*fields):
        return search.count_hits(tokens, *fields)

    # Playlists: saved ones and the current auto mixes, matched on the name
    pl_scored = [(hits(p["name"]), p) for p in get_store()["playlists"].values()]
    pl_scored = sorted((x for x in pl_scored if x[0]), key=lambda x: -x[0])
    # Artists, matched on the artist name
    ar_scored = [(hits(a), a) for a in {s["artist"] for s in library_songs}]
    ar_scored = sorted((x for x in ar_scored if x[0]), key=lambda x: (-x[0], x[1].lower()))
    # Albums, matched on the album name
    al_scored = [(hits(a), a) for a in {album_of(s) for s in library_songs}]
    al_scored = sorted((x for x in al_scored if x[0]), key=lambda x: (-x[0], x[1].lower()))
    # Songs: matched on title, artist and album together
    so_scored = search.rank_songs(tokens, library_songs)

    if not (pl_scored or ar_scored or al_scored or so_scored):
        st.info("No playlists, artists, albums or songs match your search.")
        return

    by_id = {s["id"]: s for s in library_songs}

    def playlist_row(key, p, tag=None):
        members = [by_id[sid] for sid in p["song_ids"] if sid in by_id]
        kind = "Auto mix" if p["kind"] == "auto" else "Playlist"
        nav_row(
            key, members, p["name"], [kind, f"{len(members)} songs"],
            ("playlist", p["id"], None, here), widths, gradient=playlist_gradient(p), tag=tag,
        )

    def artist_row(key, artist, tag=None):
        theirs = [s for s in library_songs if s["artist"] == artist]
        n_albums = len({album_of(s) for s in theirs})
        nav_row(
            key, theirs, artist, [f"{n_albums} albums", f"{len(theirs)} songs"],
            ("artist", artist, None, here), widths, tag=tag,
        )

    def album_row(key, album, tag=None):
        tracks = [s for s in library_songs if album_of(s) == album]
        who = sorted({s["artist"] for s in tracks})
        years = [y for y in (_int(s.get("release_year")) for s in tracks) if y]
        nav_row(
            key, tracks, album,
            [who[0] if len(who) == 1 else "Various artists", min(years) if years else ""],
            ("album", album, None, here), widths, tag=tag,
        )

    # Top results: the best scoring matches from every category together. At equal score an
    # artist comes first, then an album, a playlist and a song. Whatever is shown here is
    # left out of its own section below.
    candidates = (
        [(n, 2, "playlist", p) for n, p in pl_scored]
        + [(n, 0, "artist", a) for n, a in ar_scored]
        + [(n, 1, "album", a) for n, a in al_scored]
        + [(n, 3, "song", s) for n, _t, s in so_scored]
    )
    # (sorted() is stable, so equal score and priority keep the order built above)
    top = sorted(candidates, key=lambda c: (-c[0], c[1]))[:MAX_TOP_RESULTS]
    if top:
        st.subheader("Top results")
        for i, (_n, _pri, kind, item) in enumerate(top):
            # each row carries a tag saying what it is, since the section mixes kinds
            if kind == "playlist":
                playlist_row(f"_tp{i}", item, tag="playlist")
            elif kind == "artist":
                artist_row(f"_tp{i}", item, tag="artist")
            elif kind == "album":
                album_row(f"_tp{i}", item, tag="album")
            else:
                render_track_list([(item, None)], show_album=True, header=False, tag="song")
    shown = {(kind, item["id"] if kind in ("playlist", "song") else item) for _n, _p, kind, item in top}
    playlists = [p for _n, p in pl_scored if ("playlist", p["id"]) not in shown]
    artists = [a for _n, a in ar_scored if ("artist", a) not in shown]
    albums = [a for _n, a in al_scored if ("album", a) not in shown]
    songs = [s for _n, _t, s in so_scored if ("song", s["id"]) not in shown]

    if playlists:
        st.subheader("Playlists")
        for i, p in enumerate(playlists[:MAX_SEARCH_ROWS]):
            playlist_row(f"_pl{i}", p)

    if artists:
        st.subheader("Artists")
        for i, artist in enumerate(artists[:MAX_SEARCH_ROWS]):
            artist_row(f"_ar{i}", artist)
        if len(artists) > MAX_SEARCH_ROWS:
            st.caption(f"Showing the first {MAX_SEARCH_ROWS} of {len(artists)} artists.")

    if albums:
        st.subheader("Albums")
        for i, album in enumerate(albums[:MAX_SEARCH_ROWS]):
            album_row(f"_al{i}", album)
        if len(albums) > MAX_SEARCH_ROWS:
            st.caption(f"Showing the first {MAX_SEARCH_ROWS} of {len(albums)} albums.")

    if songs:
        st.subheader("Songs")
        render_track_list([(s, None) for s in songs[:MAX_SEARCH_SONGS]], show_album=True)
        if len(songs) > MAX_SEARCH_SONGS:
            st.caption(f"Showing the first {MAX_SEARCH_SONGS} of {len(songs)} songs.")


def render_playlist_view(view):
    """One stored playlist (view["name"] is its id): its songs in their saved order,
    each with how well it fits the playlist's recommender profile."""
    playlist = get_store()["playlists"].get(view["name"])
    if playlist is None:
        st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
        st.warning("This playlist no longer exists. Auto mixes are temporary and refresh daily.")
        return
    by_id = {s["id"]: s for s in library_songs}
    songs = [by_id[i] for i in playlist["song_ids"] if i in by_id]
    rows = [(s, None) for s in songs]
    meta = [f"{len(songs)} songs"]
    total = sum(_int(s.get("duration_seconds")) for s in songs)
    if total:
        meta.append(fmt_total(total))
    if playlist["kind"] == "auto":
        meta.append("refreshes daily")
    first = next((s for s in songs if resolve_audio_path(s)), None)
    render_view_header(
        "Auto mix" if playlist["kind"] == "auto" else "Playlist",
        playlist["name"], " &bull; ".join(meta), playlist_cover(playlist),
        first["id"] if first else None, playlist_id=playlist["id"],
        editable=playlist["kind"] == "user", savable=playlist["kind"] == "auto",
    )
    st.caption("When the last song ends, similar songs keep playing.")
    start, end = page_bounds(len(rows))
    render_pager(len(rows), "top")
    render_track_list(rows[start:end], show_album=True, playlist_id=playlist["id"])
    render_pager(len(rows), "bottom")


def render_queue_view():
    """The current play queue as a playlist page. Opened from the song title in the player
    bar. The queue is often not a saved playlist (a single song plus recommendations), so
    the page says where it comes from."""
    queue = get_queue()
    by_id = {s["id"]: s for s in library_songs}
    entries = [(n, by_id[i]) for n, i in enumerate(queue.songs) if i in by_id] if queue else []
    if not entries:
        st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
        st.info("Nothing is queued. Play a song or a playlist to start.")
        return
    playlist = get_store()["playlists"].get(queue.source)
    if playlist:
        note = f'This queue is playing from the playlist "{escape(playlist["name"])}".'
    else:
        note = (
            "This queue is not a saved playlist."
        )
    songs = [s for _n, s in entries]
    total = sum(_int(s.get("duration_seconds")) for s in songs)
    meta = [f"{len(songs)} songs"] + ([fmt_total(total)] if total else [])
    st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
    st.markdown(
        f'<div class="view-head">{art_cover_html(songs)}<div class="view-info">'
        f'<div class="view-kind">Play queue</div>'
        f'<div class="view-title">Current queue</div>'
        f'<div class="view-meta">{" &bull; ".join(meta)}</div></div></div>'
        f'<div class="queue-note">{note}</div>',
        unsafe_allow_html=True,
    )
    pos = queue.pos
    rows = [
        (s, "Now playing" if n == pos else "Played" if n < pos else None) for n, s in entries
    ]
    # Open on the page that holds the song that is playing now.
    start, end = page_bounds(len(rows), default_page=next((i for i, (n, _s) in enumerate(entries) if n == pos), 0) // MAX_TRACKS)
    render_pager(len(rows), "top")
    render_track_list(
        rows[start:end], show_album=True, queue_positions=[n for n, _s in entries[start:end]]
    )
    render_pager(len(rows), "bottom")


def playlist_cover(playlist):
    return (
        f'<div class="view-cover view-cover-grad" style="background:{playlist_gradient(playlist)}">'
        f'{escape(playlist["name"])}</div>'
    )


def tile_keys(row, n):
    """The widget keys for tile number n of a row of playlist tiles. Every tile on the
    home page is drawn by render_tile_row and named only through this function, so all
    of them get exactly the same styling.

    The CSS finds tiles by matching part of a key, so one row's name must never be the
    start of another's (a row "rec" would also match "recent"). The trailing "_" after
    the row name guarantees that, and row names are limited to lower-case letters."""
    assert row.isalpha() and row.islower(), f"tile row name must be lower-case letters: {row!r}"
    return {
        "card": f"pcard_{row}_{n}",
        "open": f"abtn_pcard_{row}_{n}",
        "play": f"pplay_{row}_{n}",
    }


def render_tile_row(prefix, items, badge=None):
    """One horizontally scrolling row of playlist cards (see "hrow" in the CSS). Every
    row of tiles on the home page (Recent, Activity, Genre) goes through here, so they
    all look and behave the same. prefix: the row's name (see tile_keys).
    badge: id of a playlist to label "Now" (the current time of day)."""
    if not items:
        return
    by_id = {s["id"]: s for s in library_songs}
    with st.container(key=f"hrow_{prefix}"):
        for i, (col, p) in enumerate(zip(st.columns(len(items)), items)):
            members = [by_id[sid] for sid in p["song_ids"] if sid in by_id]
            # No play counts are stored, so "most popular" means the artists with the
            # most songs in the mix (ties keep the order they first appear in).
            counts = {}
            for s in members:
                counts[s["artist"]] = counts.get(s["artist"], 0) + 1
            top_artists = sorted(counts, key=lambda a: -counts[a])[:3]
            now_tag = '<span class="pl-now">Now</span>' if p["id"] == badge else ""
            # Each card is a container so the invisible open button and the small play
            # button can be positioned over it (see "pcard" in the CSS).
            keys = tile_keys(prefix, i)
            with col, st.container(key=keys["card"]):
                st.markdown(
                    f'<div class="pl-card"><div class="pl-cover" '
                    f'style="background:{playlist_gradient(p)}">{now_tag}{escape(p["name"])}</div>'
                    f'<div class="pl-artists">{escape(", ".join(top_artists))}</div></div>',
                    unsafe_allow_html=True,
                )
                # Clicking anywhere on the card opens the playlist (invisible button
                # covering the card), and the green button in the corner plays it.
                st.button(
                    p["name"], key=keys["open"],
                    on_click=open_view, args=("playlist", p["id"]),
                )
                st.button(
                    ":material/play_arrow:", key=keys["play"],
                    on_click=play_playlist, args=(p["id"],),
                )


def render_recommended(playlists_to_show):
    """The recommended playlists: 3 columns by 2 rows of short cells, each with a mini
    icon on the left and the title on the right (see plib.recommended_playlists)."""
    st.subheader("Recommended Playlists")
    for row_start in range(0, len(playlists_to_show), 3):
        for col, (i, p) in zip(
            st.columns(3), list(enumerate(playlists_to_show))[row_start:row_start + 3]
        ):
            # The playlist's own gradient, the same as on its home page tile
            icon = f'<div class="alb-thumb" style="background:{playlist_gradient(p)}"></div>'
            # These cells have their own key names (arow_cell / abtn_cell / cplay), kept
            # apart from the tiles' (see the comment above "pcard" in the CSS)
            with col, st.container(key=f"arow_cell{i}"):
                st.markdown(
                    f'<div class="rec-cell">{icon}'
                    f'<div class="track-title">{escape(p["name"])}</div></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    p["name"], key=f"abtn_cell{i}", on_click=open_view, args=("playlist", p["id"]),
                )
                # Fades in on the right of the cell on hover (see "cplay" in the CSS)
                st.button(
                    ":material/play_arrow:", key=f"cplay{i}",
                    on_click=play_playlist, args=(p["id"],),
                )


def render_home():
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(f'<div class="greeting">{greeting}</div>', unsafe_allow_html=True)

    store = get_store()
    # Recommended (the grid of short cells): this time of day's mix, the activity mix that
    # suits it, then the playlists you created most recently to fill the 6 places
    render_recommended(plib.recommended_playlists(store, datetime.now()))

    # Recent (a scrolling row of tiles): the playlists you played most recently
    st.subheader("Recent Playlists")
    recent = plib.recent_playlists(store, 6)
    if recent:
        render_tile_row("recent", recent)
    else:
        st.caption("Playlists you play show up here.")

    st.subheader("Activity Mixes")
    render_tile_row("act", plib.auto_playlists(store, "activity"))

    st.subheader("Genre Mixes")
    render_tile_row("genre", plib.auto_playlists(store, "genre"))
    # (The Create a Playlist section is no longer on this page: the + button at the top of
    # the library panel opens it, see render_playlist_editor.)


# Clicking the playlist name box selects its text when it is a suggested "Playlist N", so the
# first thing typed replaces it (the suggestion "clears itself"). A name you typed
# yourself is left alone. The mouseup guard stops the click from undoing the selection.
CREATE_NAME_JS = web_fragment("create_name")


def clearable_text_input(label, key, **kwargs):
    """A text box with an x at its right end that empties it (shown only while there is
    text; see "clearbox_" in style.css). Takes st.text_input's arguments, and needs a
    placeholder so the CSS can tell when the box is empty."""
    with st.container(key=f"clearbox_{key}"):
        value = st.text_input(label, key=key, **kwargs)
        st.button(
            ":material/close:", key=f"clearbtn_{key}", on_click=clear_search, args=(key,),
            help="Clear",
        )
    return value


def add_to_draft(song_id, draft_key):
    """The + on a song search result of the create / edit page."""
    draft = st.session_state.setdefault(draft_key, [])
    if song_id not in draft:
        draft.append(song_id)


def remove_from_draft(song_id, draft_key):
    draft = st.session_state.setdefault(draft_key, [])
    if song_id in draft:
        draft.remove(song_id)


def render_playlist_editor(playlist_id=None):
    """The Create a Playlist page (opened by the + button in the library panel) and, with
    a playlist_id, the Edit page of one of your playlists (opened by its Edit pill). Both
    have a name, a song search whose results have a + to add them, and the songs picked so
    far."""
    store = get_store()
    st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
    editing = store["playlists"].get(playlist_id) if playlist_id else None
    if playlist_id and (editing is None or editing["kind"] != "user"):
        st.warning("This playlist no longer exists.")
        return
    st.title("Edit Playlist" if editing else "Create a Playlist")

    songs = get_songs()
    by_id = {s["id"]: s for s in songs}
    # Create and edit keep their own state, so one never shows up in the other
    suffix = f"_{playlist_id}" if editing else ""
    draft_key = f"edit_draft{suffix}" if editing else "create_draft"
    name_key = f"edit_name{suffix}" if editing else "new_playlist_name"
    search_key = f"edit_search{suffix}" if editing else "create_search"
    draft = st.session_state.setdefault(draft_key, list(editing["song_ids"]) if editing else [])

    # Editing: what is already saved, and what is still waiting to be saved. The two
    # section buttons (name, songs) each save only their own part; "Save All Changes" saves
    # everything.
    saved_ids = editing["song_ids"] if editing else []
    pending_adds = [i for i in draft if i not in saved_ids]
    pending_removals = [i for i in saved_ids if i not in draft]

    def save_section(message, name_to_save, song_ids):
        """Writes one part of the edit, remembers a message, and redraws the page (it stays
        open, now showing what has been saved)."""
        plib.update_playlist(store, playlist_id, name_to_save, song_ids, songs)
        plib.save_store(store)
        st.session_state.edit_flash = message
        st.rerun()

    if editing:
        flash = st.session_state.pop("edit_flash", None)
        if flash:
            st.success(flash)
        st.session_state.setdefault(name_key, editing["name"])
        name = clearable_text_input("Playlist name", name_key, placeholder="Name your playlist")
        # Section 1: the name. Saves only the name (the songs keep their saved state).
        if st.button(
            "Save name", key="save_name",
            disabled=not name.strip() or name.strip() == editing["name"],
        ):
            save_section("Saved the new name.", name, saved_ids)
    else:
        # The name box starts filled with the next free "Playlist N". It is set through
        # session state before the box is drawn (also again after a playlist is created, so
        # the box moves on to the next number); CREATE_NAME_JS selects it when clicked, so
        # typing replaces the suggestion.
        suggested = plib.next_playlist_name(store)
        if st.session_state.pop("name_reset", False) or name_key not in st.session_state:
            st.session_state[name_key] = suggested
        flash = st.session_state.pop("create_flash", None)
        if flash:
            st.success(flash)
        name = clearable_text_input("Playlist name", name_key, placeholder="Name your playlist")
        st.html(CREATE_NAME_JS, unsafe_allow_javascript=True)

    # Song search: the same matching as the top search bar (search.py). The + at the right
    # of a result adds that song to the playlist.
    st.subheader("Add songs")
    query = clearable_text_input(
        "Search for songs to add", search_key, placeholder="Search by title, artist or album",
    ).strip()
    if query:
        found = search.search_songs(query, songs)
        if not found:
            st.info("No songs match your search.")
        for song in found[:MAX_SEARCH_SONGS]:
            added = song["id"] in draft
            with st.container(key=f"arow_trkcr{song['id']}"):
                thumb, text, album, plus = st.columns([0.8, 6, 3, 1], vertical_alignment="center")
                art = song_art_url(song)
                thumb.markdown(
                    f'<img class="alb-thumb" src="{art}" alt="">' if art
                    else '<div class="alb-thumb alb-thumb-empty">?</div>',
                    unsafe_allow_html=True,
                )
                text.markdown(
                    f'<div class="track-title">{escape(song["title"])}</div>'
                    f'<div class="track-sub">{escape(song["artist"])}</div>',
                    unsafe_allow_html=True,
                )
                album.markdown(
                    f'<div class="track-sub">{escape(search.album_of(song))}</div>',
                    unsafe_allow_html=True,
                )
                plus.button(
                    ":material/check:" if added else ":material/add:", key=f"cadd_{song['id']}",
                    on_click=add_to_draft, args=(song["id"], draft_key), disabled=added,
                    help="Already in the playlist" if added else "Add to playlist",
                )
        if len(found) > MAX_SEARCH_SONGS:
            st.caption(f"Showing the first {MAX_SEARCH_SONGS} of {len(found)} songs. Narrow the search to see others.")

    # The songs added so far, in the order they were added
    st.subheader(f"In this playlist ({len(draft)})")
    if not draft:
        st.caption("Search for a song above and press + to add it.")
    for song_id in draft:
        song = by_id.get(song_id)
        if song is None:
            continue
        with st.container(key=f"arow_trkdr{song_id}"):
            text, remove = st.columns([9, 1], vertical_alignment="center")
            text.markdown(
                f'<div class="track-title">{escape(song["title"])}</div>'
                f'<div class="track-sub">{escape(song["artist"])}</div>',
                unsafe_allow_html=True,
            )
            remove.button(
                ":material/close:", key=f"cdel_{song_id}", on_click=remove_from_draft,
                args=(song_id, draft_key), help="Remove from playlist",
            )
    # Section 2: the songs, added and removed. Saves only the songs (the name keeps its
    # saved state).
    if editing and st.button(
        "Save songs", key="save_songs", disabled=not (pending_adds or pending_removals)
    ):
        if draft:
            save_section("Saved the songs.", editing["name"], list(draft))
        else:
            st.error("A playlist needs at least one song.")
    submitted = st.button("Save All Changes" if editing else "Create playlist", type="primary")

    if submitted:
        if not name.strip():
            st.error("Please give your playlist a name.")
        elif not draft:
            st.error("Add at least one song first.")
        elif editing:
            plib.update_playlist(store, playlist_id, name, list(draft), songs)
            plib.save_store(store)
            for key in (draft_key, name_key, search_key):
                st.session_state.pop(key, None)
            go_back()  # to the playlist's page, now with its new name and songs
            st.rerun()
        else:
            # Saved to data/playlists.json: the songs in the order they were added; the
            # playlist's recommender profile (used to keep playing after the last song)
            # is worked out from those songs.
            plib.create_playlist(store, name, list(draft), songs)
            plib.save_store(store)
            count = len(draft)
            draft.clear()
            # Start over with the next default name: the name box can only be changed
            # before it is drawn, so remember the message and run the page again
            st.session_state.create_flash = f'Created "{name.strip()}" with {count} songs.'
            st.session_state.name_reset = True
            st.rerun()


def send_chat():
    """Called when a message is submitted (Enter, or the send button). Empty or
    whitespace-only messages are ignored. The message is shown at once and queued for the
    agent (see start_next_agent_job), so the bubble does not wait for the model; a message
    sent while the agent is still busy waits its turn. Messages live in the session only."""
    text = (st.session_state.get("chat_text") or "").strip()
    if not text:
        return
    st.session_state.setdefault("chat_messages", []).append(
        {"role": "user", "text": text}
    )
    st.session_state.setdefault("chat_pending", []).append(text)


def reset_chat():
    """The refresh button in the chat header: throws away the conversation so far (what is
    shown, and what the agent remembers of it). An agent still working is told to stop
    at its next step; what it has already done stays done."""
    job = st.session_state.pop("agent_job", None)
    if job is not None:
        job.cancel()
    st.session_state.chat_messages = []
    st.session_state.agent_history = []
    st.session_state.chat_pending = []


def agent_context():
    """What the agent's tools work on: the library, the play queue and the playlists, the
    same ones the buttons in the app use."""
    return agent_tools.AgentContext(
        songs=get_songs(),
        get_queue=get_queue,
        playable=resolve_audio_path,
        load_store=get_store,
        save_store=plib.save_store,
        start_playback=play_song,
        start_album=play_album,
        start_playlist=play_playlist,
    )


def start_next_agent_job():
    """If the agent is idle and a chat message is waiting, starts working on it.

    The agent runs on a background thread (see music_agent.AgentJob). Streamlit re-runs the
    whole page on every click, which would cut an agent running inside the page off half
    way (say, when you pick a song while it thinks); a thread carries on regardless. It is
    given the page's session context so the tools can use the play queue and the like.

    Only the agent's final reply is ever shown in the chat. Everything else (the model's
    tool calls and the tools' results) goes into st.session_state.agent_history, which the
    agent reads to remember the conversation but the chat panel never draws."""
    pending = st.session_state.get("chat_pending") or []
    if st.session_state.get("agent_job") is not None or not pending:
        return
    text = pending.pop(0)
    history = list(st.session_state.get("agent_history", []))
    context = agent_context()
    choice = agent_settings.load_settings()["client"]

    def work(cancelled):
        try:
            client = llm_client.make_client(choice)
        except RuntimeError as problem:
            label = llm_client.CLIENT_CHOICES[choice]
            return AgentResult(
                f"I can't use {label} right now: {problem}. You can pick another one in Settings.",
                [], ok=False,
            )
        return MusicAgent(client, context).run(text, history, cancelled)

    ctx = get_script_run_ctx()
    st.session_state.agent_job = AgentJob(work).start(
        attach=lambda thread: add_script_run_ctx(thread, ctx)
    )


def finish_agent_job():
    """Takes the finished agent's reply into the chat. Returns whether there was one."""
    job = st.session_state.get("agent_job")
    if job is None or not job.done:
        return False
    st.session_state.agent_job = None
    result = job.result
    if job.cancelled() or not result.reply:
        return False  # the chat was reset while it worked: nothing to show
    if result.ok:  # a failed request leaves no trace in what the agent remembers
        st.session_state.agent_history = st.session_state.get("agent_history", []) + result.messages
    st.session_state.setdefault("chat_messages", []).append({"role": "assistant", "text": result.reply})
    return True


def chat_html(messages):
    """The whole conversation as one block of HTML: user messages are light green
    bubbles on the right, system messages white bubbles on the left (roles other than
    "user" are drawn as system messages). The last element carries the message count,
    which CHAT_SCROLL_JS watches to scroll down when a message is added."""
    parts = []
    for m in messages:
        role = "user" if m["role"] == "user" else "system"
        parts.append(
            f'<div class="chat-row {role}"><div class="chat-col">'
            f'<div class="chat-bubble">{escape(m["text"])}</div></div></div>'
        )
    parts.append(f'<div class="chat-end" data-count="{len(messages)}"></div>')
    return "".join(parts)


def draw_chat_conversation():
    """The conversation and, while the agent works, a "Thinking..." line. When the agent
    has finished, its reply is added and the whole page is drawn again, since the agent
    may have changed the queue or the playlists."""
    if finish_agent_job():
        st.rerun()  # the whole page, not just this fragment
    messages = st.session_state.get("chat_messages", [])
    if messages:
        st.html(chat_html(messages))
    else:
        st.caption("No messages yet.")
    if st.session_state.get("agent_job") is not None:
        st.caption("Thinking...")


@st.fragment(run_every=1.0)
def chat_conversation_polling():
    """The conversation while the agent works: this fragment re-runs by itself every
    second to look for the agent's reply. Clicks elsewhere re-run the page but do not
    stop the agent (it is on its own thread) or this timer."""
    draw_chat_conversation()


@st.fragment
def chat_conversation_idle():
    """The conversation when no agent is working: nothing to check on."""
    draw_chat_conversation()


def render_chat_panel():
    """Right-hand chat panel: the conversation from the top (oldest first), and a
    message box at the bottom. Enter sends, Shift+Enter starts a new line. The agent's
    replies are the white bubbles on the left; its tool calls and reasoning are not shown.

    It is built on every run and shown or hidden by CSS (see the right-hand panels CSS),
    which is what makes opening it instant and keeps a half-typed message."""
    start_next_agent_job()
    with st.container(key="chat_panel"):
        title_col, new_col = st.columns([5, 1], vertical_alignment="center")
        title_col.markdown("### :material/chat_bubble: Chat")
        new_col.button(
            ":material/refresh:", key="chat_new", on_click=reset_chat, help="Start new chat"
        )
        with st.container(height=400, border=False, key="chat_list"):
            if st.session_state.get("agent_job") is not None:
                chat_conversation_polling()
            else:
                chat_conversation_idle()
        st.chat_input("Type a message", key="chat_text", on_submit=send_chat)


# Keeps the chat list scrolled to the newest message: when the panel opens and
# whenever the message count changes, not on unrelated page updates.
CHAT_SCROLL_JS = web_fragment("chat_scroll")


# The chat and queue buttons in the player bar. A click sets (or clears) data-right-panel
# on <html>; the CSS does the rest, so there is no Streamlit rerun and no delay. Clicking
# the open panel's button closes it, and clicking the other one switches to it.
PANEL_TOGGLE_JS = web_fragment("panel_toggle")

MAX_QUEUE_ROWS = 100  # up-next rows drawn in the queue panel


def song_menu(menu_key, song_id, remove_key=None, remove_index=None, playing=False):
    """The "..." button at the right of a song. Its menu offers "Play Next" (right after
    the current song), "Add to Queue" (at the very end), "Create a mix with this song" and,
    for a queued song (give remove_key and remove_index), a red Remove that takes it out of
    the queue. The song playing now is already playing, so it has no Play Next or Add to
    Queue."""
    with st.popover("⋯", key=menu_key):
        if not playing:
            st.button(
                "Play Next", key=f"qnext_{menu_key}", on_click=queue_song_next, args=(song_id,),
            )
            st.button(
                "Add to Queue", key=f"qadd_{menu_key}", on_click=queue_song_last, args=(song_id,),
            )
        st.button(
            "Create a mix with this song", key=f"qmix_{menu_key}",
            on_click=create_song_mix, args=(song_id,),
        )
        if remove_key is not None:
            st.button("Remove", key=remove_key, on_click=remove_from_queue, args=(remove_index,))


def render_queue_panel():
    """Right-hand panel: the song playing now, then everything up next, in a
    scrollable list. Clicking a row plays that song."""
    queue = get_queue()
    by_id = {s["id"]: s for s in library_songs}

    def row(index, song, current=False, played=False):
        art = song_art_url(song)
        icon = (
            f'<img class="alb-thumb" src="{art}" alt="">' if art
            else '<div class="alb-thumb alb-thumb-empty">?</div>'
        )
        with st.container(key=f"arow_q{index}"):
            st.markdown(
                f'<div class="q-cell{" q-dim" if played else ""}">{icon}<div class="q-text">'
                f'<div class="track-title{" playing" if current else ""}">{escape(song["title"])}</div>'
                f'<div class="track-sub">{escape(song["artist"])}</div></div></div>',
                unsafe_allow_html=True,
            )
            st.button(
                song["title"], key=f"abtn_q{index}", on_click=jump_queue, args=(index,),
            )
            # the "..." menu at the far right (CSS places it, see qpmore_); the playing
            # song cannot be removed or queued, so its menu only has the mix
            song_menu(
                f"qpmore_{index}", song["id"],
                None if current else f"qrm_p{index}", index, playing=current,
            )

    with st.container(key="queue_panel"):
        st.markdown("### :material/queue_music: Queue")
        with st.container(height=400, border=False, key="queue_list"):
            if not queue or queue.current_id not in by_id:
                st.caption("Nothing is queued. Play a song or a playlist to start.")
                return
            # Songs already played sit above "Now playing" (scroll up to reach them and
            # click one to go back). The script below scrolls the list so "Now playing"
            # is at the top whenever the panel opens or the song changes.
            history = [
                (n, by_id[sid]) for n, sid in enumerate(queue.history)
                if sid in by_id
            ]
            if history:
                st.markdown('<div class="track-head">History</div>', unsafe_allow_html=True)
                for index, song in history:
                    row(index, song, played=True)
            marker = f'{queue.current_id}-{st.session_state.get("play_token", 0)}'
            gap = "margin-top:1rem" if history else ""
            # With history, a small "Queue history" hint sits just above "Now playing" and
            # the list is scrolled to it, so people can see there is more above.
            hint = (
                '<div style="text-align:center;color:#b3b3b3;font-size:0.7rem;'
                'letter-spacing:0.05em;margin-bottom:0.4rem">'
                '&uarr; Queue history &uarr;</div>'
            ) if history else ""
            st.html(
                f'<div class="q-now-label" data-now="{marker}" style="{gap}">{hint}'
                f'<div class="track-head">Now playing</div></div>'
            )
            row(queue.pos, by_id[queue.current_id], current=True)
            upcoming = [
                (queue.pos + 1 + n, by_id[sid])
                for n, sid in enumerate(queue.up_next) if sid in by_id
            ]
            st.markdown(
                f'<div class="track-head" style="margin-top:1rem">Next up '
                f'&bull; {len(upcoming)} songs</div>',
                unsafe_allow_html=True,
            )
            for index, song in upcoming[:MAX_QUEUE_ROWS]:
                row(index, song, current=False)
            if len(upcoming) > MAX_QUEUE_ROWS:
                st.caption(f"Showing the next {MAX_QUEUE_ROWS} of {len(upcoming)} songs.")
            st.markdown('<div class="list-spacer" style="height:24px"></div>', unsafe_allow_html=True)


# Keeps the queue list scrolled so "Now playing" is at the top (the history above it
# is a scroll up away). It only re-scrolls when the panel has just opened or the
# playing song has changed (the label's data-now changes), so scrolling around the
# list is not undone by unrelated page updates.
QUEUE_SCROLL_JS = web_fragment("queue_scroll")


PROFILE_ACOUSTIC_CHOICES = ["Any", "Acoustic", "Not acoustic"]  # likes_acoustic: None / True / False
PROFILE_SLIDERS = [  # (key, label, low, high, step, hint) for the "no preference" sliders
    ("energy", "Energy", 0.0, 1.0, 0.01, "0 is calm and quiet, 1 is loud and intense."),
    ("valence", "Positivity", 0.0, 1.0, 0.01, "0 is dark and sad, 1 is bright and cheerful."),
    ("tempo_bpm", "Tempo (beats per minute)", uprofile.TEMPO_MIN, uprofile.TEMPO_MAX, 1, None),
]


def clear_user_profile():
    """"Clear my preferences": forgets the saved profile and puts the page back to
    "no preference" everywhere."""
    uprofile.clear_profile()
    for key in [k for k in st.session_state if str(k).startswith("prof_")]:
        del st.session_state[key]


def render_profile_view():
    """The user profile page, opened by the gear in the top bar: the listener's taste,
    which the recommender uses for the "Made for You" mix on the home page. Every
    preference can be left as "no preference" (Any, or the checkbox beside a slider); those
    features are then ignored instead of counting as a middle value."""
    st.button(":material/arrow_back: Back", key="view_back", on_click=go_back)
    st.title("Settings")
    st.header("Your Profile")
    saved = uprofile.load_profile()  # None: never saved; {}: saved with no preference at all
    current = saved or {}
    st.caption(
        "Tell us what you like. Leave anything you don't mind as \"Any\" or tick \"No "
        "preference\": it is then ignored. Your preferences shape the Made for You mix on the "
        "home page."
    )

    songs = get_songs()
    genres = ["Any"] + sorted({s["genre"] for s in songs})
    moods = ["Any"] + sorted({s["mood"] for s in songs})

    # The widgets start from what is saved (everything on "no preference" if nothing is).
    # Streamlit forgets a widget's state when its page is left, so this runs on every visit.
    state = st.session_state
    state.setdefault("prof_genre", current.get("genre", "Any"))
    state.setdefault("prof_mood", current.get("mood", "Any"))
    for key, _label, _low, _high, _step, _hint in PROFILE_SLIDERS:
        state.setdefault(f"prof_{key}_any", key not in current)
        state.setdefault(f"prof_{key}", type(uprofile.DEFAULTS[key])(current.get(key, uprofile.DEFAULTS[key])))
    state.setdefault(
        "prof_acoustic",
        PROFILE_ACOUSTIC_CHOICES[{None: 0, True: 1, False: 2}[current.get("likes_acoustic")]],
    )

    left, right = st.columns(2)
    left.selectbox(
        "Favourite genre", genres, key="prof_genre",
        format_func=lambda g: g if g == "Any" else g.title(),
    )
    right.selectbox(
        "Favourite mood", moods, key="prof_mood",
        format_func=lambda m: m if m == "Any" else m.title(),
    )
    for key, label, low, high, step, hint in PROFILE_SLIDERS:
        slider_col, any_col = st.columns([8, 3], vertical_alignment="bottom")
        any_col.checkbox("No preference", key=f"prof_{key}_any")
        slider_col.slider(
            label, low, high, step=step, key=f"prof_{key}", help=hint,
            disabled=state[f"prof_{key}_any"],
        )
    st.radio("Acoustic sound", PROFILE_ACOUSTIC_CHOICES, key="prof_acoustic", horizontal=True)

    if st.button("Save preferences", type="primary"):
        prefs = {
            "genre": None if state.prof_genre == "Any" else state.prof_genre,
            "mood": None if state.prof_mood == "Any" else state.prof_mood,
            "likes_acoustic": {"Any": None, "Acoustic": True, "Not acoustic": False}[state.prof_acoustic],
        }
        for key, *_rest in PROFILE_SLIDERS:
            prefs[key] = None if state[f"prof_{key}_any"] else state[f"prof_{key}"]
        saved = uprofile.save_profile(prefs)
        if saved:
            st.success("Saved. Your Made for You mix on the home page now follows these preferences.")
        else:
            st.info("Saved, with no preference for anything, so there is no Made for You mix.")

    if saved is not None:
        st.button("Clear my preferences", key="profile_clear", on_click=clear_user_profile)
    if saved:
        st.subheader("Your top matches right now")
        picks = get_recommender().recommend(UserProfile.from_prefs(saved), k=5)
        by_id = {s["id"]: s for s in songs}  # the page's rows need the full library entries
        render_track_list([(by_id[pick.id], None) for pick in picks], show_album=True)

    render_agent_settings()


def render_agent_settings():
    """"Agent Settings": which LLM the chat agent uses. Picking one in the list changes
    nothing by itself: the chat keeps using the saved model until "Apply & Save" is
    pressed, and from then on (from the next chat message) uses the new one."""
    st.header("Agent Settings")
    saved = agent_settings.load_settings()["client"]
    st.session_state.setdefault("agent_client", saved)
    choice = st.selectbox(
        "Language model", list(llm_client.CLIENT_CHOICES), key="agent_client",
        format_func=llm_client.CLIENT_CHOICES.get,
        help="The model behind the chat. It changes only when you press Apply & Save.",
    )
    if choice == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        st.warning("GEMINI_API_KEY is not set, so Gemini cannot be used yet.")
    if st.button("Apply & Save", key="agent_apply", type="primary", disabled=choice == saved):
        agent_settings.save_settings({"client": choice})
        st.success(f"Saved. The chat now uses {llm_client.CLIENT_CHOICES[choice]}.")
    else:
        st.caption(f"The chat is using: {llm_client.CLIENT_CHOICES[saved]}")


view = st.session_state.get("view", {"type": "home"})
if view["type"] == "album":
    render_album_view(view["name"])
elif view["type"] == "artist":
    render_artist_view(view["name"])
elif view["type"] == "search":
    render_search_view(view["name"])
elif view["type"] == "playlist":
    render_playlist_view(view)
elif view["type"] == "queue":
    render_queue_view()
elif view["type"] == "profile":
    render_profile_view()
elif view["type"] == "create":
    render_playlist_editor()
elif view["type"] == "edit":
    render_playlist_editor(view["name"])
else:
    render_home()

# --- Pinned player bar (CSS keeps it fixed to the bottom of the window) ---
# Rendered with st.html(unsafe_allow_javascript=True), i.e. directly in the page
# (no iframe). The handlers are installed once on the document and look the
# <audio> element up at event time, so they survive Streamlit re-rendering it.
# Volume slider, shown at the right edge of the bar. It reuses the .np-scrub
# slider styling; the scrubber script below finds it by id and drives the audio.
VOLUME_HTML = web_fragment("volume")

# Clicking the song title / artist in the player bar clicks the matching hidden button.
NOW_LINKS_HTML = web_fragment("now_links")

PLAY_PAUSE_HTML = web_fragment("play_pause")

# Custom seek bar + time + volume. Like the play button, it is plain page HTML
# whose handlers are installed once and find the <audio> element at event time.
SCRUBBER_HTML = web_fragment("scrubber")

if "now_playing" not in st.session_state:
    # First load of the session: get the song that was active last time ready (paused),
    # so pressing play works straight away.
    restored = restore_active_song()
    if restored:
        st.session_state.now_playing = restored["id"]
        st.session_state.queue = start_queue(restored["id"])

now_song = next(
    (s for s in get_songs() if s["id"] == st.session_state.get("now_playing")), None
)
with st.container(key="player_bar"):
    # One compact row: song info | (transport buttons over seek/volume)
    info_col, center_col, chat_col, queue_col, volume_col = st.columns(
        [2, 5, 1, 1, 2], vertical_alignment="center"
    )
    with info_col:
        if now_song:
            # Prefer the prebuilt thumbnail named in songs.csv (no disk/share read);
            # fall back to reading the song's own art for songs not built yet.
            if (now_song.get("art_file") or "").strip():
                art_uri = ART_URL_PREFIX + now_song["art_file"].strip()
            else:
                song_file = resolve_audio_path(now_song)
                art_uri = get_album_art(str(song_file)) if song_file else None
            art_html = (
                f'<img class="np-art" src="{art_uri}" alt="">' if art_uri
                else '<div class="np-art np-art-empty">?</div>'
            )
            st.markdown(
                f'<div class="np-info">{art_html}<div class="np-text">'
                f'<div class="np-title" title="Show the play queue">{now_song["title"]}</div>'
                f'<div class="np-artist" title="Open the artist page">{now_song["artist"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            # Zero-size, invisible buttons that the title / artist above click.
            with st.container(key="np_hidden"):
                st.button("queue", key="npl_title", on_click=open_player_view, args=("queue", "queue"))
                st.button(
                    "artist", key="npl_artist", on_click=open_player_view,
                    args=("artist", now_song["artist"]),
                )
                st.html(NOW_LINKS_HTML, unsafe_allow_javascript=True)
        else:
            st.markdown('<div class="np-artist">Nothing playing</div>', unsafe_allow_html=True)
    # Center column: transport buttons stacked above the seek bar
    with center_col:
        with st.container(key="player_controls"):
            shuffle_col, prev_col, play_col, next_col, repeat_col = st.columns(5, vertical_alignment="center")
            # The icons and their colours come from the key (CSS: "Shuffle and repeat" in
            # style.css), so each state has its own key; the hints are set by native_hints.js.
            shuffle_col.button(
                "Shuffle", key=f"shuffle_{'on' if st.session_state.get('shuffle') else 'off'}",
                on_click=toggle_shuffle,
            )
            prev_col.button(":material/skip_previous:", key="prev_btn", on_click=skip, args=(-1,))
            with play_col:
                st.html(PLAY_PAUSE_HTML, width="content", unsafe_allow_javascript=True)
            next_col.button(":material/skip_next:", key="next_btn", on_click=skip, args=(1,))
            repeat_col.button(
                "Repeat", key=f"repeat_{st.session_state.get('repeat', 'autoplay')}",
                on_click=cycle_repeat,
            )
        st.html(SCRUBBER_HTML, unsafe_allow_javascript=True)
        # There is no st.audio element any more: the browser's own Audio object (see
        # PLAY_PAUSE_HTML) plays the song, so reruns cannot interrupt it. The hidden marker
        # that tells it which file to play is filled in at the very END of the script (see
        # "audio_slot" below): copying a song's file from the share can take seconds, and
        # nothing should have to wait for that, least of all the queue panel.
        audio_slot = st.empty()
    # Right side: the chat and queue buttons, then the volume, pushed to the right edge of
    # the bar. The buttons are plain HTML (see PANEL_TOGGLE_JS), not Streamlit buttons.
    with chat_col:
        st.html('<button id="chat-toggle" title="Show or hide the chat" aria-label="Chat"></button>')
    with queue_col:
        st.html('<button id="queue-toggle" title="Show or hide the queue" aria-label="Queue"></button>')
    with volume_col:
        st.html(VOLUME_HTML)

# Both right-hand panels are built on every run and shown/hidden by CSS, so opening one
# does not need a rerun (instant) and nothing typed or scrolled in them is lost.
render_queue_panel()
render_chat_panel()
st.html(PANEL_TOGGLE_JS, unsafe_allow_javascript=True)
st.html(QUEUE_SCROLL_JS, unsafe_allow_javascript=True)
st.html(CHAT_SCROLL_JS, unsafe_allow_javascript=True)
st.html(web_fragment("native_hints"), unsafe_allow_javascript=True)
st.html(web_fragment("popover_autoclose"), unsafe_allow_javascript=True)

# --- Slow part, last: get the audio files ready ---
# Everything above has already been sent to the browser. The current song's file is copied
# into src/static/now/ once, when the song starts (not on every rerun), and the next song's
# file right after it, so the browser can download it into memory (see prefetchNext in
# PLAY_PAUSE_HTML).
audio_url = None
if now_song:
    remembered = st.session_state.get("audio_src")
    if remembered and remembered["id"] == now_song["id"]:
        audio_url = remembered["url"]
    else:
        audio_url = stage_song(now_song)
        if audio_url:
            st.session_state.audio_src = {"id": now_song["id"], "url": audio_url}
            try:
                ACTIVE_FILE.write_text(str(now_song["id"]))
            except OSError:
                pass
with audio_slot.container():
    if audio_url:
        st.html(
            f'<div id="notify-audio-src" hidden data-src="{audio_url}" '
            f'data-token="{st.session_state.get("play_token", 0)}" '
            f'data-autoplay="{1 if st.session_state.get("play_token", 0) else 0}"></div>'
        )
    elif now_song:
        st.caption("No playable file found. Check file_path in songs.csv and that the share is mounted.")
    queue = get_queue()
    if audio_url and queue and queue.current_id == now_song["id"]:
        by_id = {s["id"]: s for s in library_songs}
        next_song = next(
            (by_id[i] for i in queue.up_next if i in by_id and resolve_audio_path(by_id[i])),
            None,
        )
        if next_song:
            prepared = st.session_state.get("next_src")
            if prepared and prepared["id"] == next_song["id"]:
                next_url = prepared["url"]
            else:
                next_url = stage_song(next_song)
                if next_url:
                    st.session_state.next_src = {"id": next_song["id"], "url": next_url}
            if next_url:
                st.html(f'<div id="notify-next-src" hidden data-src="{next_url}"></div>')

