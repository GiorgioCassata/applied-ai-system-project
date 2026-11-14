(function () {
  if (window.__notifyPlayerBound) return;
  window.__notifyPlayerBound = true;
  // The audio player is a plain Audio object held on window, NOT an element that
  // Streamlit renders. Streamlit reruns (opening an album, clicking anywhere) can
  // re-render or replace its own elements, which used to stop the song and reset it
  // to 0:00; they cannot touch this object. Python only says which file should be
  // playing, through the hidden #notify-audio-src element below.
  const audio = (window.__notifyAudio = window.__notifyAudio || new Audio());
  const getAudio = () => audio;
  // The next song in the queue is downloaded into memory ahead of time (Python names
  // it in the hidden #notify-next-src element). Keyed by the file's URL, holding a
  // blob: URL; only the current and the next song are kept.
  const blobs = (window.__notifyBlobs = window.__notifyBlobs || {});
  const fetching = new Set();
  const prefetchNext = () => {
    const holder = document.getElementById("notify-next-src");
    const url = holder && holder.dataset.src;
    if (!url || blobs[url] || fetching.has(url)) return;
    fetching.add(url);
    fetch(url)
      .then((r) => (r.ok ? r.blob() : Promise.reject()))
      .then((blob) => {
        blobs[url] = URL.createObjectURL(blob);
        Object.keys(blobs).forEach((k) => {
          if (k !== url && k !== window.__notifyAudioUrl) {
            URL.revokeObjectURL(blobs[k]);
            delete blobs[k];
          }
        });
      })
      .catch(() => {})
      .finally(() => fetching.delete(url));
  };
  const loadCurrent = () => {
    const holder = document.getElementById("notify-audio-src");
    const url = holder && holder.dataset.src;
    if (!url) return;
    const key = url + "|" + holder.dataset.token;  // token changes on every play click
    if (key === window.__notifyAudioKey) return;
    window.__notifyAudioKey = key;
    if (url === window.__notifyAdvanced) {
      // The song ended and the browser already started this (preloaded) next song
      // without waiting for Python; Python has now caught up. Don't restart it.
      window.__notifyAdvanced = null;
      if (audio.paused) audio.play().catch(() => {});
      return;
    }
    if (window.__notifyAudioUrl !== url) {
      window.__notifyAudioUrl = url;
      audio.src = blobs[url] || url;  // preloaded copy if there is one: starts instantly
    } else {
      audio.currentTime = 0;  // same song clicked again: start it over
    }
    // autoplay is 0 for the song restored at startup: it is loaded but waits for Play.
    if (holder.dataset.autoplay !== "0") audio.play().catch(() => {});
  };
  setInterval(() => { loadCurrent(); prefetchNext(); }, 200);
  const syncIcon = () => {
    const state = audio && !audio.paused ? "playing" : "paused";
    // The play button on the current song's row mirrors the main control. Its icon
    // is a font ligature (the element's text is the icon name), so find the element
    // inside the button whose text is "play_arrow" or "pause" and swap that text.
    const iconName = state === "playing" ? "pause" : "play_arrow";
    document.querySelectorAll('[class*="st-key-vplay_cur"] button *').forEach((el) => {
      if (el.children.length !== 0 || !el.firstChild) return;
      const text = el.firstChild.nodeValue;
      if ((text === "play_arrow" || text === "pause") && text !== iconName) {
        el.firstChild.nodeValue = iconName;
      }
    });
    const btn = document.getElementById("notify-pp");
    if (!btn || btn.dataset.state === state) return;
    btn.dataset.state = state;  // CSS draws the play or pause icon from this
  };
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#notify-pp")) return;
    const audio = getAudio();
    if (audio) audio.paused ? audio.play() : audio.pause();
  }, true);
  // Picking another song (library list, playlist Play buttons, previous/next):
  // stop the current one right away in the browser, instead of letting it play on
  // until the server has rerun and the new song has loaded.
  document.addEventListener("click", (e) => {
    // The play button on the row of the song that is already current acts like the
    // main play/pause control: toggle in the browser, no rerun, no restart.
    if (e.target.closest('[class*="st-key-vplay_cur"]')) {
      e.stopPropagation();
      e.preventDefault();
      audio.paused ? audio.play().catch(() => {}) : audio.pause();
      return;
    }
    // Track rows play on a DOUBLE click: swallow the first click of a pair (detail
    // is the click count) before Streamlit sees it, so it neither plays nor pauses.
    if (e.target.closest('[class*="st-key-abtn_song"]') && e.detail < 2) {
      e.stopPropagation();
      e.preventDefault();
      return;
    }
    if (!e.target.closest(
      '[class*="st-key-song_btn"], [class*="st-key-btn"], [class*="st-key-abtn_song"], ' +
      '[class*="st-key-vplay"], [class*="st-key-pplay"], [class*="st-key-cplay"], ' +
      '[class*="st-key-abtn_q"], ' +
      '[class*="st-key-artist_play"], [class*="st-key-album_play"], [class*="st-key-playlist_play"], ' +
      '.st-key-prev_btn, .st-key-next_btn'
    )) return;
    if (window.__notifyKeepPlaying) {  // our own automatic "next" click after a song ended
      window.__notifyKeepPlaying = false;
      return;
    }
    const audio = getAudio();
    if (audio) audio.pause();
  }, true);
  ["play", "pause", "ended", "emptied", "loadstart"].forEach((type) => {
    audio.addEventListener(type, () => {
      syncIcon();
      if (type === "ended") {
        // Start the preloaded next song right away instead of waiting for the server
        // round trip; the click below only tells Python to advance the queue.
        const nh = document.getElementById("notify-next-src");
        const nu = nh && nh.dataset.src;
        if (nu && blobs[nu]) {
          window.__notifyAudioUrl = nu;
          window.__notifyAdvanced = nu;
          audio.src = blobs[nu];
          audio.play().catch(() => {});
          window.__notifyKeepPlaying = true;  // the click below must not pause it
        }
        const next = document.querySelector(".st-key-next_btn button");
        if (next) next.click();
      }
    });
  });
  setInterval(syncIcon, 250);
})();
