(function () {
  window.__notifyArtMap = __ART_MAP__;  // {artist|album|song|playlist: {name or id: file}}
  window.__notifyArtPrefix = "__ART_PREFIX__";
  if (window.__notifyArtBound) return;
  window.__notifyArtBound = true;
  const SEL = '[class*="st-key-artist_btn"], [class*="st-key-album_btn"], [class*="st-key-playlist_btn"]';
  const visible = new Set();
  const seen = new WeakSet();

  // Playing from the library: a single click on an item still opens its page, but
  //  - a double click anywhere on the item plays it, and
  //  - a click on its round bubble plays it (the bubble shows a play icon on hover).
  // Playing is done by clicking the item's hidden "Play" button, which sits next to it
  // with the same key suffix (artist_btn7 -> artist_play7).
  const playButtonFor = (item) => {
    const cls = [...item.classList].find((c) => /^st-key-(artist|album|playlist)_btn/.test(c));
    if (!cls) return null;
    const holder = document.querySelector("." + cls.replace("_btn", "_play"));
    return holder && holder.querySelector("button");
  };
  const onBubble = (e, button) => {
    const r = button.getBoundingClientRect();
    const collapsed = document.documentElement.dataset.libCollapsed === "1";
    const left = collapsed ? r.left + (r.width - 40) / 2
                           : r.left + (parseFloat(getComputedStyle(button).paddingLeft) || 8);
    const top = r.top + (r.height - 40) / 2;  // the bubble is 40px, centred vertically
    return e.clientX >= left && e.clientX <= left + 40 && e.clientY >= top && e.clientY <= top + 40;
  };
  document.addEventListener("click", (e) => {
    const item = e.target.closest(SEL);
    const button = item && item.querySelector("button");
    const play = item && playButtonFor(item);
    if (!button || !play) return;
    const bubble = onBubble(e, button);
    if (!bubble && e.detail < 2) return;  // single click elsewhere: open the page as before
    e.stopPropagation();  // never also open the page for a play click
    e.preventDefault();
    // A bubble click plays on its first click (a second click of a double click must not
    // restart it); elsewhere only the second click of a double click plays.
    if (bubble ? e.detail === 1 : e.detail === 2) play.click();
  }, true);
  const urlFor = (el) => {
    const cls = [...el.classList].find((c) => /^st-key-(artist|album|playlist)_btn/.test(c));
    if (!cls) return null;
    const kind = cls.match(/^st-key-(artist|album|playlist)_btn/)[1];
    let value;
    if (kind === "playlist") {
      value = cls.replace("st-key-playlist_btn", "");  // the playlist's id
    } else {
      const p = el.querySelector("button p");
      value = p ? p.textContent.trim() : "";
    }
    const file = value && (window.__notifyArtMap[kind] || {})[value];
    return file ? window.__notifyArtPrefix + encodeURIComponent(file) : null;
  };
  const refresh = () => {
    const collapsed = document.documentElement.dataset.libCollapsed === "1";
    visible.forEach((el) => {
      if (!el.isConnected) { visible.delete(el); return; }
      // While the panel is collapsed the names are hidden, so show them as a hover
      // hint (the browser's own tooltip, which the panel's clipping cannot cut off)
      const button = el.querySelector("button");
      const label = el.querySelector("button p");
      const hint = collapsed && label ? label.textContent.trim() : "";
      if (button && button.title !== hint) button.title = hint;
      const url = urlFor(el) || "";
      // Streamlit reuses elements when the list changes (e.g. searching), so the
      // label can change under the same element: compare against the last URL.
      if ((el.dataset.artUrl || "") !== url) {
        el.dataset.artUrl = url;
        if (url) el.style.setProperty("--art", 'url("' + url + '")');
        else el.style.removeProperty("--art");
      }
    });
  };
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { e.isIntersecting ? visible.add(e.target) : visible.delete(e.target); });
    refresh();
  }, { rootMargin: "120px" });
  setInterval(() => {
    document.querySelectorAll(SEL).forEach((el) => {
      if (!seen.has(el)) { seen.add(el); io.observe(el); }
    });
    refresh();
    // The "+" (new playlist) gets its hover hint the same way as the library bubbles: the
    // browser's own tooltip (a title), not Streamlit's styled help popup.
    const plus = document.querySelector('[class*="st-key-lib_new"] button');
    if (plus && plus.title !== "Create a new playlist") plus.title = "Create a new playlist";
  }, 400);
})();
