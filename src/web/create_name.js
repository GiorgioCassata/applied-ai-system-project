(function () {
  if (window.__notifyCreateNameBound) return;
  window.__notifyCreateNameBound = true;
  document.addEventListener("focusin", (e) => {
    const input = e.target;
    if (!input.matches || !input.matches('[class*="st-key-new_playlist_name"] input')) return;
    if (!/^Playlist \d+$/.test(input.value)) return;
    input.select();
    const keep = (ev) => { ev.preventDefault(); input.removeEventListener("mouseup", keep); };
    input.addEventListener("mouseup", keep);
  });
})();
