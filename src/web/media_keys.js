// Keyboard / headset media keys (Next track, Previous track) skip songs by clicking the
// player's own skip buttons. The browser routes those keys to whichever page owns the
// Media Session; the Audio object in play_pause.js is what makes this page the owner.
(function () {
  if (window.__notifyMediaKeysBound) return;
  window.__notifyMediaKeysBound = true;
  if (!("mediaSession" in navigator)) return;
  const click = (key) => {
    const btn = document.querySelector('.st-key-' + key + ' button');
    if (btn) btn.click();
  };
  try {
    navigator.mediaSession.setActionHandler("nexttrack", () => click("next_btn"));
    navigator.mediaSession.setActionHandler("previoustrack", () => click("prev_btn"));
  } catch (e) {}
})();
