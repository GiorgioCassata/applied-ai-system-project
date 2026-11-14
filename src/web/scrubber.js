(function () {
  if (window.__notifyScrubBound) return;
  window.__notifyScrubBound = true;
  let dragging = false;
  window.__notifyVolume = 1;
  window.__notifyMuted = false;
  // The audio player is the Audio object created by the play/pause script above.
  const getAudio = () => (window.__notifyAudio = window.__notifyAudio || new Audio());
  const fmt = (s) => {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ":" + String(sec).padStart(2, "0");
  };
  document.addEventListener("pointerdown", (e) => {
    if (e.target.id === "np-seek") dragging = true;
  }, true);
  document.addEventListener("pointerup", () => { dragging = false; }, true);
  document.addEventListener("click", (e) => {
    if (e.target.closest && e.target.closest("#np-mute")) {
      window.__notifyMuted = !window.__notifyMuted;
      // Unmuting with the slider at 0 would stay silent, so bring it back up a bit.
      if (!window.__notifyMuted && window.__notifyVolume === 0) window.__notifyVolume = 0.5;
    }
  }, true);
  document.addEventListener("input", (e) => {
    const audio = getAudio();
    if (!audio) return;
    if (e.target.id === "np-seek" && isFinite(audio.duration)) {
      audio.currentTime = (e.target.value / 1000) * audio.duration;
    } else if (e.target.id === "np-vol") {
      window.__notifyVolume = parseFloat(e.target.value);
      window.__notifyMuted = false;  // moving the slider unmutes
      audio.volume = window.__notifyVolume;
    }
  }, true);
  setInterval(() => {
    const audio = getAudio();
    const seek = document.getElementById("np-seek");
    if (!seek) return;
    const muted = window.__notifyMuted || window.__notifyVolume === 0;
    const wanted = muted ? 0 : window.__notifyVolume;
    if (audio && audio.volume !== wanted) audio.volume = wanted;
    const ok = audio && isFinite(audio.duration) && audio.duration > 0;
    document.getElementById("np-cur").textContent = fmt(audio ? audio.currentTime : 0);
    document.getElementById("np-dur").textContent = fmt(ok ? audio.duration : 0);
    if (!dragging) seek.value = ok ? (audio.currentTime / audio.duration) * 1000 : 0;
    seek.disabled = !ok;
    seek.style.setProperty("--p", (seek.value / 10) + "%");
    const vol = document.getElementById("np-vol");
    if (vol) {
      const shown = muted ? 0 : window.__notifyVolume;
      vol.value = shown;
      vol.style.setProperty("--p", (shown * 100) + "%");
    }
    const mb = document.getElementById("np-mute");
    if (mb) {
      mb.dataset.muted = muted ? "1" : "0";
      mb.setAttribute("aria-label", muted ? "Unmute" : "Mute");
    }
  }, 250);
})();
