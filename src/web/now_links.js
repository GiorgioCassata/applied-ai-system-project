(function () {
  if (window.__notifyLinksBound) return;
  window.__notifyLinksBound = true;
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest(".np-title, .np-artist");
    if (!t) return;
    const key = t.classList.contains("np-title") ? "npl_title" : "npl_artist";
    const button = document.querySelector('[class*="st-key-' + key + '"] button');
    if (button) button.click();
  }, true);
})();
