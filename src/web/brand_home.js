(function () {
  if (window.__notifyBrandBound) return;
  window.__notifyBrandBound = true;
  document.addEventListener("click", (e) => {
    if (!e.target.closest || !e.target.closest(".brand")) return;
    const home = document.querySelector('[class*="st-key-home_btn"] button');
    if (home) home.click();
  }, true);
})();
