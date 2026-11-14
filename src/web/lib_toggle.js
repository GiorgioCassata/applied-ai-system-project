(function () {
  if (window.__notifyLibToggleBound) return;
  window.__notifyLibToggleBound = true;
  const root = document.documentElement;
  try {
    if (localStorage.getItem("notifyLibCollapsed") === "1") root.dataset.libCollapsed = "1";
  } catch (e) {}
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#lib-toggle")) return;
    const wasCollapsed = root.dataset.libCollapsed === "1";
    if (wasCollapsed) delete root.dataset.libCollapsed;
    else root.dataset.libCollapsed = "1";
    try { localStorage.setItem("notifyLibCollapsed", wasCollapsed ? "0" : "1"); } catch (e) {}
  }, true);
})();
