// Closes a popover menu ("...") once one of the buttons inside it has been clicked.
// Streamlit keeps a popover open across the rerun that the click causes, so the menu would
// otherwise stay on screen. The click itself is left alone (the button's own handler still
// runs); a moment later the popover's trigger button, which toggles it and is marked
// aria-expanded="true" while it is open, is clicked to close it. If the rerun has already
// closed or replaced the popover, nothing is open and nothing happens.
(function () {
  if (window.__notifyPopoverAutoCloseBound) return;
  window.__notifyPopoverAutoCloseBound = true;
  document.addEventListener("click", (e) => {
    const body = e.target.closest && e.target.closest('[data-testid="stPopoverBody"]');
    if (!body || !e.target.closest("button")) return;
    setTimeout(() => {
      document
        .querySelectorAll('[data-testid="stPopoverButton"][aria-expanded="true"]')
        .forEach((trigger) => trigger.click());
    }, 150);
  }, true);
})();
