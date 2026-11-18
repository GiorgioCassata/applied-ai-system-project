// Hover hints for some buttons, as the browser's own tooltip (a title attribute), like the
// library panel's buttons, instead of Streamlit's styled help popup. Streamlit can redraw
// a button at any time, so the titles are put back whenever they go missing.
(function () {
  if (window.__notifyNativeHintsBound) return;
  window.__notifyNativeHintsBound = true;
  const HINTS = {
    '.st-key-vmore button': "More",
    '.st-key-profile_btn button': "Settings",
    '.st-key-shuffle_off button': "Shuffle: off",
    '.st-key-shuffle_on button': "Shuffle: on",
    '.st-key-repeat_off button': "Repeat: off (nothing plays after the songs)",
    '.st-key-repeat_all button': "Repeat: album, playlist or mix",
    '.st-key-repeat_one button': "Repeat: this song",
    '.st-key-repeat_autoplay button': "Autoplay: keeps adding recommended songs",
  };
  setInterval(() => {
    for (const [selector, text] of Object.entries(HINTS)) {
      document.querySelectorAll(selector).forEach((el) => {
        if (el.title !== text) el.title = text;
      });
    }
  }, 300);
})();
