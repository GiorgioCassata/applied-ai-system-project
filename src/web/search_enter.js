(function () {
  if (window.__notifySearchEnterBound) return;
  window.__notifySearchEnterBound = true;
  // Streamlit only reruns on Enter when the text changed. When Enter is pressed again on
  // the text already searched, click the hidden "search_go" button to run the search again.
  let last = null;
  const sel = '[class*="st-key-top_bar"] [class*="st-key-search"] input';
  document.addEventListener("input", (e) => {
    if (e.target.matches && e.target.matches(sel) && e.target.value === "") last = null;
  }, true);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || !e.target.matches || !e.target.matches(sel)) return;
    const value = e.target.value.trim();
    if (!value) return;
    if (value === last) {
      const go = document.querySelector('[class*="st-key-search_go"] button');
      if (go) go.click();
    }
    last = value;
  }, true);
})();
