(function () {
  if (window.__notifyQueueScrollBound) return;
  window.__notifyQueueScrollBound = true;
  const scroller = (root) => {
    const all = [root, ...root.querySelectorAll("*")];
    return all.find((el) => /(auto|scroll)/.test(getComputedStyle(el).overflowY)
                            && el.scrollHeight > el.clientHeight + 1);
  };
  let last = null;
  setInterval(() => {
    const label = document.querySelector(".q-now-label");
    // Queue panel not the open one: scroll to "Now playing" again when it is opened
    const open = document.documentElement.dataset.rightPanel === "queue";
    if (!label || !open) { last = null; return; }
    if (label.dataset.now === last) return;
    const root = document.querySelector('[class*="st-key-queue_list"]');
    const box = root && scroller(root);
    if (!box) return;
    last = label.dataset.now;
    box.scrollTop += label.getBoundingClientRect().top - box.getBoundingClientRect().top - 4;
  }, 150);
})();
