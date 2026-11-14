(function () {
  if (window.__notifyChatScrollBound) return;
  window.__notifyChatScrollBound = true;
  const scroller = (root) => {
    const all = [root, ...root.querySelectorAll("*")];
    return all.find((el) => /(auto|scroll)/.test(getComputedStyle(el).overflowY)
                            && el.scrollHeight > el.clientHeight + 1);
  };
  let last = null;
  setInterval(() => {
    const end = document.querySelector(".chat-end");
    // No messages, or the chat panel is not the open one: forget the position, so the
    // list scrolls to the newest message again each time the panel is opened.
    const open = document.documentElement.dataset.rightPanel === "chat";
    if (!end || !open) { last = null; return; }
    if (end.dataset.count === last) return;
    const root = document.querySelector('[class*="st-key-chat_list"]');
    const box = root && scroller(root);
    if (!box) return;
    last = end.dataset.count;
    box.scrollTop = box.scrollHeight;
  }, 150);
})();
