(function () {
  if (window.__notifyChatScrollBound) return;
  window.__notifyChatScrollBound = true;
  const scroller = (root) => {
    const all = [root, ...root.querySelectorAll("*")];
    return all.find((el) => /(auto|scroll)/.test(getComputedStyle(el).overflowY)
                            && el.scrollHeight > el.clientHeight + 1);
  };
  // What was seen last time round: the message count, and the list and its height. The
  // list is scrolled to the bottom when a message is added, and then kept there as long
  // as it grows (the "Thinking..." line, a long reply) while it was already at the bottom;
  // a listener who has scrolled up to read is left where they are.
  let last = null, prevBox = null, prevHeight = 0;
  setInterval(() => {
    const end = document.querySelector(".chat-end");
    // No messages, or the chat panel is not the open one: forget the position, so the
    // list scrolls to the newest message again each time the panel is opened.
    const open = document.documentElement.dataset.rightPanel === "chat";
    if (!end || !open) { last = null; prevBox = null; return; }
    const root = document.querySelector('[class*="st-key-chat_list"]');
    const box = root && scroller(root);
    if (!box) return;
    const wasAtBottom = prevBox === box && box.scrollTop + box.clientHeight >= prevHeight - 40;
    const grew = box.scrollHeight > prevHeight;
    if (end.dataset.count !== last || (wasAtBottom && grew)) {
      box.scrollTop = box.scrollHeight;
    }
    last = end.dataset.count;
    prevBox = box;
    prevHeight = box.scrollHeight;
  }, 150);
})();
