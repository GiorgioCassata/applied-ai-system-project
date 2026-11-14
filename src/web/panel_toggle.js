(function () {
  if (window.__notifyPanelToggleBound) return;
  window.__notifyPanelToggleBound = true;
  const root = document.documentElement;
  document.addEventListener("click", (e) => {
    const button = e.target.closest("#queue-toggle, #chat-toggle");
    if (!button) return;
    const name = button.id === "chat-toggle" ? "chat" : "queue";
    if (root.dataset.rightPanel === name) {
      delete root.dataset.rightPanel;
      return;
    }
    root.dataset.rightPanel = name;
    if (name === "chat") {  // ready to type as soon as the chat opens
      setTimeout(() => {
        const box = document.querySelector('[class*="st-key-chat_text"] textarea');
        if (box) box.focus();
      }, 150);
    }
  }, true);
})();
