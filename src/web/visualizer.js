(function () {
  if (window.__notifyVisualizerBound) return;
  window.__notifyVisualizerBound = true;

  // The visualizer is a full-screen overlay built once and attached to the page body, outside
  // Streamlit's own elements, so a rerun cannot remove it. The eye button in the player bar
  // opens it; a click anywhere on it (or Escape) closes it. It listens to the song through
  // a Web Audio AnalyserNode on the same Audio object that plays the music.
  let overlay = null, canvas = null, ctx2d = null, art = null, title = null, artist = null;
  let frame = 0, infoTimer = 0, open = false;
  // The colours move with the song's tempo: a beat clock advances at the song's bpm (taken
  // from the player bar), and every colour change is a function of it, so a fast song
  // shifts colour faster than a slow one. bpm eases towards a new song's tempo, and the
  // clock only ever moves forward, so colours never jump when the song changes.
  let targetBpm = 110, bpm = 110, beats = 0, lastTime = 0;

  const build = () => {
    // Built with createElement, not HTML strings: Streamlit's sanitizer deletes a script
    // whose text contains anything that looks like a tag.
    const make = (tag, id, parent) => {
      const el = document.createElement(tag);
      if (id) el.id = id;
      if (parent) parent.appendChild(el);
      return el;
    };
    overlay = make("div", "viz-overlay", document.body);
    canvas = make("canvas", "viz-canvas", overlay);
    ctx2d = canvas.getContext("2d");
    const info = make("div", "viz-info", overlay);
    art = make("div", "viz-art", info);
    const text = make("div", "viz-text", info);
    title = make("div", "viz-title", text);
    artist = make("div", "viz-artist", text);
    overlay.addEventListener("click", close);
    window.addEventListener("resize", () => { if (open) resize(); });
  };

  // One analyser for the page's lifetime (an audio element can only be wrapped once, and
  // after that its sound goes through this graph, so the graph must stay connected).
  const analyser = () => {
    if (window.__notifyAnalyser) return window.__notifyAnalyser;
    const audio = (window.__notifyAudio = window.__notifyAudio || new Audio());
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    try {
      const context = new AC();
      const source = context.createMediaElementSource(audio);
      const node = context.createAnalyser();
      node.fftSize = 1024;
      node.smoothingTimeConstant = 0.78;
      source.connect(node);
      node.connect(context.destination);
      window.__notifyAnalyser = { context, node };
    } catch (e) {
      return null;
    }
    return window.__notifyAnalyser;
  };

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    ctx2d.fillStyle = "#000";
    ctx2d.fillRect(0, 0, canvas.width, canvas.height);
  };

  // The song info is copied from the player bar, so it always matches what is playing.
  const syncInfo = () => {
    const src = document.querySelector(".st-key-player_bar .np-art, .np-art");
    if (src && src.tagName === "IMG") {
      let img = art.querySelector("img");
      if (!img) { art.textContent = ""; img = document.createElement("img"); art.appendChild(img); }
      if (img.getAttribute("src") !== src.getAttribute("src")) img.setAttribute("src", src.getAttribute("src"));
    } else if (!art.querySelector(".viz-art-empty")) {
      art.textContent = "";
      const empty = document.createElement("div");
      empty.className = "viz-art-empty";
      empty.textContent = "?";
      art.appendChild(empty);
    }
    const holder = document.querySelector(".np-info");
    const tempo = holder ? parseFloat(holder.getAttribute("data-bpm")) : 0;
    targetBpm = tempo > 0 ? Math.min(200, Math.max(60, tempo)) : 110;
    const t = document.querySelector(".np-title"), a = document.querySelector(".np-artist");
    title.textContent = t ? t.textContent : "";
    artist.textContent = a ? a.textContent : "";
  };

  const draw = (time) => {
    if (!open) return;
    frame = requestAnimationFrame(draw);
    const w = canvas.width, h = canvas.height, s = Math.min(w, h);
    const cx = w / 2, cy = h / 2;
    const live = window.__notifyAnalyser;
    const bins = live ? live.node.frequencyBinCount : 0;
    const freq = new Uint8Array(bins), wave = new Uint8Array(bins);
    if (live) { live.node.getByteFrequencyData(freq); live.node.getByteTimeDomainData(wave); }

    let bass = 0;
    for (let i = 0; i < 6 && i < bins; i++) bass += freq[i];
    bass = bins ? bass / (6 * 255) : 0;
    const t = time / 1000;
    const dt = lastTime ? Math.min(0.1, (time - lastTime) / 1000) : 0;
    lastTime = time;
    bpm += (targetBpm - bpm) * 0.04;
    beats += dt * bpm / 60;
    const beat = beats % 1;
    const thump = Math.pow(1 - beat, 3);  // 1 on each beat, fading away before the next
    // The hue travels right round the colour wheel once every 32 beats (about 16 s at 120
    // bpm); the colour ebbs and flows in strength over 8 beats; each beat adds a soft pulse.
    const hueBase = (beats * 360 / 32) % 360;
    const flow = 0.5 + 0.5 * Math.sin(beats * Math.PI * 2 / 8);  // 0..1, slowly
    const sat = 70 + flow * 25;

    // fade the last frame instead of clearing it, which leaves soft trails
    ctx2d.globalCompositeOperation = "source-over";
    ctx2d.fillStyle = "rgba(0, 0, 0, 0.2)";
    ctx2d.fillRect(0, 0, w, h);

    const base = s * 0.17 * (1 + bass * 0.3 + thump * 0.04);
    const glow = ctx2d.createRadialGradient(cx, cy, base * 0.2, cx, cy, s * 0.62);
    glow.addColorStop(0, `hsla(${hueBase}, ${sat}%, 40%, ${0.10 + bass * 0.30 + thump * 0.08})`);
    glow.addColorStop(1, `hsla(${hueBase}, ${sat}%, 40%, 0)`);
    ctx2d.fillStyle = glow;
    ctx2d.fillRect(0, 0, w, h);

    ctx2d.globalCompositeOperation = "lighter";
    ctx2d.lineCap = "round";
    ctx2d.lineWidth = Math.max(2, s * 0.006);
    const N = 120, usable = Math.min(bins, 360);
    for (let i = 0; i < N; i++) {
      const k = i < N / 2 ? i : N - 1 - i;  // mirrored, so both sides move together
      const bin = Math.floor(Math.pow(k / (N / 2), 1.5) * (usable - 1));
      let v = bins ? freq[bin] / 255 : 0;
      if (v < 0.02) v = 0.03 + 0.02 * Math.sin(t * 2 + i * 0.3);  // a little life when quiet
      const angle = (i / N) * Math.PI * 2 - Math.PI / 2 + t * 0.05;
      const len = 4 + v * v * s * 0.3 + v * s * 0.05;
      const x0 = cx + Math.cos(angle) * base, y0 = cy + Math.sin(angle) * base;
      const x1 = cx + Math.cos(angle) * (base + len), y1 = cy + Math.sin(angle) * (base + len);
      // each bar is a little further round the wheel than its neighbour, so a ring shows a
      // band of colours that slides round with the beat clock
      const hue = hueBase + (k / (N / 2)) * 90 + v * 30;
      ctx2d.strokeStyle = `hsla(${hue}, ${sat}%, ${45 + v * 25 + flow * 8}%, ${0.5 + v * 0.5})`;
      ctx2d.beginPath();
      ctx2d.moveTo(x0, y0);
      ctx2d.lineTo(x1, y1);
      ctx2d.stroke();
    }

    // a ring that follows the waveform itself
    ctx2d.lineWidth = Math.max(1.5, s * 0.003);
    ctx2d.strokeStyle = `hsla(${hueBase + 180}, 60%, 85%, ${0.45 + thump * 0.3})`;
    ctx2d.beginPath();
    const points = 180;
    for (let i = 0; i <= points; i++) {
      const sample = bins ? (wave[Math.floor((i % points) / points * (bins - 1))] - 128) / 128 : 0;
      const r = base * 0.82 + sample * base * 0.35;
      const angle = (i / points) * Math.PI * 2;
      const x = cx + Math.cos(angle) * r, y = cy + Math.sin(angle) * r;
      if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
  };

  const show = () => {
    if (!overlay) build();
    const live = analyser();
    if (live && live.context.state === "suspended") live.context.resume().catch(() => {});
    open = true;
    lastTime = 0;
    overlay.style.display = "block";
    resize();
    syncInfo();
    infoTimer = setInterval(syncInfo, 300);
    frame = requestAnimationFrame(draw);
  };

  function close() {
    open = false;
    cancelAnimationFrame(frame);
    clearInterval(infoTimer);
    if (overlay) overlay.style.display = "none";
  }

  document.addEventListener("click", (e) => {
    if (e.target.closest("#viz-toggle")) show();
  }, true);
  document.addEventListener("keydown", (e) => {
    if (open && e.key === "Escape") close();
  });
})();
