// Status bar: what each long request is doing, while it runs.
//
// Long server paths report stages to utils/status.py under a session cookie;
// this script sets that cookie, counts Dash's in-flight callback requests, and
// polls `<base>status` only while one has been open for a moment — a quick
// callback never makes the bar flicker. Entries render into #status-bar.
(function () {
  const COOKIE = "asifast_sid";
  const SHOW_AFTER_MS = 400;
  const POLL_MS = 700;
  let inFlight = 0;
  let since = 0;
  let timer = null;

  function ensureCookie() {
    if (document.cookie.split("; ").some(function (c) { return c.indexOf(COOKIE + "=") === 0; })) {
      return;
    }
    const id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);
    document.cookie = COOKIE + "=" + id + "; path=/; SameSite=Lax";
  }

  function base() {
    const cfg = document.getElementById("_dash-config");
    try {
      return JSON.parse(cfg.textContent).requests_pathname_prefix || "/";
    } catch (e) {
      return "/";
    }
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text) { node.textContent = text; }
    return node;
  }

  function render(entries) {
    const bar = document.getElementById("status-bar");
    if (!bar) { return; }
    bar.replaceChildren();
    entries.forEach(function (e) {
      const row = el("div", "status-row status-" + e.state);
      row.appendChild(el("span", "status-label", e.label));
      const stage = e.stage + (e.total ? " " + e.done + "/" + e.total : "") + (e.detail ? " · " + e.detail : "");
      row.appendChild(el("span", "status-stage", stage));
      if (e.total) {
        const track = el("span", "status-track");
        const fill = el("span", "status-fill");
        fill.style.width = Math.min(100, Math.round(100 * e.done / e.total)) + "%";
        track.appendChild(fill);
        row.appendChild(track);
      }
      bar.appendChild(row);
    });
    bar.classList.toggle("is-active", entries.length > 0);
  }

  function poll() {
    fetch(base() + "status", { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (entries) {
        render(entries);
        const running = entries.some(function (e) { return e.state === "running"; });
        if (running) { start(); }  // e.g. a Copernicus job waiting between quick polls
        if (inFlight === 0 && !running && !entries.length) { stop(); }
      })
      .catch(function () { /* the bar is a convenience; never break the page */ });
  }

  function start() {
    if (!timer) { timer = setInterval(poll, POLL_MS); }
  }

  function stop() {
    clearInterval(timer);
    timer = null;
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.indexOf("_dash-update-component") === -1) {
      return realFetch(input, init);
    }
    if (inFlight === 0) { since = Date.now(); }
    inFlight += 1;
    const pending = setTimeout(function () { if (inFlight > 0) { start(); poll(); } }, SHOW_AFTER_MS);
    return realFetch(input, init).finally(function () {
      inFlight -= 1;
      clearTimeout(pending);
      poll();  // shows "Done"/"Failed" once, and picks up jobs that outlive the request
    });
  };

  ensureCookie();
})();
