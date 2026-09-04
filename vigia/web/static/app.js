/* VIGÍA operator view — vanilla JS, no build step, no dependencies.
 *
 * Two channels, matching the server: MJPEG carries the video via a plain <img>
 * (the browser's own decoder does the work), and a WebSocket carries only small
 * JSON state. Frames never travel on the event channel — that is the bottleneck
 * the reference implementation reported, and it is avoided rather than tuned.
 *
 * If the socket drops, the view falls back to polling /api/state so the demo
 * degrades to "slower" rather than to "frozen". On a stage, a frozen panel
 * looks like a crash.
 */

const $ = (id) => document.getElementById(id);

const el = {
  context: $("context"), chips: $("chips"), gate: $("gate"),
  levels: $("levels"), confirmedCount: $("confirmed-count"),
  events: $("events"), toggle: $("toggle-suppressed"),
  cameraSelect: $("camera-select"), video: $("video"),
  tlCandidates: $("tl-candidates"), tlConfirmed: $("tl-confirmed"),
  seen: $("m-seen"), confirmed: $("m-confirmed"), suppressed: $("m-suppressed"),
  rate: $("m-rate"), dedup: $("m-dedup"), fps: $("m-fps"), latency: $("m-latency"),
  link: $("link"), linkDot: $("link-dot"),
};

/* Levels the validator actually runs. Size plausibility was implemented,
   measured across three hazards, found to contribute nothing on two and to be
   harmful on the third, and removed — so it is absent here rather than shown
   greyed out. The interface should not advertise a level that does not exist. */
const LEVEL_ORDER = ["confidence", "colour", "persistence", "cooldown"];

function setLink(state, text) {
  el.linkDot.className = "dot" + (state ? " " + state : "");
  el.link.textContent = text;
}

function renderChips(active, all) {
  const set = new Set(active);
  el.chips.innerHTML = "";
  (all || []).forEach((hazard) => {
    const chip = document.createElement("div");
    chip.className = "chip" + (set.has(hazard) ? " on" : "");
    chip.innerHTML = `<i></i>${hazard.replace(/_/g, " ")}`;
    el.chips.appendChild(chip);
  });
}

function renderLevels(levels) {
  const byName = {};
  (levels || []).forEach((level) => { byName[level.name] = level; });

  el.levels.innerHTML = "";
  LEVEL_ORDER.forEach((name, index) => {
    const level = byName[name] || { seen: 0, passed: 0, rejected: 0 };
    const seen = level.seen || 0;
    const passed = level.passed || 0;
    // A level that has seen nothing is disabled for this hazard, not failing.
    const inactive = seen === 0;
    const pct = seen ? (passed / seen) * 100 : 0;

    const row = document.createElement("div");
    row.className = "level";
    row.innerHTML =
      `<div class="level-head">
         <span class="level-name${inactive ? " off" : ""}">${index + 1} ${name}</span>
         <span class="level-counts"><b>${passed}</b> / ${seen}</span>
       </div>
       <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>`;
    el.levels.appendChild(row);
  });
}

function renderTimeline(timeline) {
  const rows = timeline || [];
  el.tlCandidates.innerHTML = "";
  el.tlConfirmed.innerHTML = "";

  rows.forEach((row) => {
    // Candidate row: height encodes how many were proposed that frame.
    const candidate = document.createElement("i");
    candidate.className = "tick";
    candidate.style.height = row.proposed ? Math.min(16, 3 + row.proposed * 3) + "px" : "1px";
    el.tlCandidates.appendChild(candidate);

    // Confirmed row: sparse by design. The contrast is the argument.
    const confirmed = document.createElement("i");
    confirmed.className = "tick" + (row.confirmed ? " hit" : "");
    confirmed.style.height = row.confirmed ? "16px" : "1px";
    el.tlConfirmed.appendChild(confirmed);
  });
}

function renderEvents(events) {
  if (!events || !events.length) {
    el.events.innerHTML = '<p class="empty">none yet</p>';
    return;
  }
  el.events.innerHTML = "";
  events.forEach((event) => {
    const node = document.createElement("div");
    node.className = "event";
    node.innerHTML =
      `<div class="top">
         <span class="haz">${event.hazard.replace(/_/g, " ")}</span>
         <span class="meta">${event.confidence.toFixed(2)}</span>
       </div>
       <div class="meta">frame ${event.frame} · ${event.supporting} supporting ·
         ${event.seconds_to_confirm}s to confirm</div>`;
    el.events.appendChild(node);
  });
}

function render(state) {
  el.context.textContent = state.context || "—";
  renderChips(state.hazards, state.all_hazards);
  renderLevels(state.levels);
  renderTimeline(state.timeline);
  renderEvents(state.events);

  el.confirmedCount.textContent = state.confirmed;
  el.seen.textContent = state.proposed;
  el.confirmed.textContent = state.confirmed;
  el.suppressed.textContent = state.suppressed;
  // Two numbers, not one: rejecting a candidate as not credible and declining
  // to re-report an event already sent support different claims.
  const proposed = Math.max(1, state.proposed);
  const filtered = state.filtered || 0;
  const deduped = state.deduplicated || 0;
  el.rate.textContent = Math.round((filtered / proposed) * 100) + "%";
  if (el.dedup) el.dedup.textContent = Math.round((deduped / proposed) * 100) + "%";
  el.fps.textContent = state.fps.toFixed(1);
  el.latency.textContent = Math.round(state.latency_ms) + " ms";


}

async function loadGate() {
  try {
    const response = await fetch("/api/gate");
    const data = await response.json();
    const summary = data.summary || {};
    if (summary.detector_invocations_per_frame_gated !== undefined) {
      el.gate.textContent =
        `gate ${summary.detector_invocations_per_frame_gated}/` +
        `${summary.detector_invocations_per_frame_ungated} ` +
        `(${Math.round((summary.reduction || 0) * 100)}% fewer)`;
    }
  } catch (_) { /* gate detail is optional furniture */ }
}

/* Which camera the view is showing. Switching it re-points the video element
   and the event socket — the header selector is how Tier 0 becomes legible,
   because the detector chips visibly change with the camera. */
let camera = null;

async function loadCameras() {
  try {
    const { cameras } = await (await fetch("/api/cameras")).json();
    if (!cameras || !cameras.length) return;
    el.cameraSelect.innerHTML = "";
    cameras.forEach((c) => {
      const option = document.createElement("option");
      option.value = c.camera_id;
      option.textContent = `${c.camera_id} — ${c.context}`;
      el.cameraSelect.appendChild(option);
    });
    camera = cameras[0].camera_id;
    el.cameraSelect.value = camera;
    el.video.src = `/video?camera=${encodeURIComponent(camera)}`;
    el.cameraSelect.disabled = cameras.length < 2;
  } catch (_) { /* single-camera run: the selector stays as it is */ }
}

function switchCamera(next) {
  camera = next;
  // Re-point the MJPEG stream, then reconnect the event socket to match.
  el.video.src = `/video?camera=${encodeURIComponent(camera)}`;
  if (socket) { socket.onclose = null; socket.close(); }
  connect();
}

let socket = null;
let pollTimer = null;

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const q = camera ? `?camera=${encodeURIComponent(camera)}` : "";
      render(await (await fetch("/api/state" + q)).json());
      setLink("live", "polling");
    } catch (_) {
      setLink("down", "offline");
    }
  }, 500);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const q = camera ? `?camera=${encodeURIComponent(camera)}` : "";
  socket = new WebSocket(`${scheme}://${location.host}/events${q}`);

  socket.onopen = () => { stopPolling(); setLink("live", "live"); };
  socket.onmessage = (message) => render(JSON.parse(message.data));
  socket.onerror = () => setLink("down", "socket error");
  socket.onclose = () => {
    setLink("down", "reconnecting");
    startPolling();                 // degrade to slower, never to frozen
    setTimeout(connect, 2000);
  };
}

el.toggle.addEventListener("change", () => {
  fetch(`/api/suppressed/${el.toggle.checked}`, { method: "POST" });
});

el.cameraSelect.addEventListener("change", (e) => switchCamera(e.target.value));

el.video = $("video");
loadGate();
loadCameras().then(connect);
