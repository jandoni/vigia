/* VIGÍA demonstration interface.
 *
 * Vanilla JS, no build step — a judge or a municipality should be able to
 * clone this and run it. Three independent screens share only the navigation.
 *
 * The live screen keeps the two-channel split the engineer's view uses: MJPEG
 * in an <img> for video, a WebSocket carrying only small JSON for state.
 * Frames never travel on the event channel.
 */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/* ------------------------------------------------------------ navigation */
/* Every screen has a real address. The interface stays one document — there is
 * no build step and no router library — but /analyse is a link you can send
 * someone, and the back button does what a back button should. */

const PATHS = {
  home: "/overview",
  analyse: "/analyse",
  live: "/live",
  how: "/how-it-works",
  train: "/teach",
};
const PAGES = Object.fromEntries(
  Object.entries(PATHS).map(([page, path]) => [path, page]));

let current = null;

function show(page) {
  if (page === current) return;
  current = page;
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("on"));
  $("page-" + page).classList.add("on");
  document.querySelectorAll("nav button").forEach((b) =>
    b.classList.toggle("on", b.dataset.go === page));
  window.scrollTo(0, 0);
  if (page === "live") startLive(); else stopLive();
  if (page === "train") loadTraining();
}

function go(page) {
  if (page === current) return;
  history.pushState({ page }, "", PATHS[page] || "/overview");
  show(page);
}

// An unknown path shows the overview rather than an error: a mistyped demo
// URL should still open the demo.
const pageForLocation = () =>
  PAGES[location.pathname.replace(/\/+$/, "") || "/"] || "home";

window.addEventListener("popstate", () => show(pageForLocation()));

document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-go]");
  if (target) go(target.dataset.go);
});

/* --------------------------------------------------------------- analyse */

const state = { clip: null, clipTitle: "", credit: "", scene: null, result: null };

async function loadClips() {
  const list = $("clip-list");
  list.innerHTML = "";
  try {
    const { clips } = await (await fetch("/api/clips")).json();
    if (!clips.length) {
      list.appendChild(el("p", "faint", "No sample footage found. Build it with: make clips"));
      return;
    }
    clips.forEach((c) => {
      const row = el("div", "sample");
      const poster = new Image();
      poster.src = c.thumb;
      poster.alt = "";
      row.appendChild(poster);
      const meta = el("div", "meta");
      meta.appendChild(el("div", "title", c.title));
      meta.appendChild(el("div", "blurb", c.blurb || ""));
      row.appendChild(meta);
      row.addEventListener("click", () => pickClip(c, row));
      list.appendChild(row);
    });
  } catch (_) {
    list.appendChild(el("p", "faint", "Could not reach the server."));
  }
}

/* Choosing footage runs the whole thing. The system works out what the
 * footage shows and picks the detector itself — that is the product, and
 * asking first made the viewer do the system's job. The reading it made is
 * stated in the result, and the alternatives stay one click away underneath. */

async function pickClip(clip, row) {
  document.querySelectorAll("#clip-list .sample")
    .forEach((r) => r.classList.remove("on"));
  if (row) row.classList.add("on");
  state.clip = clip.file;
  state.clipTitle = clip.title || clip.file;
  state.credit = clip.attribution || "";
  state.scene = null;
  state.result = null;
  $("scene-block").style.display = "none";
  await runAnalysis(null);
}

async function runAnalysis(hazard) {
  if (!state.clip) return;
  busy(hazard
    ? "Running that detector over the whole clip…"
    : "Working out what this footage shows, then running the right detector…");
  try {
    const body = { file: state.clip };
    if (hazard) body.hazard = hazard;
    const result = await postJSON("/api/analyse/run", body);
    if (result.error) { idle(result.error); return; }
    state.scene = result.hazard;
    state.result = result;
    renderResult(result);
    renderScenes(result.considered || []);
    $("scene-block").style.display = "block";
  } catch (_) {
    idle("The analysis did not finish.");
  }
}

function renderScenes(considered) {
  const list = $("scene-list");
  list.innerHTML = "";
  considered.forEach((c) => {
    const row = el("div", "choice-row");
    row.appendChild(el("span", "pick"));
    row.appendChild(el("span", "name", c.label));
    row.appendChild(el("span", "evidence", c.evidence || ""));
    if (c.hazard === state.scene) row.classList.add("on");
    row.addEventListener("click", () => {
      if (c.hazard === state.scene) return;
      list.querySelectorAll(".choice-row").forEach((r) => r.classList.remove("on"));
      row.classList.add("on");
      runAnalysis(c.hazard);
    });
    list.appendChild(row);
  });
}

function renderResult(r) {
  $("analyse-busy").style.display = "none";
  $("analyse-idle").style.display = "none";
  $("analyse-result").style.display = "block";

  $("result-headline").textContent =
    `${capitalise(r.scene_sentence)} — so it used the ${r.hazard_label} detector.`;
  $("result-sub").textContent =
    `Chosen by the system, not by you. ${r.frames_examined} frames examined ` +
    `in ${r.elapsed_seconds} seconds.`;
  $("result-credit").textContent = state.credit || "";

  const n = r.confirmed;
  $("result-count").textContent =
    n === 0 ? "Nothing confirmed" : `${spell(n)} ${n === 1 ? "finding" : "findings"}`;
  $("result-blurb").textContent = n === 0
    ? (r.note || "The detector proposed things, but nothing stayed put long enough to be trusted.")
    : `Each stayed visible long enough to be sure of, across ${Math.round(r.duration_seconds)} seconds of footage.`;

  const list = $("finding-list");
  list.innerHTML = "";
  r.findings.forEach((f, i) => {
    const row = el("div", "finding");
    row.appendChild(el("span", "at", clock(f.seconds)));
    row.appendChild(el("span", "what", capitalise(f.label)));
    row.appendChild(el("span", "score", Math.round(f.confidence * 100)));
    row.addEventListener("click", () => showFrame(r, i));
    list.appendChild(row);
  });

  $("saved-line").innerHTML =
    `The detector flagged something <span class="mono muted">${r.proposed.toLocaleString()}</span> times. ` +
    `<span class="mono" style="color:var(--confirmed)">${n}</span> ${n === 1 ? "was" : "were"} worth your attention.`;

  if (r.findings.length) showFrame(r, 0);
  else $("stage").innerHTML = '<div class="empty">nothing to show</div>';
}

function showFrame(r, index) {
  const finding = r.findings[index];
  document.querySelectorAll("#finding-list .finding")
    .forEach((n, i) => n.classList.toggle("on", i === index));

  const stage = $("stage");
  stage.innerHTML = "";
  const img = new Image();
  img.src = `/api/frame?file=${encodeURIComponent(state.clip)}&seconds=${finding.seconds}`;
  img.alt = "frame";
  img.onload = () => {
    // Boxes arrive as fractions of the frame, so they need no knowledge of
    // how the still was resized on the way here — percentages just work.
    const [x1, y1, x2, y2] = finding.box;
    const box = el("div", "box");
    box.style.left = `${x1 * 100}%`;
    box.style.top = `${y1 * 100}%`;
    box.style.width = `${(x2 - x1) * 100}%`;
    box.style.height = `${(y2 - y1) * 100}%`;
    const label = el("div", "box-label",
      `${capitalise(finding.label)} ${Math.round(finding.confidence * 100)}`);
    label.style.left = `${x1 * 100}%`;
    label.style.top = `${y1 * 100}%`;
    stage.appendChild(box);
    stage.appendChild(label);
  };
  stage.appendChild(img);
  $("stage-note").textContent =
    `At ${clock(finding.seconds)} · confirmed after ${finding.supporting} sightings`;
}

/* ------------------------------------------------------------ file input */

$("pick-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("upload-note").textContent = "sending…";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await (await fetch("/api/analyse/upload", { method: "POST", body: form })).json();
    $("upload-note").textContent = `${res.name} · ${res.size_mb} MB`;
    document.querySelectorAll("#clip-list .sample").forEach((r) => r.classList.remove("on"));
    await pickClip({ file: res.file, title: res.name, attribution: "" }, null);
  } catch (_) {
    $("upload-note").textContent = "upload failed";
  }
});

/* ------------------------------------------------------------------ live */

let socket = null, poll = null;

let liveCameras = [];

function startLive() {
  fetch("/api/cameras").then((r) => r.json()).then(({ cameras }) => {
    liveCameras = cameras || [];
    const bar = $("live-cameras");
    bar.innerHTML = "";
    liveCameras.forEach((c, i) => {
      const b = el("button", "camera-pick" + (c.live ? " is-live" : ""));
      b.appendChild(el("span", "cam-name", c.name || c.camera_id));
      // A viewer who cannot tell a live public camera from a replayed clip
      // cannot judge either of them, so each says which it is and, for the
      // live ones, how often the people who run it publish a new image.
      b.appendChild(el("span", "cam-kind", c.live
        ? `live \u00b7 every ${Math.round((c.interval_seconds || 60) / 60)} min`
        : "recorded clip"));
      if (i === 0) { b.classList.add("on"); liveCamera = c.camera_id; }
      b.addEventListener("click", () => {
        bar.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        liveCamera = c.camera_id;
        connectLive();
      });
      bar.appendChild(b);
    });
    connectLive();
  }).catch(() => {});
}

let liveCamera = null;

function connectLive() {
  stopLive();
  $("live-video").src = `/video?camera=${encodeURIComponent(liveCamera || "")}`;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/events?camera=${encodeURIComponent(liveCamera || "")}`);
  socket.onmessage = (m) => renderLive(JSON.parse(m.data));
  socket.onclose = () => {
    if (current !== "live") return;
    poll = poll || setInterval(async () => {
      try {
        renderLive(await (await fetch(`/api/state?camera=${encodeURIComponent(liveCamera || "")}`)).json());
      } catch (_) {}
    }, 700);
  };
}

function stopLive() {
  if (socket) { socket.onclose = null; socket.close(); socket = null; }
  if (poll) { clearInterval(poll); poll = null; }
}

function renderLive(s) {
  $("live-context").textContent =
    `${s.context} · watching for ${(s.hazards || []).join(", ").replace(/_/g, " ")}`;
  $("live-credit").textContent = liveCredit(s);
  $("live-headline").innerHTML =
    `The camera has raised <span class="mono muted">${s.proposed.toLocaleString()}</span> ` +
    `possible ${s.proposed === 1 ? "alarm" : "alarms"}. ` +
    `It woke someone <span class="mono" style="color:var(--confirmed)">${s.confirmed}</span> ${s.confirmed === 1 ? "time" : "times"}.`;

  const by = s.by_level || {};
  const reasons = [
    ["persistence", "showed up for a moment, then vanished"],
    ["cooldown", "the same event, already reported"],
    ["confidence", "too faint to take seriously"],
  ];
  const box = $("live-reasons");
  box.innerHTML = "";
  reasons.forEach(([key, words]) => {
    const count = by[key] || 0;
    const row = el("div", "finding");
    row.style.cursor = "default";
    const n = el("span", "mono", count.toLocaleString());
    n.style.cssText = "font-size:24px;width:74px;flex:none;letter-spacing:-0.02em;" +
      (count ? "" : "color:var(--faint);");
    row.appendChild(n);
    const w = el("span", "what", words);
    if (!count) w.style.color = "var(--faint)";
    row.appendChild(w);
    box.appendChild(row);
  });

  const events = $("live-events");
  events.innerHTML = "";
  (s.events || []).slice(0, 4).forEach((e, i) => {
    const row = el("div", "finding");
    row.style.cursor = "default";
    row.appendChild(el("span", "at", String(i + 1).padStart(2, "0")));
    const w = el("span", "what");
    w.appendChild(el("div", null, capitalise((e.hazard || "").replace(/_/g, " "))));
    const sub = el("div", "faint",
      `confirmed after ${e.supporting} sightings · ${e.seconds_to_confirm}s`);
    sub.style.cssText = "font-size:13px;padding-top:3px;";
    w.appendChild(sub);
    row.appendChild(w);
    events.appendChild(row);
  });
  if (!(s.events || []).length) {
    events.appendChild(el("p", "faint", "Nothing confirmed yet."));
  }
}

$("live-suppressed").addEventListener("change", (e) => {
  fetch(`/api/suppressed/${e.target.checked}`, { method: "POST" });
});

/* ----------------------------------------------------------------- train */

let trainingLoaded = false;

async function loadTraining() {
  if (trainingLoaded) return;
  trainingLoaded = true;
  const list = $("train-runs");
  list.innerHTML = "";
  try {
    const { runs } = await (await fetch("/api/training")).json();
    if (!runs.length) {
      list.appendChild(el("p", "faint", "No completed training runs on this machine."));
      return;
    }
    runs.forEach((run, i) => {
      const row = el("div", "choice-row");
      row.appendChild(el("span", "pick"));
      row.appendChild(el("span", "name", run.title));
      row.addEventListener("click", () => {
        list.querySelectorAll(".choice-row").forEach((r) => r.classList.remove("on"));
        row.classList.add("on");
        renderRun(run);
      });
      list.appendChild(row);
      if (i === 0) { row.classList.add("on"); renderRun(run); }
    });
  } catch (_) {
    list.appendChild(el("p", "faint", "Could not read the training history."));
  }
}

function renderRun(run) {
  const panel = $("train-detail");
  panel.innerHTML = "";

  panel.appendChild(Object.assign(el("div", "serif", run.title),
    { style: "font-size:30px;line-height:1.2;" }));
  panel.appendChild(Object.assign(el("p", "muted", run.summary),
    { style: "font-size:15px;line-height:1.65;margin:10px 0 0;max-width:600px;" }));

  const rule = el("div", "rule"); rule.style.margin = "26px 0"; panel.appendChild(rule);

  const scores = el("div");
  scores.style.cssText = "display:flex;gap:0;";
  (run.scores || []).forEach((s, i) => {
    const cell = el("div");
    cell.style.cssText = i === 0
      ? "flex-grow:1;padding-right:40px;"
      : "flex-grow:1;padding-left:40px;border-left:1px solid var(--line);";
    const fig = el("div", "mono", s.value);
    fig.style.cssText = `font-size:42px;line-height:1;letter-spacing:-0.03em;color:${s.honest ? "var(--confirmed)" : "var(--faint)"};`;
    cell.appendChild(fig);
    cell.appendChild(Object.assign(el("div", "muted", s.label),
      { style: "font-size:14px;padding-top:8px;" }));
    cell.appendChild(Object.assign(el("div", "faint", s.note),
      { style: "font-size:13px;line-height:1.5;padding-top:6px;" }));
    scores.appendChild(cell);
  });
  panel.appendChild(scores);

  if (run.curve && run.curve.length) {
    const rule2 = el("div", "rule"); rule2.style.margin = "26px 0"; panel.appendChild(rule2);
    panel.appendChild(Object.assign(el("div", "label", "Every stage"), {}));
    const wrap = el("div");
    wrap.style.cssText = "display:flex;align-items:flex-end;gap:3px;height:78px;margin-top:14px;";
    const max = Math.max(...run.curve.map((c) => c.value), 0.001);
    run.curve.forEach((c) => {
      const bar = el("div");
      bar.title = `stage ${c.epoch}: ${(c.value * 100).toFixed(1)}%`;
      bar.style.cssText = `flex-grow:1;height:${Math.max(2, (c.value / max) * 100)}%;background:${c.best ? "var(--confirmed)" : "var(--edge)"};`;
      wrap.appendChild(bar);
    });
    panel.appendChild(wrap);
    panel.appendChild(Object.assign(el("div", "faint", run.curve_note || ""),
      { style: "font-size:13px;line-height:1.55;margin-top:12px;max-width:600px;" }));
  }
}

/* --------------------------------------------------------------- helpers */

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function busy(text) {
  $("analyse-idle").style.display = "none";
  $("analyse-result").style.display = "none";
  $("analyse-busy").style.display = "block";
  $("busy-text").textContent = text;
}

function idle(text) {
  $("analyse-busy").style.display = "none";
  $("analyse-result").style.display = "none";
  $("analyse-idle").style.display = "block";
  $("analyse-idle").textContent = text;
}

/* A live public camera is somebody else's infrastructure. The credit and the
 * licence are shown because two of the four require it, and the age of the
 * newest image is shown because these cameras publish every few minutes — a
 * still picture must not be allowed to read as a crashed demo. */
function liveCredit(s) {
  if (!s.live) return s.attribution || "";
  const bits = [];
  const status = s.source_status || {};
  if (status.last_capture_time) {
    const age = Math.max(0, Date.now() / 1000 - status.last_capture_time);
    bits.push(age < 90
      ? "newest image just now"
      : `newest image ${Math.round(age / 60)} min ago`);
  }
  if (status.images_seen) {
    bits.push(`${status.images_seen} ` +
      `${status.images_seen === 1 ? "image" : "images"} so far`);
  }
  if (status.last_error) bits.push(status.last_error);
  const when = bits.length ? ` \u2014 ${bits.join(" \u00b7 ")}` : "";
  return `${s.attribution || ""} ${s.licence || ""}`.trim() + when;
}

const capitalise = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
               "Eight", "Nine", "Ten"];
const spell = (n) => (n <= 10 ? WORDS[n] : String(n));

show(pageForLocation());
history.replaceState({ page: current }, "", PATHS[current]);
loadClips();
