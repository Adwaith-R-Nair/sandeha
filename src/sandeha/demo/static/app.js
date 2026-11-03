"use strict";

// The page holds no simulation state. What it shows comes from the newest frame the server sent.

const SOLID = [126, 143, 134];
const TRACE_FRAMES = 300;
const MAX_STAMPS = 40; // Per pointer move, so one wild movement can't flood the queue.
const KEY_STEP = 0.02; // Domain units.
const RETRY_MS = 1000;
const SLIDER_NEVER = 100;
const SLIDER_HOLD_MS = 600; // How long the slider ignores the server after the user moves it.
const THUMB_HALF_PX = 7; // Half the height of the slider's thumb, in style.css.
const AI_COLOUR = "#7ad7ff";
const PHYSICS_COLOUR = "#ffb000";

const $ = (id) => document.getElementById(id);
const canvases = [$("smoke"), $("doubt")];
const contexts = canvases.map((c) => c.getContext("2d", { willReadFrequently: true }));
const trace = $("trace");
const traceContext = trace.getContext("2d");
const tauSlider = $("tau");

function hex(colour) {
  return [1, 3, 5].map((i) => parseInt(colour.slice(i, i + 2), 16));
}

function lookup(stops) {
  // 256 entries, by straight lines between evenly spaced colours.
  const colours = stops.map(hex);
  const table = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const place = (i / 255) * (colours.length - 1);
    const low = Math.min(Math.floor(place), colours.length - 2);
    const part = place - low;
    for (let c = 0; c < 3; c++) table[i * 3 + c] = colours[low][c] * (1 - part) + colours[low + 1][c] * part;
  }
  return table;
}

const SMOKE_TABLE = lookup(["#0a1c2b", "#3c5266", "#94a7b3", "#f4f7f6"]);
const DOUBT_TABLE = lookup(["#0a1c2b", "#3b2a5e", "#a53f62", "#ee812f", "#ffe28c"]);

// Drawing a frame

function paint(context, bytes, solid, table, res) {
  // Row 0 of the data is the bottom of the box, and the canvas's first row is the top.
  const canvas = context.canvas;
  if (canvas.width !== res) canvas.width = canvas.height = res;
  const image = context.createImageData(res, res);
  const out = image.data;
  for (let row = 0; row < res; row++) {
    const from = (res - 1 - row) * res;
    const to = row * res * 4;
    for (let col = 0; col < res; col++) {
      const at = from + col;
      const write = to + col * 4;
      if (solid[at]) {
        out[write] = SOLID[0];
        out[write + 1] = SOLID[1];
        out[write + 2] = SOLID[2];
      } else {
        const value = bytes[at] * 3;
        out[write] = table[value];
        out[write + 1] = table[value + 1];
        out[write + 2] = table[value + 2];
      }
      out[write + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

const history = [];

function drawTrace(scale, tau) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(trace.clientWidth * ratio);
  const height = Math.round(trace.clientHeight * ratio);
  if (trace.width !== width || trace.height !== height) {
    trace.width = width;
    trace.height = height;
  }
  const c = traceContext;
  c.clearRect(0, 0, width, height);
  // Inset by the thumb, so that the threshold line is level with the slider's thumb.
  const inset = THUMB_HALF_PX * ratio;
  const y = (value) => height - inset - Math.min(value / scale, 1) * ((SLIDER_NEVER - 1) / SLIDER_NEVER) * (height - 2 * inset);

  c.lineWidth = 2 * ratio;
  const step = width / (TRACE_FRAMES - 1);
  const start = width - (history.length - 1) * step;
  for (let i = 1; i < history.length; i++) {
    const before = history[i - 1];
    const now = history[i];
    if (before.doubt === null || now.doubt === null) continue;
    c.strokeStyle = now.source === "solver" ? PHYSICS_COLOUR : AI_COLOUR;
    c.beginPath();
    c.moveTo(start + (i - 1) * step, y(before.doubt));
    c.lineTo(start + i * step, y(now.doubt));
    c.stroke();
  }

  if (tau !== null) {
    c.strokeStyle = "#ffffff";
    c.lineWidth = ratio;
    c.setLineDash([6 * ratio, 4 * ratio]);
    c.beginPath();
    c.moveTo(0, y(tau) + 0.5);
    c.lineTo(width, y(tau) + 0.5);
    c.stroke();
    c.setLineDash([]);
  }
}

function digits(value) {
  if (value === null) return "none";
  return value.toFixed(value < 0.1 ? 4 : 3);
}

function show(header) {
  document.body.dataset.source = header.source;
  document.body.dataset.frames = String(Number(document.body.dataset.frames || 0) + 1);
  const speaking = header.source === "solver" ? "Physics solver in control" : "AI in control";
  if ($("source-text").textContent !== speaking) $("source-text").textContent = speaking;
  $("lamp-ai").className = header.source === "ai" ? "lamp lit" : "lamp";
  $("lamp-physics").className = header.source === "solver" ? "lamp lit" : "lamp";
  $("fps").textContent = header.fps.toFixed(1);
  $("share").textContent = `${Math.round(header.solver_share * 100)}%`;
  $("weights").hidden = header.weights !== "random";
  $("members").textContent = header.members < 5 ? `Where ${header.members} of 5 models disagree` : "Where 5 models disagree";
  $("scale-top").textContent = digits(header.doubt_scale);
  $("summary").textContent =
    `${header.members} neural models run this smoke. When they disagree too much, the physics solver takes over.`;

  const modeInput = document.querySelector(`input[name=mode][value=${header.mode}]`);
  if (!modeInput.checked && !pendingMode) modeInput.checked = true;
  if (pendingMode === header.mode) pendingMode = null;

  for (const button of document.querySelectorAll("[data-preset]")) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === header.preset));
  }

  doubtScale = header.doubt_scale;
  if (performance.now() > sliderHeldUntil) {
    const sliderValue = header.tau === null ? SLIDER_NEVER : Math.min(Math.round((header.tau / doubtScale) * (SLIDER_NEVER - 1)), SLIDER_NEVER - 1);
    tauSlider.value = String(sliderValue);
  }
  tauSlider.setAttribute("aria-valuetext", header.tau === null ? "never" : digits(header.tau));
  $("tau-text").textContent = `Doubt ${digits(header.doubt)}\nHands off ${header.tau === null ? "never" : `at ${digits(header.tau)}`}`;

  history.push({ doubt: header.doubt, source: header.source });
  if (history.length > TRACE_FRAMES) history.shift();
  drawTrace(doubtScale, header.tau);
}

// The newest frame wins, and drawing keeps to the browser's own pace.

let newest = null;
let doubtScale = 1;
let sliderHeldUntil = 0;
let pendingMode = null;

function decode(buffer) {
  const view = new DataView(buffer);
  const length = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
  const res = header.res;
  const start = 4 + length;
  return {
    header,
    density: new Uint8Array(buffer, start, res * res),
    doubt: new Uint8Array(buffer, start + res * res, res * res),
    solid: new Uint8Array(buffer, start + 2 * res * res, res * res),
  };
}

function render() {
  if (newest) {
    const { header, density, doubt, solid } = newest;
    paint(contexts[0], density, solid, SMOKE_TABLE, header.res);
    paint(contexts[1], doubt, solid, DOUBT_TABLE, header.res);
    show(header);
    newest = null;
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// The connection

let socket = null;

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    $("connection").hidden = true;
    pendingMode = null;
  };
  socket.onmessage = (event) => {
    newest = decode(event.data);
  };
  socket.onclose = () => {
    const notice = $("connection");
    notice.textContent = "The demo server isn't answering. Retrying every second. If this stays, start scripts/run_demo.py again.";
    notice.hidden = false;
    setTimeout(connect, RETRY_MS);
  };
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
window.sandehaSend = send; // For the browser tests.
connect();

// Drawing, with mouse, touch or pen on either window

function brushRadius() {
  return Number(document.querySelector("input[name=size]:checked").value);
}

function erasing() {
  return document.querySelector("input[name=tool]:checked").value === "erase";
}

function stamp(x, y) {
  send({ type: "brush", x, y, r: brushRadius(), erase: erasing() });
}

function toDomain(canvas, event) {
  const box = canvas.getBoundingClientRect();
  return { x: (event.clientX - box.left) / box.width, y: 1 - (event.clientY - box.top) / box.height };
}

function strokeTo(from, to) {
  const gap = brushRadius() * 0.5;
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const count = Math.min(Math.ceil(distance / gap), MAX_STAMPS);
  for (let i = 1; i <= count; i++) {
    stamp(from.x + ((to.x - from.x) * i) / count, from.y + ((to.y - from.y) * i) / count);
  }
}

function placeRings(point) {
  // The same ring on both windows, so that either one shows where a stroke lands.
  for (const canvas of canvases) {
    const ring = canvas.parentElement.querySelector(".ring");
    if (point === null) {
      ring.hidden = true;
      continue;
    }
    const size = canvas.clientWidth;
    const diameter = brushRadius() * 2 * size;
    ring.hidden = false;
    ring.style.width = ring.style.height = `${diameter}px`;
    ring.style.transform = `translate(${point.x * size - diameter / 2}px, ${(1 - point.y) * size - diameter / 2}px)`;
  }
}

let keyPoint = { x: 0.5, y: 0.5 };

for (const canvas of canvases) {
  let last = null;
  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    last = toDomain(canvas, event);
    keyPoint = last;
    stamp(last.x, last.y);
    placeRings(last);
  });
  canvas.addEventListener("pointermove", (event) => {
    const point = toDomain(canvas, event);
    placeRings(point);
    if (last === null) return;
    strokeTo(last, point);
    last = point;
    keyPoint = point;
  });
  const end = () => {
    last = null;
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => {
    if (last === null) placeRings(null);
  });

  canvas.addEventListener("keydown", (event) => {
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    if (moves[event.key]) {
      const [dx, dy] = moves[event.key];
      keyPoint = {
        x: Math.min(Math.max(keyPoint.x + dx * KEY_STEP, 0), 1),
        y: Math.min(Math.max(keyPoint.y + dy * KEY_STEP, 0), 1),
      };
      placeRings(keyPoint);
      event.preventDefault();
    } else if (event.key === " " || event.key === "Enter") {
      stamp(keyPoint.x, keyPoint.y);
      placeRings(keyPoint);
      event.preventDefault();
    }
  });
  canvas.addEventListener("focus", () => {
    if (canvas.matches(":focus-visible")) placeRings(keyPoint);
  });
  canvas.addEventListener("blur", () => placeRings(null));
}

// The controls

for (const input of document.querySelectorAll("input[name=mode]")) {
  input.addEventListener("change", () => {
    pendingMode = input.value;
    send({ type: "mode", mode: input.value });
  });
}

for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => {
    history.length = 0;
    send({ type: "preset", name: button.dataset.preset });
  });
}

$("reset").addEventListener("click", () => {
  history.length = 0;
  send({ type: "reset" });
});

tauSlider.addEventListener("input", () => {
  sliderHeldUntil = performance.now() + SLIDER_HOLD_MS;
  const value = Number(tauSlider.value);
  send({ type: "threshold", tau: value >= SLIDER_NEVER ? null : (value / (SLIDER_NEVER - 1)) * doubtScale });
});
