'use strict';

/* =====================================================
   CYBER VISION — script.js
   1. Настройки      4. Камера (getUserMedia)
   2. Элементы       5. Главный цикл анализа
   3. Журнал и UI    6. Лица, движение, рисование, снимки
   ===================================================== */

/* ---------- 1. НАСТРОЙКИ (можно менять) ---------- */
const CONFIG = {
  detectionIntervalMs: 120,   // как часто анализировать кадр (~8 раз в секунду)
  minFaceConfidence: 0.8,     // минимальная уверенность модели (0..1)
  faceLostDelayMs: 800,       // через сколько мс без лица писать "Face lost"
  motionW: 64,                // размер уменьшенного кадра для поиска движения
  motionH: 48,
  pixelThreshold: 28,         // насколько должна измениться яркость пикселя (0..255)
  motionRatio: 0.02,          // доля изменившихся пикселей = "есть движение"
  motionHoldMs: 800,          // сколько держать статус MOTION после последнего движения
  motionLogCooldownMs: 3000,  // не чаще чем раз в N мс писать "Motion detected" в журнал
  maxLogEntries: 100,
  maxSnapshots: 8,
  // V3
  objectEveryN: 3,           // объекты ищем на каждом 3-м анализе (модель тяжёлая)
  minObjectScore: 0.6,       // минимальная уверенность для объектов
  objectLogCooldownMs: 5000, // повторно писать про объект, если он пропал дольше N мс
  clipTailMs: 3000,          // сколько ещё писать после остановки движения
  clipMaxMs: 15000,          // максимальная длина клипа
  maxClips: 6,
};

/* ---------- 2. ЭЛЕМЕНТЫ СТРАНИЦЫ ---------- */
const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), overlay: $('overlay'), stage: $('stage'),
  offline: $('offline'), hud: $('hud'), hudRes: $('hudRes'),
  systemStatus: $('systemStatus'), clock: $('clock'), errorBox: $('errorBox'),
  faces: $('faces'), events: $('events'), motion: $('motion'),
  cameraState: $('cameraState'), modelState: $('modelState'), fps: $('fps'),
  meterFill: $('meterFill'), log: $('log'), logEmpty: $('logEmpty'),
  snaps: $('snaps'), snapsEmpty: $('snapsEmpty'),
  objects: $('objects'), recState: $('recState'), clips: $('clips'), clipsEmpty: $('clipsEmpty'),
  btnStart: $('btnStart'), btnStop: $('btnStop'), btnSnap: $('btnSnap'), btnClear: $('btnClear'),
};
const overlayCtx = el.overlay.getContext('2d');

// Маленький скрытый canvas — на нём считаем движение
const motionCanvas = document.createElement('canvas');
motionCanvas.width = CONFIG.motionW;
motionCanvas.height = CONFIG.motionH;
const motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });

/* ---------- Состояние программы ---------- */
const state = {
  stream: null, model: null, running: false, starting: false,
  rafId: 0, lastAnalysis: 0, analyzing: false, analysisErrors: 0,
  faces: 0, lastFaceSeen: 0,
  motion: false, lastMotionTime: 0, lastMotionLog: 0, prevGray: null,
  events: 0, frameCount: 0, fpsTimer: 0,
  snapUrls: [],
  objModel: null, objTick: 0, objects: [], objSeen: new Map(),
  recorder: null, recStart: 0, clipUrls: [],
};

/* ---------- 3. ЖУРНАЛ И ИНТЕРФЕЙС ---------- */
function timeNow() {
  return new Date().toLocaleTimeString('ru-RU', { hour12: false });
}

// type: info | success | motion | warn | error
function addLog(message, type = 'info', countAsEvent = false) {
  if (el.logEmpty.isConnected) el.logEmpty.remove();

  const li = document.createElement('li');
  li.className = type;
  const time = document.createElement('time');
  time.textContent = timeNow();
  const text = document.createElement('span');
  text.textContent = message;          // textContent — безопасно, без HTML-вставок
  li.append(time, text);
  el.log.prepend(li);                  // новые записи сверху

  while (el.log.children.length > CONFIG.maxLogEntries) el.log.lastElementChild.remove();

  if (countAsEvent) {
    state.events++;
    el.events.textContent = state.events;
    bump(el.events);
  }
}

function bump(node) {
  node.classList.remove('bump');
  void node.offsetWidth;               // перезапуск CSS-анимации
  node.classList.add('bump');
}

function setStatus(node, text, status) {
  node.textContent = text;
  node.dataset.state = status;         // CSS красит по data-state: on / off / warn / bad
}

function showError(message) {
  el.errorBox.textContent = message;
  el.errorBox.hidden = false;
  addLog(message, 'error');
}
function hideError() { el.errorBox.hidden = true; }

function updateButtons() {
  el.btnStart.disabled = state.running || state.starting;
  el.btnStop.disabled = !state.running;
  el.btnSnap.disabled = !state.running;
}

function clearLog() {
  el.log.replaceChildren(el.logEmpty);
  state.events = 0;
  el.events.textContent = '0';
}

setInterval(() => { el.clock.textContent = timeNow(); }, 1000);
el.clock.textContent = timeNow();

/* ---------- Загрузка модели распознавания лиц ---------- */
async function loadModel() {
  setStatus(el.modelState, 'LOADING', 'warn');
  try {
    if (typeof blazeface === 'undefined') throw new Error('Библиотека BlazeFace не загрузилась (нет интернета?)');
    state.model = await blazeface.load();
    setStatus(el.modelState, 'READY', 'on');
    addLog('AI model loaded', 'success');
  } catch (err) {
    console.error(err);
    setStatus(el.modelState, 'UNAVAILABLE', 'bad');
    addLog('Модель лиц недоступна: ' + err.message + '. Движение будет работать.', 'warn');
  }
}

/* ---------- V3: модель распознавания объектов (COCO-SSD, ~80 классов) ---------- */
async function loadObjectModel() {
  try {
    if (typeof cocoSsd === 'undefined') throw new Error('COCO-SSD не загрузилась');
    state.objModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    addLog('Object model loaded', 'success');
  } catch (err) {
    console.error(err);
    addLog('Объекты недоступны: ' + err.message, 'warn');
  }
}

/* ---------- 4. КАМЕРА ---------- */
// Понятные сообщения для типичных ошибок getUserMedia
function describeCameraError(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Доступ к камере запрещён. Разрешите камеру в настройках сайта (значок замка в адресной строке) и нажмите START CAMERA снова.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'Камера не найдена. Подключите камеру и попробуйте снова.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Камера занята другим приложением (Zoom, Teams и т.п.). Закройте его и попробуйте снова.';
    case 'OverconstrainedError':
      return 'Камера не поддерживает запрошенные параметры.';
    case 'SecurityError':
      return 'Браузер заблокировал камеру из соображений безопасности. Откройте страницу через http://localhost (Live Server).';
    default:
      return 'Не удалось запустить камеру: ' + ((err && err.message) || err);
  }
}

async function startCamera() {
  if (state.running || state.starting) return;
  hideError();

  // navigator.mediaDevices есть только в безопасном контексте (https или localhost)
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Этот браузер или адрес не поддерживает доступ к камере. Используйте Chrome/Edge/Firefox и адрес http://localhost или https.');
    return;
  }

  state.starting = true;
  updateButtons();
  setStatus(el.systemStatus, 'STARTING', 'warn');

  try {
    // Запрашиваем у браузера видеопоток. Появится окно "Разрешить камеру?"
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });

    el.video.srcObject = state.stream;   // подключаем поток к <video>
    await el.video.play();
    if (!el.video.videoWidth) {          // ждём, пока станет известен размер кадра
      await new Promise((res) => el.video.addEventListener('loadedmetadata', res, { once: true }));
    }

    const w = el.video.videoWidth, h = el.video.videoHeight;
    el.overlay.width = w;                // canvas поверх видео — в том же разрешении
    el.overlay.height = h;
    el.stage.style.aspectRatio = w + ' / ' + h;
    el.hudRes.textContent = w + '×' + h;

    // Если камеру отключили физически — корректно остановимся
    state.stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (state.running) { showError('Камера отключена.'); stopCamera(true); }
    });

    state.running = true;
    state.prevGray = null;
    state.faces = 0;
    el.stage.classList.add('live');
    el.offline.classList.add('hidden');
    el.hud.hidden = false;
    setStatus(el.systemStatus, 'ONLINE', 'on');
    setStatus(el.cameraState, 'ACTIVE', 'on');
    addLog('Camera started (' + w + '×' + h + ')', 'success');
    updateButtons();
    state.rafId = requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    stopStream();
    setStatus(el.systemStatus, 'ERROR', 'bad');
    showError(describeCameraError(err));
  } finally {
    state.starting = false;
    updateButtons();
  }
}

function stopStream() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());  // выключаем камеру
  state.stream = null;
  el.video.srcObject = null;
}

function stopCamera(silent = false) {
  cancelAnimationFrame(state.rafId);
  state.running = false;
  stopRecording();
  state.objects = []; state.objSeen.clear(); el.objects.textContent = '0';
  stopStream();
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);

  state.faces = 0; state.motion = false; state.prevGray = null;
  el.faces.textContent = '0';
  el.stage.classList.remove('live', 'motion');
  el.offline.classList.remove('hidden');
  el.hud.hidden = true;
  el.meterFill.style.width = '0%';
  el.fps.textContent = '0';
  setStatus(el.motion, 'NONE', 'off');
  setStatus(el.cameraState, 'OFF', 'off');
  setStatus(el.systemStatus, silent ? 'ERROR' : 'OFFLINE', silent ? 'bad' : 'off');
  if (!silent) addLog('Camera stopped', 'info');
  updateButtons();
}

/* ---------- 5. ГЛАВНЫЙ ЦИКЛ ---------- */
function loop() {
  if (!state.running) return;
  state.rafId = requestAnimationFrame(loop);   // браузер вызовет loop перед следующей отрисовкой

  const now = performance.now();
  // FPS цикла
  state.frameCount++;
  if (now - state.fpsTimer >= 1000) {
    el.fps.textContent = state.frameCount;
    state.frameCount = 0;
    state.fpsTimer = now;
  }

  // Анализируем кадр не на каждом проходе — это тяжёлая операция
  if (!state.analyzing && now - state.lastAnalysis >= CONFIG.detectionIntervalMs) {
    state.lastAnalysis = now;
    analyzeFrame(now);
  }
}

async function analyzeFrame(now) {
  state.analyzing = true;
  try {
    updateMotion(measureMotion(), now);

    let faces = [];
    if (state.model) {
      const predictions = await state.model.estimateFaces(el.video, false);
      faces = predictions
        .map((p) => ({
          x1: p.topLeft[0], y1: p.topLeft[1],
          x2: p.bottomRight[0], y2: p.bottomRight[1],
          score: Array.isArray(p.probability) ? p.probability[0] : p.probability,
          landmarks: p.landmarks,
        }))
        .filter((f) => f.score >= CONFIG.minFaceConfidence);
    }
    // V3: объекты (телефон, чашка, ноутбук и т.д.)
    if (state.objModel && state.running && ++state.objTick % CONFIG.objectEveryN === 0) {
      const found = await state.objModel.detect(el.video);
      state.objects = found
        .filter((o) => o.score >= CONFIG.minObjectScore)
        .map((o) => ({ label: o.class, score: o.score, x: o.bbox[0], y: o.bbox[1], w: o.bbox[2], h: o.bbox[3] }));
      updateObjects(state.objects, now);
    }
    if (!state.running) return;     // камеру могли остановить, пока шёл анализ
    updateFaces(faces, now);
    drawOverlay(faces);
    state.analysisErrors = 0;
  } catch (err) {
    console.error(err);
    if (++state.analysisErrors >= 5) {
      state.model = null;
      setStatus(el.modelState, 'ERROR', 'bad');
      addLog('Ошибка распознавания лиц, модель отключена', 'error');
    }
  } finally {
    state.analyzing = false;
  }
}

/* ---------- 6. ЛИЦА, ДВИЖЕНИЕ, РИСОВАНИЕ ---------- */
function updateFaces(faces, now) {
  const count = faces.length;
  if (count > 0) {
    state.lastFaceSeen = now;
    if (count !== state.faces) {
      const msg = state.faces === 0
        ? (count === 1 ? 'Face detected' : count + ' faces detected')
        : 'Faces in frame: ' + count;
      addLog(msg, 'success', true);
      state.faces = count;
      el.faces.textContent = count;
      bump(el.faces);
    }
  } else if (state.faces > 0 && now - state.lastFaceSeen > CONFIG.faceLostDelayMs) {
    addLog('Face lost', 'info');
    state.faces = 0;
    el.faces.textContent = '0';
  }
}

// Движение: сравниваем текущий кадр с предыдущим в оттенках серого
function measureMotion() {
  const { motionW: w, motionH: h } = CONFIG;
  motionCtx.drawImage(el.video, 0, 0, w, h);          // уменьшаем кадр
  const data = motionCtx.getImageData(0, 0, w, h).data; // RGBA для каждого пикселя
  const gray = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
  }
  let changed = 0;
  if (state.prevGray) {
    for (let p = 0; p < gray.length; p++) {
      if (Math.abs(gray[p] - state.prevGray[p]) > CONFIG.pixelThreshold) changed++;
    }
  }
  state.prevGray = gray;
  return changed / gray.length;       // доля изменившихся пикселей (0..1)
}

function updateMotion(ratio, now) {
  el.meterFill.style.width = Math.min(100, (ratio / 0.15) * 100) + '%';
  if (ratio > CONFIG.motionRatio) state.lastMotionTime = now;

  const active = now - state.lastMotionTime < CONFIG.motionHoldMs;
  if (active && !state.motion && now - state.lastMotionLog > CONFIG.motionLogCooldownMs) {
    addLog('Motion detected', 'motion', true);
    state.lastMotionLog = now;
  }
  if (active !== state.motion) {
    state.motion = active;
    setStatus(el.motion, active ? 'DETECTED' : 'NONE', active ? 'bad' : 'off');
    el.stage.classList.toggle('motion', active);
  }

  // V3: клип пишется, пока есть движение, и ещё немного после него
  if (state.motion && !state.recorder) startRecording(now);
  else if (state.recorder && (now - state.lastMotionTime > CONFIG.clipTailMs || now - state.recStart > CONFIG.clipMaxMs)) stopRecording();
}

function updateObjects(list, now) {
  el.objects.textContent = list.length;
  list.forEach((o) => {
    const seen = state.objSeen.get(o.label);
    if (seen === undefined || now - seen > CONFIG.objectLogCooldownMs) {
      addLog('Object detected: ' + o.label, 'info', true);
    }
    state.objSeen.set(o.label, now);
  });
}

/* ---------- V3: запись видеоклипов (MediaRecorder) ---------- */
function startRecording(now) {
  if (!state.stream || typeof MediaRecorder === 'undefined') return;
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
    .find((t) => MediaRecorder.isTypeSupported(t));
  try {
    const rec = new MediaRecorder(state.stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => saveClip(chunks, rec.mimeType);
    rec.start();
    state.recorder = rec;
    state.recStart = now;
    setStatus(el.recState, 'REC', 'bad');
    addLog('Recording started', 'motion');
  } catch (err) {
    console.error(err);
    addLog('Запись недоступна: ' + err.message, 'warn');
  }
}

function stopRecording() {
  const rec = state.recorder;
  if (!rec) return;
  state.recorder = null;
  setStatus(el.recState, 'IDLE', 'off');
  if (rec.state !== 'inactive') rec.stop();    // после stop сработает onstop -> saveClip
}

function saveClip(chunks, mime) {
  if (!chunks.length) return;
  const blob = new Blob(chunks, { type: mime || 'video/webm' });
  const ext = (mime || '').includes('mp4') ? 'mp4' : 'webm';
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'clip-' + Date.now() + '.' + ext;
  a.textContent = timeNow() + ' · clip.' + ext + ' · ' + Math.round(blob.size / 1024) + ' KB (скачать)';
  if (el.clipsEmpty.isConnected) el.clipsEmpty.remove();
  el.clips.prepend(a);
  state.clipUrls.unshift(url);
  while (state.clipUrls.length > CONFIG.maxClips) {
    URL.revokeObjectURL(state.clipUrls.pop());
    el.clips.lastElementChild.remove();
  }
  addLog('Clip saved', 'success', true);
}

// Рисуем рамки. Видео в CSS отзеркалено, поэтому X считаем зеркально (текст остаётся читаемым).
function drawOverlay(faces) {
  const W = el.overlay.width, H = el.overlay.height;
  const k = W / 640;                                  // масштаб линий и шрифта
  const mx = (x) => W - x;
  overlayCtx.clearRect(0, 0, W, H);
  overlayCtx.lineWidth = 2 * k;
  overlayCtx.font = (13 * k) + 'px ui-monospace, Consolas, monospace';

  // V3: рамки объектов (янтарные, пунктиром)
  overlayCtx.save();
  overlayCtx.setLineDash([6 * k, 4 * k]);
  overlayCtx.strokeStyle = '#ffb02e';
  overlayCtx.fillStyle = '#ffb02e';
  state.objects.forEach((o) => {
    const ox = mx(o.x + o.w);
    overlayCtx.strokeRect(ox, o.y, o.w, o.h);
    overlayCtx.fillText(o.label + ' ' + Math.round(o.score * 100) + '%', ox, o.y + o.h + 14 * k);
  });
  overlayCtx.restore();

  faces.forEach((f, i) => {
    const x = mx(f.x2), y = f.y1, w = f.x2 - f.x1, h = f.y2 - f.y1;
    const c = Math.min(w, h) * 0.22;                   // длина уголков рамки

    overlayCtx.strokeStyle = '#35ff8a';
    overlayCtx.shadowColor = '#35ff8a';
    overlayCtx.shadowBlur = 10;
    overlayCtx.beginPath();
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([cx, cy, dx, dy]) => {
      overlayCtx.moveTo(cx + dx * c, cy);
      overlayCtx.lineTo(cx, cy);
      overlayCtx.lineTo(cx, cy + dy * c);
    });
    overlayCtx.stroke();
    overlayCtx.shadowBlur = 0;

    overlayCtx.fillStyle = '#35ff8a';
    overlayCtx.fillText('FACE ' + String(i + 1).padStart(2, '0') + '  ' + Math.round(f.score * 100) + '%', x, Math.max(14 * k, y - 8 * k));

    overlayCtx.fillStyle = '#00e5ff';                  // точки: глаза, нос, рот, уши
    (f.landmarks || []).forEach(([lx, ly]) => {
      overlayCtx.beginPath();
      overlayCtx.arc(mx(lx), ly, 2.5 * k, 0, Math.PI * 2);
      overlayCtx.fill();
    });
  });
}

/* ---------- Снимок ---------- */
function takeSnapshot() {
  if (!state.running) return;
  const w = el.video.videoWidth, h = el.video.videoHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');

  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);              // как на экране — зеркально
  ctx.drawImage(el.video, 0, 0, w, h);
  ctx.restore();
  ctx.drawImage(el.overlay, 0, 0, w, h);              // рамки поверх

  const stamp = new Date().toLocaleString('ru-RU');
  ctx.font = (14 * (w / 640)) + 'px monospace';
  ctx.fillStyle = '#00e5ff';
  ctx.fillText('CYBER VISION  ' + stamp, 12, h - 12);

  c.toBlob((blob) => {
    if (!blob) { showError('Не удалось создать снимок.'); return; }
    const url = URL.createObjectURL(blob);
    const name = 'cyber-vision-' + Date.now() + '.png';

    if (el.snapsEmpty.isConnected) el.snapsEmpty.remove();
    const a = document.createElement('a');
    a.href = url; a.download = name; a.title = 'Скачать ' + name;
    const img = document.createElement('img');
    img.src = url; img.alt = 'Снимок ' + stamp;
    a.append(img);
    el.snaps.prepend(a);

    state.snapUrls.unshift(url);
    while (el.snaps.querySelectorAll('a').length > CONFIG.maxSnapshots) {
      el.snaps.querySelector('a:last-of-type').remove();
      URL.revokeObjectURL(state.snapUrls.pop());
    }
    addLog('Snapshot saved (click thumbnail to download)', 'success', true);
  }, 'image/png');
}

/* ---------- Подключаем кнопки и запускаем ---------- */
el.btnStart.addEventListener('click', startCamera);
el.btnStop.addEventListener('click', () => stopCamera());
el.btnSnap.addEventListener('click', takeSnapshot);
el.btnClear.addEventListener('click', clearLog);

document.querySelectorAll('.nav-link').forEach((link) => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((l) => l.classList.remove('active'));
    link.classList.add('active');
  });
});

window.addEventListener('beforeunload', stopStream);
updateButtons();
addLog('System ready. Press START CAMERA.', 'info');
loadModel();
loadObjectModel();
