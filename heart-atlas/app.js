// Kalp Atlası — studio UI around the HeartAnatomy library.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HeartAnatomy, SYSTEMS, PARTS, PART_MAP, SYSTEM_MAP, PHASES, RHYTHMS } from './src/index.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const isPhone = matchMedia('(max-width: 900px)').matches;
const quality = params.get('q') || (isPhone ? 'medium' : 'high');

// ── renderer / scene ─────────────────────────────────────────────────────
const host = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: params.has('capture') });
renderer.setPixelRatio(Math.min(devicePixelRatio, isPhone ? 1.5 : 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.localClippingEnabled = true;
renderer.domElement.setAttribute('aria-label', 'Etkileşimli kalp anatomisi. Döndürmek için sürükleyin, yakınlaştırmak için kaydırın, bir yapıyı incelemek için tıklayın.');
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#f2f3f3');
const camera = new THREE.PerspectiveCamera(30, 1, 0.5, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.085;
controls.minDistance = 6;
controls.maxDistance = 140;
controls.rotateSpeed = 0.8;
{
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  pm.dispose();
}
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa1a8, 0.75));
const key = new THREE.DirectionalLight(0xfff3e8, 2.1);
key.position.set(-6, 10, 12);
scene.add(key);
const rim = new THREE.DirectionalLight(0xdfe9ff, 1.1);
rim.position.set(8, 4, -10);
scene.add(rim);
const fill = new THREE.DirectionalLight(0xffffff, 0.45);
fill.position.set(6, -6, 8);
scene.add(fill);

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// ── heart ────────────────────────────────────────────────────────────────
const heart = new HeartAnatomy({
  quality,
  palette: params.get('palette') === 'real' ? 'real' : 'atlas',
  heartRate: +(params.get('hr') || 72),
  rhythm: params.get('rhythm') || 'nsr',
});
window.heart = heart; // handy for experimenting with the API from the console
const center = new THREE.Vector3(...heart.lm.heartCenter).add(new THREE.Vector3(-0.6, 1.6, 0));

const VIEWS = {
  anterior: [0.14, 0.1, 1], posterior: [-0.12, 0.1, -1], left: [1, 0.14, 0.12],
  right: [-1, 0.12, 0.08], superior: [0.05, 1, 0.3], inferior: [0.1, -1, 0.42],
};
let baseDistance = isPhone ? 60 : 43;
camera.position.copy(center).addScaledVector(new THREE.Vector3(...VIEWS.anterior).normalize(), baseDistance * 1.35);
controls.target.copy(center);

// camera tween (spherical arc)
let tween = null;
function flyTo(dir, target = center, dist = null) {
  const from = camera.position.clone().sub(controls.target);
  const d0 = from.length();
  const d1 = dist ?? d0;
  tween = {
    t: 0, dur: 750,
    dirA: from.normalize(), dirB: new THREE.Vector3(...dir).normalize(),
    tA: controls.target.clone(), tB: target.clone(), dA: d0, dB: d1,
  };
}
function stepTween(dt) {
  if (!tween) return;
  tween.t = Math.min(1, tween.t + dt / tween.dur);
  const e = tween.t < 0.5 ? 4 * tween.t ** 3 : 1 - (-2 * tween.t + 2) ** 3 / 2;
  const q = new THREE.Quaternion().setFromUnitVectors(tween.dirA, tween.dirB);
  const dir = tween.dirA.clone().applyQuaternion(new THREE.Quaternion().slerp(q, e));
  if (tween.dirA.dot(tween.dirB) < -0.999) dir.copy(tween.dirA).lerp(tween.dirB, e).normalize();
  controls.target.lerpVectors(tween.tA, tween.tB, e);
  camera.position.copy(controls.target).addScaledVector(dir, tween.dA + (tween.dB - tween.dA) * e);
  if (tween.t >= 1) tween = null;
}
function setView(name) {
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  flyTo(VIEWS[name], center, baseDistance);
}
function focusPart(id) {
  const box = heart.visibleBounds([id]);
  if (box.isEmpty()) return;
  const s = box.getBoundingSphere(new THREE.Sphere());
  const dist = Math.max(14, (s.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.35);
  const dir = camera.position.clone().sub(controls.target).normalize();
  flyTo(dir.toArray(), s.center, dist);
}

// ── UI state ─────────────────────────────────────────────────────────────
const ui = { open: new Set(), speed: 1, sound: false, rotate: false };
const norm = (s) => s.toLocaleLowerCase('tr').normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/ı/g, 'i');
const partsOf = (sys) => [...heart.parts.values()].filter((p) => p.meta.system === sys);

function renderSystems() {
  const list = $('systemList');
  list.innerHTML = '';
  for (const sys of SYSTEMS) {
    const parts = partsOf(sys.id);
    if (!parts.length) continue;
    const on = heart.systemVisible[sys.id] !== false;
    const row = document.createElement('div');
    row.className = 'system-row' + (on ? ' enabled' : '') + (ui.open.has(sys.id) ? ' open' : '');
    row.innerHTML = `<button class="system-name"><i class="system-dot" style="background:${sys.color}"></i><span>${sys.tr}</span><span class="chev">▶</span></button>
      <span class="system-count">${parts.length}</span><input type="checkbox" class="switch" ${on ? 'checked' : ''} aria-label="${sys.tr} görünürlüğü">`;
    row.querySelector('.system-name').onclick = () => { ui.open.has(sys.id) ? ui.open.delete(sys.id) : ui.open.add(sys.id); renderSystems(); };
    row.querySelector('.switch').onchange = (e) => { heart.isolate(null); heart.setSystemVisible(sys.id, e.target.checked); setPresetButtons(null); renderSystems(); };
    list.appendChild(row);
    if (ui.open.has(sys.id)) {
      const pl = document.createElement('div');
      pl.className = 'part-list';
      for (const p of parts) {
        const vis = heart.partVisible[p.id] !== false;
        const r = document.createElement('div');
        r.className = 'part-row' + (vis ? '' : ' hidden') + (heart.selected === p.id ? ' selected' : '');
        r.innerHTML = `<button class="name">${p.meta.tr}</button><button class="eye" aria-label="Göster/gizle">${vis ? eyeIcon : eyeOffIcon}</button>`;
        r.querySelector('.name').onclick = () => selectPart(p.id, true);
        r.querySelector('.eye').onclick = () => { heart.setPartVisible(p.id, !vis); renderSystems(); };
        pl.appendChild(r);
      }
      list.appendChild(pl);
    }
  }
  const visible = [...heart.parts.values()].filter((p) => p.group.visible).length;
  $('visibleCount').textContent = `${visible} / ${heart.parts.size}`;
}
const eyeIcon = '<svg viewBox="0 0 24 24" width="15" height="15"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>';
const eyeOffIcon = '<svg viewBox="0 0 24 24" width="15" height="15"><path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.8 8.4 2 12 2 12s3.6 7 10 7a9.6 9.6 0 0 0 5.4-1.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';

const PRESETS = {
  all: { systems: { chambers: 1, valves: 1, subvalvular: 1, vessels: 1, coronary: 1, veins: 1, conduction: 1, pericardium: 0 }, opacity: 1, flow: false, electric: false },
  outer: { systems: { chambers: 1, valves: 0, subvalvular: 0, vessels: 1, coronary: 1, veins: 1, conduction: 0, pericardium: 0 }, opacity: 1, flow: false, electric: false },
  inner: { systems: { chambers: 1, valves: 1, subvalvular: 1, vessels: 1, coronary: 0, veins: 0, conduction: 1, pericardium: 0 }, opacity: 0.16, flow: true, electric: false },
  electric: { systems: { chambers: 1, valves: 1, subvalvular: 0, vessels: 0, coronary: 0, veins: 0, conduction: 1, pericardium: 0 }, opacity: 0.32, flow: false, electric: true },
};
function setPresetButtons(name) {
  document.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === name)));
}
function applyPreset(name) {
  const p = PRESETS[name];
  heart.isolate(null);
  for (const s in p.systems) heart.systemVisible[s] = !!p.systems[s];
  heart.partVisible = {};
  heart.applyVisibility();
  setOpacity(p.opacity);
  $('tFlow').checked = p.flow; heart.setFlowVisible(p.flow);
  $('tElectric').checked = p.electric; heart.setElectricView(p.electric);
  setPresetButtons(name);
  renderSystems();
}
function setOpacity(v) {
  $('opacity').value = v;
  heart.setWallOpacity(+v);
  $('opacityOut').textContent = Math.round(v * 100) + '%';
  fillRange($('opacity'));
}
function fillRange(el) {
  const f = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty('--fill', f + '%');
}

// ── selection / detail ───────────────────────────────────────────────────
function selectPart(id, focus = false) {
  heart.select(id);
  const p = heart.parts.get(id);
  if (!p) { $('detail').hidden = true; renderSystems(); return; }
  const m = p.meta, sys = SYSTEM_MAP.get(m.system);
  $('dAccent').style.background = sys.color;
  $('dSystem').textContent = sys.tr;
  $('dTitle').textContent = m.tr;
  $('dLatin').textContent = m.la;
  $('dDesc').textContent = m.desc;
  $('dFn').textContent = m.fn;
  $('dClin').textContent = m.clin;
  $('dEn').textContent = m.en;
  $('dId').textContent = m.id;
  $('dMembersTitle').textContent = `${sys.tr} · diğer yapılar`;
  const mem = $('dMembers');
  mem.innerHTML = '';
  for (const o of partsOf(m.system)) {
    if (o.id === id) continue;
    const b = document.createElement('button');
    b.textContent = o.meta.tr;
    b.onclick = () => selectPart(o.id, true);
    mem.appendChild(b);
  }
  $('isolateLabel').textContent = heart.isolated === id ? 'Tümünü göster' : 'Yalnız bunu göster';
  $('detail').hidden = false;
  $('searchPanel').hidden = true;
  if (focus) focusPart(id);
  renderSystems();
}
$('detailClose').onclick = () => { heart.select(null); $('detail').hidden = true; renderSystems(); };
$('btnIsolate').onclick = () => {
  const id = heart.selected;
  if (!id) return;
  if (heart.isolated === id) { heart.isolate(null); setView('anterior'); }
  else {
    heart.isolate(id);
    if (heart.parts.get(id).meta.system !== 'chambers') heart.setWallOpacity(1);
    focusPart(id);
  }
  selectPart(id);
};
$('btnFocus').onclick = () => heart.selected && focusPart(heart.selected);
$('btnHide').onclick = () => {
  if (!heart.selected) return;
  heart.setPartVisible(heart.selected, false);
  heart.select(null);
  $('detail').hidden = true;
  renderSystems();
};

// ── search ───────────────────────────────────────────────────────────────
let searchIdx = 0;
function openSearch() {
  $('searchPanel').hidden = false;
  $('searchInput').value = '';
  runSearch();
  $('searchInput').focus();
}
function runSearch() {
  const q = norm($('searchInput').value.trim());
  const res = PARTS.filter((p) => heart.parts.has(p.id) && (!q || [p.tr, p.en, p.la, p.id].some((s) => norm(s).includes(q)))).slice(0, 14);
  searchIdx = Math.min(searchIdx, Math.max(0, res.length - 1));
  const box = $('searchResults');
  box.innerHTML = '';
  res.forEach((p, i) => {
    const b = document.createElement('button');
    if (i === searchIdx) b.className = 'active';
    b.innerHTML = `<i class="system-dot" style="background:${SYSTEM_MAP.get(p.system).color}"></i><span>${p.tr}</span><small>${p.la}</small>`;
    b.onclick = () => { heart.isolate(null); heart.setPartVisible(p.id, true); heart.setSystemVisible(p.system, true); selectPart(p.id, true); };
    box.appendChild(b);
  });
  if (!res.length) box.innerHTML = '<p class="search-note">Sonuç yok.</p>';
  return res;
}
$('btnSearch').onclick = () => ($('searchPanel').hidden ? openSearch() : ($('searchPanel').hidden = true));
$('searchInput').oninput = () => { searchIdx = 0; runSearch(); };
$('searchInput').onkeydown = (e) => {
  const res = runSearch();
  if (e.key === 'ArrowDown') { searchIdx = Math.min(res.length - 1, searchIdx + 1); runSearch(); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { searchIdx = Math.max(0, searchIdx - 1); runSearch(); e.preventDefault(); }
  else if (e.key === 'Enter' && res[searchIdx]) $('searchResults').children[searchIdx].click();
};

// ── controls wiring ──────────────────────────────────────────────────────
document.querySelectorAll('[data-view]').forEach((b) => (b.onclick = () => setView(b.dataset.view)));
document.querySelectorAll('[data-preset]').forEach((b) => (b.onclick = () => applyPreset(b.dataset.preset)));
document.querySelectorAll('[data-palette]').forEach((b) => (b.onclick = () => {
  heart.setPalette(b.dataset.palette);
  document.querySelectorAll('[data-palette]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
}));
document.querySelectorAll('[data-speed]').forEach((b) => (b.onclick = () => setSpeed(+b.dataset.speed)));
function setSpeed(v) {
  ui.speed = v;
  heart.cycle.speed = v;
  document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', String(+x.dataset.speed === v)));
}
function setPaused(p) {
  heart.cycle.paused = p;
  $('icoPause').hidden = p;
  $('icoPlay').hidden = !p;
  $('btnPlay').setAttribute('aria-label', p ? 'Oynat' : 'Duraklat');
}
$('btnPlay').onclick = () => setPaused(!heart.cycle.paused);
$('hr').oninput = (e) => { heart.cycle.setHeartRate(+e.target.value); $('hrOut').textContent = e.target.value; fillRange(e.target); };
$('rhythm').onchange = (e) => { heart.cycle.setRhythm(e.target.value); $('rhythmName').textContent = RHYTHMS[e.target.value].tr; };
$('explode').oninput = (e) => {
  const v = +e.target.value;
  heart.setExplode(v);
  $('explodeOut').textContent = Math.round(v * 100);
  fillRange(e.target);
  // pull the camera back so the exploded inventory stays in frame
  const dir = camera.position.clone().sub(controls.target).normalize();
  flyTo(dir.toArray(), center, baseDistance * (1 + 0.75 * v));
};
$('btnAssemble').onclick = () => { heart.playAssembly(); $('explode').value = 0; $('explodeOut').textContent = 0; fillRange($('explode')); };
$('opacity').oninput = (e) => { setOpacity(+e.target.value); setPresetButtons(null); };
$('tFlow').onchange = (e) => heart.setFlowVisible(e.target.checked);
$('tElectric').onchange = (e) => heart.setElectricView(e.target.checked);
$('tDeform').onchange = (e) => heart.setDeformation(e.target.checked);
$('btnShowAll').onclick = () => applyPreset('all');
$('btnRotate').onclick = () => { ui.rotate = !ui.rotate; controls.autoRotate = ui.rotate; controls.autoRotateSpeed = 0.9; $('btnRotate').classList.toggle('active', ui.rotate); };
$('section').onchange = (e) => {
  const mode = e.target.value;
  $('sectionOffset').disabled = mode === 'none';
  $('sectionOffset').value = 0;
  fillRange($('sectionOffset'));
  heart.setSection(mode, 0);
  if (mode !== 'none') {
    if (heart.wallOpacity < 1) setOpacity(1);
    heart.setSystemVisible('pericardium', false);
    flyTo(heart.sectionViewDir().toArray(), center, baseDistance * 0.9);
  }
};
$('sectionOffset').oninput = (e) => { heart.setSection($('section').value, +e.target.value); fillRange(e.target); };
$('btnAbout').onclick = () => ($('about').hidden = !$('about').hidden);
$('aboutClose').onclick = () => ($('about').hidden = true);
$('btnLayers').onclick = () => $('layersPanel').classList.toggle('mobile-open');
for (const id of ['hr', 'explode', 'opacity', 'sectionOffset']) fillRange($(id));

addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type !== 'range' && e.target.type !== 'checkbox') {
    if (e.key === 'Escape') $('searchPanel').hidden = true;
    return;
  }
  if (e.target instanceof HTMLSelectElement) return;
  const k = e.key.toLowerCase();
  if (k === ' ') { setPaused(!heart.cycle.paused); e.preventDefault(); }
  else if (k === '/') { openSearch(); e.preventDefault(); }
  else if (k === 'escape') { $('searchPanel').hidden = true; $('about').hidden = true; $('detail').hidden = true; heart.select(null); }
  else if (k === 'e') { const v = heart.explodeTarget > 0.5 ? 0 : 1; $('explode').value = v; $('explode').oninput({ target: $('explode') }); }
  else if (k === 'b') $('btnAssemble').click();
  else if (k === 'r') $('btnRotate').click();
  else if (k === 'f' && heart.selected) focusPart(heart.selected);
  else if ('123456'.includes(k)) setView(Object.keys(VIEWS)[+k - 1]);
});

// ── picking: tap to select, hover for names ───────────────────────────────
let down = null, hoverAt = 0;
const canvas = renderer.domElement;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; tween = null; });
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  const quick = performance.now() - down.t < 450;
  down = null;
  if (moved > 6 || !quick) return;
  const id = pickAt(e);
  if (id) selectPart(id);
  else { heart.select(null); $('detail').hidden = true; renderSystems(); }
});
canvas.addEventListener('dblclick', (e) => { const id = pickAt(e); if (id) selectPart(id, true); });
canvas.addEventListener('pointermove', (e) => {
  if (down || e.pointerType === 'touch') return;
  const now = performance.now();
  if (now - hoverAt < 90) return;
  hoverAt = now;
  const id = pickAt(e);
  heart.hover(id);
  const tip = $('hoverTip');
  if (id) {
    const m = PART_MAP.get(id);
    tip.innerHTML = `${m.tr}<small>${m.la}</small>`;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + 'px';
    tip.style.top = e.clientY + 16 + 'px';
    tip.hidden = false;
  } else tip.hidden = true;
});
canvas.addEventListener('pointerleave', () => { $('hoverTip').hidden = true; heart.hover(null); });
function pickAt(e) {
  const r = canvas.getBoundingClientRect();
  const dpr = renderer.getPixelRatio();
  return heart.pick(renderer, scene, camera, (e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
}

// ── heart sounds (synthesised S1 / S2) ───────────────────────────────────
let actx = null;
function thump(freq, dur, gain) {
  if (!actx) return;
  const t = actx.currentTime;
  const o = actx.createOscillator();
  const g = actx.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(freq * 1.6, t);
  o.frequency.exponentialRampToValueAtTime(freq, t + dur * 0.4);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(actx.destination);
  o.start(t);
  o.stop(t + dur + 0.02);
}
heart.cycle.on('S1', () => ui.sound && ui.speed >= 0.25 && thump(48, 0.16, 0.9));
heart.cycle.on('S2', () => ui.sound && ui.speed >= 0.25 && thump(72, 0.11, 0.6));
$('btnSound').onclick = () => {
  ui.sound = !ui.sound;
  if (ui.sound && !actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  $('btnSound').setAttribute('aria-pressed', String(ui.sound));
};

// ── monitor chart ────────────────────────────────────────────────────────
const chart = $('chart');
const cx = chart.getContext('2d');
function drawChart() {
  const dpr = Math.min(devicePixelRatio, 2);
  const W = chart.clientWidth, H = chart.clientHeight;
  if (chart.width !== W * dpr || chart.height !== H * dpr) { chart.width = W * dpr; chart.height = H * dpr; }
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cx.clearRect(0, 0, W, H);
  const hist = heart.cycle.history;
  if (hist.length < 2) return;
  const tEnd = hist[hist.length - 1][0], span = 4000, t0 = tEnd - span;
  const X = (t) => ((t - t0) / span) * (W - 6) + 2;
  const small = H < 100;
  const lanes = small
    ? [{ y: 0, h: H * 0.55 }, { y: H * 0.55, h: H * 0.45 }, null]
    : [{ y: 0, h: H * 0.36 }, { y: H * 0.36, h: H * 0.4 }, { y: H * 0.76, h: H * 0.24 }];
  cx.strokeStyle = '#18253612';
  cx.lineWidth = 1;
  for (const l of lanes) if (l) { cx.beginPath(); cx.moveTo(0, l.y + l.h); cx.lineTo(W, l.y + l.h); cx.stroke(); }
  const line = (lane, idx, lo, hi, color, width = 1.5, dash = null) => {
    if (!lane) return;
    cx.beginPath();
    cx.strokeStyle = color;
    cx.lineWidth = width;
    cx.setLineDash(dash || []);
    let first = true;
    for (const h of hist) {
      if (h[0] < t0) continue;
      const y = lane.y + lane.h - 4 - ((h[idx] - lo) / (hi - lo)) * (lane.h - 8);
      if (first) { cx.moveTo(X(h[0]), y); first = false; } else cx.lineTo(X(h[0]), y);
    }
    cx.stroke();
    cx.setLineDash([]);
  };
  line(lanes[0], 1, -0.6, 1.3, '#1f7a5a', 1.6);
  line(lanes[1], 3, 0, 140, '#7a1f1a', 1.3, [3, 2]);
  line(lanes[1], 4, 0, 140, '#d08a2b', 1.3);
  line(lanes[1], 2, 0, 140, '#c8352b', 1.7);
  line(lanes[2], 5, 35, 140, '#3a5ec8', 1.7);
  cx.font = '600 9px Inter, sans-serif';
  cx.fillStyle = '#8a949e';
  cx.fillText('EKG II', 6, 11);
  cx.fillText('BASINÇ mmHg', 6, lanes[1].y + 11);
  cx.fillStyle = '#c8352b'; cx.fillText('LV', 76, lanes[1].y + 11);
  cx.fillStyle = '#7a1f1a'; cx.fillText('Aort', 92, lanes[1].y + 11);
  cx.fillStyle = '#d08a2b'; cx.fillText('LA', 118, lanes[1].y + 11);
  if (lanes[2]) { cx.fillStyle = '#8a949e'; cx.fillText('LV HACMİ mL', 6, lanes[2].y + 11); }
  // sweep cursor
  cx.fillStyle = '#263b48';
  cx.fillRect(W - 4, 0, 1.5, H);
}

function updateStats(st) {
  const hist = heart.cycle.history;
  const recent = hist.filter((h) => h[0] > st.t - 3000);
  if (!recent.length) return;
  let pMax = -1e9, pMin = 1e9, vMax = -1e9, vMin = 1e9;
  for (const h of recent) { pMax = Math.max(pMax, h[3]); pMin = Math.min(pMin, h[3]); vMax = Math.max(vMax, h[5]); vMin = Math.min(vMin, h[5]); }
  $('sHR').textContent = st.rhythm === 'vf' ? '—' : st.heartRate;
  $('sBP').textContent = `${Math.round(pMax)}/${Math.round(pMin)}`;
  $('sEF').textContent = Math.max(0, Math.round(((vMax - vMin) / vMax) * 100)) + '%';
  $('sVol').textContent = `${Math.round(vMax)}/${Math.round(vMin)}`;
  const ph = PHASES[st.phase];
  const v = (x) => (x > 0.15 ? 'açık' : 'kapalı');
  $('captionText').innerHTML = `<b>${ph.tr}</b> · Mitral ${v(st.mitral)} · Aort ${v(st.aortic)}`;
  $('captionText').title = ph.note;
}

// ── loop ─────────────────────────────────────────────────────────────────
let last = performance.now(), lastUi = 0, lastChart = 0, frames = 0, fpsT = 0, prevSinceQ = 1e9, ready = false;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(100, now - last);
  last = now;
  stepTween(dt);
  controls.update();
  if (ready) {
    const st = heart.update(dt);
    if (st.sinceQ < prevSinceQ - 50) {
      $('beatDot').classList.add('on'); $('statusDot').classList.add('beat');
      setTimeout(() => { $('beatDot').classList.remove('on'); $('statusDot').classList.remove('beat'); }, 140);
    }
    prevSinceQ = st.sinceQ;
    if (now - lastChart > 33) { drawChart(); lastChart = now; }
    if (now - lastUi > 120) { updateStats(st); lastUi = now; }
  }
  renderer.render(scene, camera);
  frames++;
  if (now - fpsT > 1000) {
    $('fps').textContent = ready ? `${frames} fps · ${(heart.triangles / 1000).toFixed(0)}k üçgen` : '';
    frames = 0;
    fpsT = now;
  }
}
requestAnimationFrame(frame);

// ── build ────────────────────────────────────────────────────────────────
(async () => {
  try {
    await heart.build((p, label) => {
      $('loadingBar').style.width = Math.round(p * 100) + '%';
      $('loadingStep').textContent = label + '…';
    });
  } catch (err) {
    console.error(err);
    $('loadingStep').textContent = 'Model üretilemedi: ' + err.message;
    return;
  }
  heart.triangles = [...heart.parts.values()].reduce((s, p) => s + p.meshes.reduce((a, m) => a + m.geometry.index.count / 3, 0), 0);
  scene.add(heart.root);
  $('loading').hidden = true;
  $('metaParts').textContent = heart.parts.size;
  $('metaSystems').textContent = SYSTEMS.length;
  $('qualityLabel').textContent = 'Kalite: ' + ({ low: 'düşük', medium: 'orta', high: 'yüksek', ultra: 'ultra' }[quality] || quality);
  $('rhythm').value = heart.cycle.rhythm;
  $('rhythmName').textContent = RHYTHMS[heart.cycle.rhythm].tr;
  $('hr').value = heart.cycle.heartRate; $('hrOut').textContent = heart.cycle.heartRate; fillRange($('hr'));
  document.querySelectorAll('[data-palette]').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.palette === heart.palette)));
  renderSystems();
  ready = true;
  if (!params.has('noassembly')) heart.playAssembly(4600);
  flyTo(VIEWS.anterior, center, baseDistance);
  window.__ready = true;
})();
