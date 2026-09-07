/* ibm-1 hero: the substrate as a node graph, and its materializations as views.
   expects window.IBM_GRAPH (site/data/graph.js) and the three.js UMD global. */
(function () {
  'use strict';
  const G = window.IBM_GRAPH;
  if (!G || !window.THREE) return;

  // ------------------------------------------------------------ dom ----
  const hero = document.getElementById('hero');
  const canvas = document.getElementById('hero-canvas');
  const svg = document.getElementById('hero-leaders');
  const ring = document.getElementById('ring');
  const sel = document.getElementById('selected');
  const selTitle = document.getElementById('sel-title');
  const selGroup = document.getElementById('sel-group');
  const selDoc = document.getElementById('sel-doc');
  const selMeta = document.getElementById('sel-meta');
  const inCol = document.getElementById('inputs');
  const outCol = document.getElementById('outputs');
  const closeBtn = document.getElementById('sel-close');
  const hint = document.getElementById('hero-hint');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------- geometry ----
  // surface RAS (x right, y anterior, z superior) -> three (y up, viewer faces the subject)
  const N = G.nodes.length;
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const size = new Float32Array(N);
  const cx = G.center;
  const toWorld = (p) => [-(p[0] - cx[0]), p[2] - cx[2], p[1] - cx[1]];
  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  const GROUP_SIZE = { cortex: 2.6, thalamus: 3.2, basal_ganglia: 3.2, hippocampus: 3.2, amygdala: 3.2, brainstem: 3.6, cerebellum: 3.4, spinal: 3.4, retina: 2.4, cochlea: 2.4, eeg: 4.2, meg: 3.2 };
  const REST = { cortex: 0.78, thalamus: 0.8, basal_ganglia: 0.8, hippocampus: 0.8, amygdala: 0.8, brainstem: 0.8, cerebellum: 0.72, spinal: 0.55, retina: 0.5, cochlea: 0.5, eeg: 0.3, meg: 0.18 };
  const group = new Array(N);
  for (let i = 0; i < N; i++) {
    const n = G.nodes[i];
    const w = toWorld(n.p);
    pos[3 * i] = w[0]; pos[3 * i + 1] = w[1]; pos[3 * i + 2] = w[2];
    const c = hex(n.c);
    col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    size[i] = GROUP_SIZE[n.g] || 2.6;
    group[i] = n.g;
  }
  const wCur = new Float32Array(N);
  const wTgt = new Float32Array(N);
  const rest = new Float32Array(N);
  for (let i = 0; i < N; i++) { rest[i] = REST[group[i]] || 0.6; wCur[i] = rest[i]; wTgt[i] = rest[i]; }

  const E = G.edges.length;
  const epos = new Float32Array(E * 6);
  const ecol = new Float32Array(E * 6);
  const ew = new Float32Array(E * 2);
  for (let e = 0; e < E; e++) {
    const [a, b] = G.edges[e];
    for (let k = 0; k < 3; k++) {
      epos[6 * e + k] = pos[3 * a + k]; epos[6 * e + 3 + k] = pos[3 * b + k];
      ecol[6 * e + k] = col[3 * a + k]; ecol[6 * e + 3 + k] = col[3 * b + k];
    }
  }

  // ---------------------------------------------------------- three ----
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 10, 3000);
  const pivot = new THREE.Group();
  scene.add(pivot);

  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  pgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  pgeo.setAttribute('size', new THREE.BufferAttribute(size, 1));
  pgeo.setAttribute('w', new THREE.BufferAttribute(wCur, 1));
  const pmat = new THREE.ShaderMaterial({
    uniforms: { pr: { value: renderer.getPixelRatio() } },
    vertexShader: `
      attribute float size; attribute float w; attribute vec3 color;
      varying vec3 vC; varying float vW; varying float vD; uniform float pr;
      void main(){
        vC = color; vW = w;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vD = clamp((-mv.z - 260.0) / 260.0, 0.0, 1.0);
        gl_PointSize = size * (0.7 + 1.1 * w) * (420.0 / -mv.z) * pr;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vC; varying float vW; varying float vD;
      void main(){
        vec2 q = gl_PointCoord - 0.5; float d = length(q);
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.3, d);
        float lum = mix(0.28, 1.0, vW);
        vec3 c = mix(vC * lum, vec3(1.0), 0.12 * vW * (1.0 - smoothstep(0.0, 0.25, d)));
        float alpha = a * mix(0.10, 1.0, vW) * mix(1.0, mix(0.45, 0.8, vW), vD);
        gl_FragColor = vec4(c, alpha);
      }`,
    transparent: true, depthWrite: false, depthTest: true,
  });
  const points = new THREE.Points(pgeo, pmat);
  pivot.add(points);

  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute('position', new THREE.BufferAttribute(epos, 3));
  lgeo.setAttribute('color', new THREE.BufferAttribute(ecol, 3));
  lgeo.setAttribute('w', new THREE.BufferAttribute(ew, 1));
  const lmat = new THREE.ShaderMaterial({
    vertexShader: `
      attribute float w; attribute vec3 color; varying vec3 vC; varying float vW; varying float vD;
      void main(){ vC = color; vW = w; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vD = clamp((-mv.z - 260.0) / 260.0, 0.0, 1.0); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `
      varying vec3 vC; varying float vW; varying float vD;
      void main(){ float alpha = mix(0.03, 0.42, vW) * mix(1.0, 0.4, vD);
        gl_FragColor = vec4(mix(vC, vec3(1.0), 0.15 * vW), alpha); }`,
    transparent: true, depthWrite: false, depthTest: true,
  });
  const lines = new THREE.LineSegments(lgeo, lmat);
  pivot.add(lines);

  const state = { selected: null, hover: null };

  // ---------------------------------------------------------- orbit ----
  const orbit = { az: -0.65, el: 0.32, dist: 430, tAz: -0.65, tEl: 0.32, tDist: 430, vAz: 0, vEl: 0, dragging: false, lastX: 0, lastY: 0, idle: 0 };
  function applyCamera() {
    const { az, el, dist } = orbit;
    camera.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az));
    camera.lookAt(0, 0, 0);
  }
  function resize() {
    const w = hero.clientWidth, h = hero.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const fit = Math.max(1, 1.15 / Math.min(1, w / h));   // portrait: pull back
    orbit.tDist = (state.selected ? 470 : 430) * fit * (w < 720 ? 1.15 : 1);
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    layoutRing();
  }
  const onDown = (x, y) => { orbit.dragging = true; orbit.lastX = x; orbit.lastY = y; orbit.vAz = 0; orbit.vEl = 0; hero.classList.add('dragging'); };
  const onMove = (x, y) => {
    if (!orbit.dragging) return;
    const dx = x - orbit.lastX, dy = y - orbit.lastY;
    orbit.lastX = x; orbit.lastY = y;
    orbit.tAz -= dx * 0.006; orbit.tEl = Math.max(-1.3, Math.min(1.3, orbit.tEl + dy * 0.006));
    orbit.vAz = -dx * 0.006; orbit.vEl = dy * 0.006; orbit.idle = 0;
  };
  const onUp = () => { orbit.dragging = false; hero.classList.remove('dragging'); };
  canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); onDown(e.clientX, e.clientY); });
  canvas.addEventListener('pointermove', (e) => onMove(e.clientX, e.clientY));
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', (e) => {
    if (!e.ctrlKey && Math.abs(e.deltaY) < 1) return;
    if (!e.ctrlKey) return;               // plain wheel scrolls the page
    e.preventDefault();
    orbit.tDist = Math.max(220, Math.min(900, orbit.tDist * (1 + e.deltaY * 0.0015)));
  }, { passive: false });

  // ---------------------------------------------------- annotations ----
  const MATS = G.materializations;
  const byId = Object.fromEntries(MATS.map((m) => [m.id, m]));
  const expand = (ranges) => { const out = []; for (const [a, b] of ranges) for (let i = a; i < b; i++) out.push(i); return out; };
  MATS.forEach((m) => { m._hot = expand(m.hot); m._inv = expand(m.involved); if (!m._hot.length) m._hot = m._inv; });

  const v3 = new THREE.Vector3();
  const up = new THREE.Vector3();
  function silhouette(t) {
    // where the brain's outline is, seen from the ring at angle t
    const [cx0, cy0] = project(cx);
    up.set(0, 1, 0).applyQuaternion(camera.quaternion);
    const w = toWorld(cx); v3.set(w[0] + up.x * 78, w[1] + up.y * 78, w[2] + up.z * 78).project(camera);
    const px = (v3.x * 0.5 + 0.5) * hero.clientWidth, py = (-v3.y * 0.5 + 0.5) * hero.clientHeight;
    const r = Math.hypot(px - cx0, py - cy0);
    return [cx0 + r * 1.02 * Math.cos(t), cy0 + r * 0.98 * Math.sin(t), 0];
  }
  function project(p) {                       // RAS mm -> screen px
    const w = toWorld(p);
    v3.set(w[0], w[1], w[2]).applyMatrix4(pivot.matrixWorld).project(camera);
    return [(v3.x * 0.5 + 0.5) * hero.clientWidth, (-v3.y * 0.5 + 0.5) * hero.clientHeight, v3.z];
  }

  // the ring: one label per materialization, grouped by family around the circle
  const GROUP_ORDER = ['electrophysiology', 'decoding', 'stimulation', 'state', 'slow', 'hemodynamic', 'surrogate'];
  const ordered = GROUP_ORDER.flatMap((g) => MATS.filter((m) => m.group === g)).concat(MATS.filter((m) => !GROUP_ORDER.includes(m.group)));
  const ringItems = [];
  ordered.forEach((m, i) => {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'ring-item';
    el.dataset.id = m.id;
    const first = i === 0 || ordered[i - 1].group !== m.group;
    el.innerHTML = (first ? `<span class="ring-group">${m.group_title}</span>` : '') + `<span class="ring-label">${m.id}</span>`;
    el.addEventListener('click', () => select(m.id));
    el.addEventListener('mouseenter', () => { if (!state.selected) preview(m.id); });
    el.addEventListener('mouseleave', () => { if (!state.selected) preview(null); });
    el.addEventListener('focus', () => { if (!state.selected) preview(m.id); });
    el.addEventListener('blur', () => { if (!state.selected) preview(null); });
    ring.appendChild(el);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.setAttribute('class', 'leader');
    svg.appendChild(line);
    ringItems.push({ m, el, line, x: 0, y: 0, side: 'l' });
  });

  function layoutRing() {
    const w = hero.clientWidth, h = hero.clientHeight;
    const compact = w < 760;
    hero.classList.toggle('compact', compact);
    if (compact) return;
    const n = ringItems.length;
    const rx = Math.min(w * 0.5 - 200, h * 0.95), ry = h * 0.5 - 70;
    const cy = h * 0.5, cxp = w * 0.5;
    // labels are wide and short, so space them by a metric that charges
    // horizontal travel more than vertical: the top and bottom arcs thin out
    // on their own and the sides fill evenly.
    const K = 2.4, S = 720, cum = [0];
    for (let k = 1; k <= S; k++) {
      const a0 = (k - 1) / S * Math.PI * 2, a1 = k / S * Math.PI * 2;
      const dx = rx * (Math.cos(a1) - Math.cos(a0)), dy = ry * (Math.sin(a1) - Math.sin(a0));
      cum.push(cum[k - 1] + Math.hypot(dx * K, dy));
    }
    const total = cum[S];
    ringItems.forEach((it, i) => {
      const target = (i / n) * total;
      let k = 0; while (k < S && cum[k + 1] < target) k++;
      const t = -Math.PI / 2 + (k / S) * Math.PI * 2;
      const c = Math.cos(t), sn = Math.sin(t);
      const x = cxp + rx * c, y = cy + ry * sn;
      const side = Math.abs(c) < 0.1 ? 'c' : c < 0 ? 'l' : 'r';
      it.x = x; it.y = y; it.side = side; it.t = t;
      it.el.style.left = x + 'px'; it.el.style.top = y + 'px';
      it.el.dataset.side = side;
    });
  }

  // selected-mode annotations: inputs left, outputs right, each with a leader
  const annItems = [];
  function buildAnnotations(m) {
    annItems.forEach((a) => { a.el.remove(); a.line.remove(); });
    annItems.length = 0;
    inCol.innerHTML = ''; outCol.innerHTML = '';
    const make = (col, item, kind) => {
      const el = document.createElement('div');
      el.className = 'ann ann-' + kind + (item.kind === 'intervention' ? ' ann-int' : '');
      const eyebrow = kind === 'in' ? (item.kind === 'intervention' ? 'clamped' : 'observed') : (item.field || 'target');
      el.innerHTML = `<span class="ann-kind">${eyebrow}</span><span class="ann-label">${item.label}</span><span class="ann-id">${item.id}</span>`;
      col.appendChild(el);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('class', 'leader leader-' + kind);
      svg.appendChild(line);
      annItems.push({ el, line, anchor: item.anchor, kind });
    };
    m.inputs.forEach((x) => make(inCol, x, 'in'));
    m.outputs.forEach((x) => make(outCol, x, 'out'));
  }

  // ------------------------------------------------------------ state ----
  function setWeights(id, soft) {
    if (!id) { for (let i = 0; i < N; i++) wTgt[i] = rest[i]; return; }
    const m = byId[id];
    const lo = soft ? 0.22 : 0.1, mid = soft ? 0.45 : 0.4;
    for (let i = 0; i < N; i++) wTgt[i] = lo * (group[i] === 'meg' ? 0.6 : 1);
    m._inv.forEach((i) => { wTgt[i] = mid; });
    m._hot.forEach((i) => { wTgt[i] = 1; });
  }
  function preview(id) {
    state.hover = id;
    setWeights(id, true);
    ringItems.forEach((it) => it.el.classList.toggle('is-hover', it.m.id === id));
  }
  function faceAnchor(p) {
    // turn the orbit so the region this materialization names faces the camera
    const w = toWorld(p);
    const r = Math.hypot(w[0], w[2]);
    if (r < 18) return;
    let az = Math.atan2(w[0], w[2]);
    let d = az - orbit.tAz; d = Math.atan2(Math.sin(d), Math.cos(d));
    orbit.tAz += d;
    orbit.tEl = Math.max(0.05, Math.min(0.75, Math.atan2(w[1], r) * 0.6 + 0.2));
  }
  function select(id) {
    const m = byId[id];
    if (!m) return;
    state.selected = id;
    hero.classList.add('has-selection');
    hero.classList.remove('is-hovering');
    setWeights(id, false);
    ringItems.forEach((it) => it.el.classList.toggle('is-active', it.m.id === id));
    selTitle.textContent = m.id.replace(/_/g, ' ');
    selGroup.textContent = m.group_title;
    selDoc.textContent = m.doc;
    const dt = m.window_dt_s;
    const dtText = dt >= 1 ? `${dt} s` : dt >= 1e-3 ? `${(dt * 1e3).toPrecision(3).replace(/\.?0+$/, '')} ms` : dt >= 1e-6 ? `${(dt * 1e6).toPrecision(3).replace(/\.?0+$/, '')} µs` : `${(dt * 1e9).toPrecision(3).replace(/\.?0+$/, '')} ns`;
    const regions = m.regions.length ? m.regions.join(', ') : (m.systems.length ? m.systems.join(', ') : m.supports.join(', ') || 'whole substrate');
    selMeta.innerHTML = `<span><b>names</b> ${regions}</span><span><b>window</b> ${m.window_n} × ${dtText}</span><span><b>nodes lit</b> ${m._hot.length} of ${N}</span>`;
    buildAnnotations(m);
    faceAnchor(m.focus_anchor);
    orbit.tDist *= 470 / 430;
    closeBtn.focus({ preventScroll: true });
  }
  function deselect() {
    if (state.selected) orbit.tDist *= 430 / 470;
    state.selected = null;
    hero.classList.remove('has-selection');
    setWeights(null, false);
    ringItems.forEach((it) => it.el.classList.remove('is-active', 'is-hover'));
    annItems.forEach((a) => { a.el.remove(); a.line.remove(); });
    annItems.length = 0;
    inCol.innerHTML = ''; outCol.innerHTML = '';
  }
  closeBtn.addEventListener('click', deselect);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.selected) deselect(); });
  ring.addEventListener('mouseenter', () => hero.classList.add('is-hovering'));
  ring.addEventListener('mouseleave', () => hero.classList.remove('is-hovering'));

  // ----------------------------------------------------------- leaders ----
  function leaderPath(x0, y0, x1, y1, side) {
    // a hand-drawn feeling: one gentle curve, with an elbow so labels read cleanly
    const dx = x1 - x0;
    const k = side === 'c' ? 0 : Math.sign(dx) * Math.min(60, Math.abs(dx) * 0.35);
    return `M${x0.toFixed(1)},${y0.toFixed(1)} C${(x0 + k).toFixed(1)},${y0.toFixed(1)} ${(x1 - k * 0.6).toFixed(1)},${y1.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }
  function edgePoint(el, side) {
    const r = el.getBoundingClientRect(), h = hero.getBoundingClientRect();
    const y = r.top - h.top + r.height / 2;
    if (side === 'l') return [r.right - h.left + 4, y];
    if (side === 'r') return [r.left - h.left - 4, y];
    return [r.left - h.left + r.width / 2, r.top - h.top + (r.top - h.top < h.height / 2 ? r.height + 2 : -2)];
  }
  function drawLeaders() {
    const w = hero.clientWidth, h = hero.clientHeight;
    const compact = hero.classList.contains('compact');
    if (!state.selected) {
      ringItems.forEach((it) => {
        if (compact) { it.line.setAttribute('d', ''); return; }
        const [ax, ay, az] = it.m.whole ? silhouette(it.t) : project(it.m.focus_anchor);
        const [x0, y0] = edgePoint(it.el, it.side);
        it.line.setAttribute('d', leaderPath(x0, y0, ax, ay, it.side));
        it.line.style.opacity = az > 1 ? 0 : '';
      });
      annItems.forEach((a) => a.line.setAttribute('d', ''));
    } else {
      ringItems.forEach((it) => it.line.setAttribute('d', ''));
      annItems.forEach((a) => {
        if (compact) { a.line.setAttribute('d', ''); return; }
        const [ax, ay] = project(a.anchor);
        const [x0, y0] = edgePoint(a.el, a.kind === 'in' ? 'l' : 'r');
        a.line.setAttribute('d', leaderPath(x0, y0, Math.max(8, Math.min(w - 8, ax)), Math.max(8, Math.min(h - 8, ay)), a.kind === 'in' ? 'l' : 'r'));
      });
    }
  }

  // ------------------------------------------------------------ loop ----
  let last = performance.now();
  let visible = true;
  const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) requestAnimationFrame(frame); }, { threshold: 0.02 });
  io.observe(hero);

  function frame(now) {
    if (!visible) return;
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    orbit.idle += dt;
    if (!orbit.dragging) {
      orbit.tAz += orbit.vAz; orbit.tEl += orbit.vEl;
      orbit.vAz *= 0.92; orbit.vEl *= 0.92;
      orbit.tEl = Math.max(-1.3, Math.min(1.3, orbit.tEl));
      if (!reduceMotion && orbit.idle > 4 && !state.selected) orbit.tAz += dt * 0.06;   // ambient drift
    }
    orbit.az += (orbit.tAz - orbit.az) * 0.12;
    orbit.el += (orbit.tEl - orbit.el) * 0.12;
    orbit.dist += (orbit.tDist - orbit.dist) * 0.1;
    applyCamera();
    pivot.updateMatrixWorld();

    let moving = false;
    const k = reduceMotion ? 1 : 0.1;
    for (let i = 0; i < N; i++) {
      const d = wTgt[i] - wCur[i];
      if (Math.abs(d) > 0.002) { wCur[i] += d * k; moving = true; } else wCur[i] = wTgt[i];
    }
    if (moving || now < 3000) {
      pgeo.attributes.w.needsUpdate = true;
      for (let e = 0; e < E; e++) {
        const [a, b] = G.edges[e];
        const v = Math.min(wCur[a], wCur[b]);
        ew[2 * e] = v; ew[2 * e + 1] = v;
      }
      lgeo.attributes.w.needsUpdate = true;
    }
    renderer.render(scene, camera);
    drawLeaders();
    requestAnimationFrame(frame);
  }

  window.addEventListener('resize', resize);
  resize();
  orbit.dist = orbit.tDist;
  applyCamera();
  requestAnimationFrame(frame);
  hero.classList.add('ready');
  if (hint) setTimeout(() => hint.classList.add('fade'), 9000);

  // deep link: #m=eeg_forward
  const hash = new URLSearchParams(location.hash.slice(1)).get('m');
  if (hash && byId[hash]) select(hash);
})();
