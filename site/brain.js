/* ibm-1: the substrate as a figure.
   one data model (site/data/graph.js), any number of views: the interactive
   opener, and static snapshots rendered once into the article.  expects the
   three.js UMD global. */
window.IBMBrain = (function () {
  'use strict';
  const G = window.IBM_GRAPH;
  if (!G || !window.THREE) return null;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------- data ----
  // surface RAS (x right, y anterior, z superior) -> three (y up, viewer faces the subject)
  const N = G.nodes.length;
  const cx = G.center;
  const toWorld = (p) => [-(p[0] - cx[0]), p[2] - cx[2], p[1] - cx[1]];
  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  const GROUP_SIZE = { cortex: 2.4, thalamus: 3.0, basal_ganglia: 3.0, hippocampus: 3.0, amygdala: 3.0, brainstem: 3.4, cerebellum: 3.2, spinal: 3.4, retina: 2.4, cochlea: 2.4, eeg: 4.2, meg: 3.2 };
  const REST = { cortex: 0.74, thalamus: 0.8, basal_ganglia: 0.8, hippocampus: 0.8, amygdala: 0.8, brainstem: 0.8, cerebellum: 0.7, spinal: 0.55, retina: 0.5, cochlea: 0.5, eeg: 0.32, meg: 0.18 };
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), size = new Float32Array(N), rest = new Float32Array(N);
  const group = new Array(N), region = new Array(N), hemi = new Array(N);
  for (let i = 0; i < N; i++) {
    const n = G.nodes[i], w = toWorld(n.p), c = hex(n.c);
    pos.set(w, 3 * i); col.set(c, 3 * i);
    size[i] = GROUP_SIZE[n.g] || 2.6; rest[i] = REST[n.g] || 0.6;
    group[i] = n.g; region[i] = n.r; hemi[i] = n.h;
  }
  const E = G.edges.length;
  const epos = new Float32Array(E * 6), ecol = new Float32Array(E * 6);
  for (let e = 0; e < E; e++) {
    const [a, b] = G.edges[e];
    for (let k = 0; k < 3; k++) { epos[6 * e + k] = pos[3 * a + k]; epos[6 * e + 3 + k] = pos[3 * b + k]; ecol[6 * e + k] = col[3 * a + k]; ecol[6 * e + 3 + k] = col[3 * b + k]; }
  }
  const cortexIndex = new Uint16Array(G.cortex_faces.flat());
  // hulls: non-indexed triangles over node positions, remembering which node each vertex is
  const hullNode = [];
  G.hulls.forEach((h) => h.faces.forEach((f) => hullNode.push(f[0], f[1], f[2])));
  const HV = hullNode.length;
  const hpos = new Float32Array(HV * 3), hcol = new Float32Array(HV * 3);
  for (let v = 0; v < HV; v++) { const i = hullNode[v]; hpos.set(pos.subarray(3 * i, 3 * i + 3), 3 * v); hcol.set(col.subarray(3 * i, 3 * i + 3), 3 * v); }
  const spos = new Float32Array(G.scalp.xyz.length * 3);
  const scol = new Float32Array(G.scalp.xyz.length * 3);
  G.scalp.xyz.forEach((p, i) => {
    const w = toWorld(p); spos.set(w, 3 * i);
    const k = Math.max(0, Math.min(1, (w[1] + 110) / 70));         // fade toward the neck
    scol.set([0.56 * k, 0.59 * k, 0.68 * k], 3 * i);
  });
  // drop the neck: faces entirely below the chin are not part of the figure
  const sidx = new Uint16Array(G.scalp.faces.filter((f) => f.some((v) => spos[3 * v + 1] > -100)).flat());

  const MATS = G.materializations;
  const byId = Object.fromEntries(MATS.map((m) => [m.id, m]));
  const expand = (ranges) => { const out = []; for (const [a, b] of ranges) for (let i = a; i < b; i++) out.push(i); return out; };
  MATS.forEach((m) => {
    m._hot = expand(m.hot); m._inv = expand(m.involved);
    if (!m._hot.length) m._hot = m._inv;
    m.inputs.forEach((x) => { x._nodes = expand(x.nodes); });
    m.outputs.forEach((x) => { x._nodes = expand(x.nodes); });
  });
  const nodesWhere = (fn) => { const out = []; for (let i = 0; i < N; i++) if (fn(i)) out.push(i); return out; };
  const centroid = (idx) => { const c = [0, 0, 0]; idx.forEach((i) => { c[0] += G.nodes[i].p[0]; c[1] += G.nodes[i].p[1]; c[2] += G.nodes[i].p[2]; }); return c.map((v) => v / (idx.length || 1)); };

  // ----------------------------------------------------------- shaders ----
  const POINT_VS = `
    attribute float size; attribute float w; attribute vec3 color;
    varying vec3 vC; varying float vW; varying float vD; uniform float pr;
    void main(){
      vC = color; vW = w;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vD = clamp((-mv.z - 260.0) / 260.0, 0.0, 1.0);
      gl_PointSize = size * (0.7 + 1.1 * w) * (420.0 / -mv.z) * pr;
      gl_Position = projectionMatrix * mv;
    }`;
  const POINT_FS = `
    varying vec3 vC; varying float vW; varying float vD; uniform float light;
    void main(){
      vec2 q = gl_PointCoord - 0.5; float d = length(q);
      if (d > 0.5) discard;
      float w = clamp(vW, 0.0, 1.0);
      float a = smoothstep(0.5, 0.3, d);
      float lum = mix(0.28, 1.0, w);
      vec3 dark = mix(vC * lum, vec3(1.0), 0.14 * w * (1.0 - smoothstep(0.0, 0.25, d)));
      vec3 pale = mix(vC * mix(0.9, 0.72, w), vec3(0.55), 0.35 * (1.0 - w));
      vec3 c = mix(dark, pale, light);
      float alpha = a * mix(mix(0.10, 0.28, light), 1.0, w) * mix(1.0, mix(0.45, 0.8, w), vD);
      gl_FragColor = vec4(c, alpha);
    }`;
  const LINE_VS = `
    attribute float w; attribute vec3 color; varying vec3 vC; varying float vW; varying float vD;
    void main(){ vC = color; vW = clamp(w, 0.0, 1.0); vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vD = clamp((-mv.z - 260.0) / 260.0, 0.0, 1.0); gl_Position = projectionMatrix * mv; }`;
  const LINE_FS = `
    varying vec3 vC; varying float vW; varying float vD; uniform float light;
    void main(){ float alpha = mix(mix(0.03, 0.06, light), mix(0.42, 0.55, light), vW) * mix(1.0, 0.4, vD);
      vec3 c = mix(mix(vC, vec3(1.0), 0.15 * vW), vC * 0.6, light);
      gl_FragColor = vec4(c, alpha); }`;
  const SHEET_VS = `
    attribute float w; attribute vec3 color; varying vec3 vC; varying float vW; varying float vD;
    void main(){ vC = color; vW = clamp(w, 0.0, 1.0); vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vD = clamp((-mv.z - 260.0) / 260.0, 0.0, 1.0); gl_Position = projectionMatrix * mv; }`;
  const SHEET_FS = `
    varying vec3 vC; varying float vW; varying float vD; uniform float base; uniform float light;
    void main(){ float alpha = base * mix(0.25, 1.0, vW) * mix(1.0, 0.5, vD);
      vec3 c = mix(mix(vC, vec3(0.5, 0.55, 0.7), 0.25), mix(vC, vec3(0.35, 0.38, 0.45), 0.3), light);
      gl_FragColor = vec4(c, alpha); }`;

  // ------------------------------------------------------------ scene ----
  function createScene(renderer, opts) {
    const light = { value: opts && opts.light ? 1 : 0 };
    const scene = new THREE.Scene();
    const pivot = new THREE.Group();
    scene.add(pivot);
    const wCur = new Float32Array(N);
    const ew = new Float32Array(E * 2);
    const hw = new Float32Array(HV);

    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    pgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    pgeo.setAttribute('size', new THREE.BufferAttribute(size, 1));
    pgeo.setAttribute('w', new THREE.BufferAttribute(wCur, 1));
    const points = new THREE.Points(pgeo, new THREE.ShaderMaterial({ uniforms: { pr: { value: renderer.getPixelRatio() }, light }, vertexShader: POINT_VS, fragmentShader: POINT_FS, transparent: true, depthWrite: false }));
    points.renderOrder = 5;

    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute('position', new THREE.BufferAttribute(epos, 3));
    lgeo.setAttribute('color', new THREE.BufferAttribute(ecol, 3));
    lgeo.setAttribute('w', new THREE.BufferAttribute(ew, 1));
    const lines = new THREE.LineSegments(lgeo, new THREE.ShaderMaterial({ uniforms: { light }, vertexShader: LINE_VS, fragmentShader: LINE_FS, transparent: true, depthWrite: false }));
    lines.renderOrder = 4;

    // the cortical sheet: the same triangulation the edges came from, as a translucent surface
    const cgeo = new THREE.BufferGeometry();
    cgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    cgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    cgeo.setAttribute('w', new THREE.BufferAttribute(wCur, 1));
    cgeo.setIndex(new THREE.BufferAttribute(cortexIndex, 1));
    const sheet = new THREE.Mesh(cgeo, new THREE.ShaderMaterial({ uniforms: { base: { value: 0.13 }, light }, vertexShader: SHEET_VS, fragmentShader: SHEET_FS, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    sheet.renderOrder = 2;

    // subcortical structures as convex volumes
    const hgeo = new THREE.BufferGeometry();
    hgeo.setAttribute('position', new THREE.BufferAttribute(hpos, 3));
    hgeo.setAttribute('color', new THREE.BufferAttribute(hcol, 3));
    hgeo.setAttribute('w', new THREE.BufferAttribute(hw, 1));
    const hulls = new THREE.Mesh(hgeo, new THREE.ShaderMaterial({ uniforms: { base: { value: 0.2 }, light }, vertexShader: SHEET_VS, fragmentShader: SHEET_FS, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    hulls.renderOrder = 3;

    // the scalp: the subject's outer-skin BEM surface, as a ghost of the head
    const sgeo = new THREE.BufferGeometry();
    sgeo.setAttribute('position', new THREE.BufferAttribute(spos, 3));
    sgeo.setAttribute('color', new THREE.BufferAttribute(scol, 3));
    sgeo.setIndex(new THREE.BufferAttribute(sidx, 1));
    const scalp = new THREE.Mesh(sgeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.05, depthWrite: false, side: THREE.BackSide }));
    const scalpWire = new THREE.Mesh(sgeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.07, depthWrite: false, wireframe: true }));
    if (light.value) { scalp.material.color.setRGB(0.5, 0.5, 0.5); scalpWire.material.color.setRGB(0.45, 0.45, 0.5); scalp.material.opacity = 0.07; scalpWire.material.opacity = 0.16; }

    // pulses: a small pool of travelling points, driven by the view
    const PN = 480;
    const ppos = new Float32Array(PN * 3), pcol = new Float32Array(PN * 3), psize = new Float32Array(PN), pw = new Float32Array(PN);
    const pgeo2 = new THREE.BufferGeometry();
    pgeo2.setAttribute('position', new THREE.BufferAttribute(ppos, 3));
    pgeo2.setAttribute('color', new THREE.BufferAttribute(pcol, 3));
    pgeo2.setAttribute('size', new THREE.BufferAttribute(psize, 1));
    pgeo2.setAttribute('w', new THREE.BufferAttribute(pw, 1));
    const pulses = new THREE.Points(pgeo2, new THREE.ShaderMaterial({ uniforms: { pr: { value: renderer.getPixelRatio() }, light }, vertexShader: POINT_VS, fragmentShader: POINT_FS, transparent: true, depthWrite: false, depthTest: false }));
    pulses.renderOrder = 6; pulses.frustumCulled = false;
    scalp.renderOrder = 0; scalpWire.renderOrder = 1;

    pivot.add(scalp, scalpWire, sheet, hulls, lines, points, pulses);

    function syncDerived() {
      for (let e = 0; e < E; e++) { const [a, b] = G.edges[e]; const v = Math.min(wCur[a], wCur[b]); ew[2 * e] = v; ew[2 * e + 1] = v; }
      for (let v = 0; v < HV; v++) hw[v] = wCur[hullNode[v]];
      pgeo.attributes.w.needsUpdate = true; cgeo.attributes.w.needsUpdate = true;
      lgeo.attributes.w.needsUpdate = true; hgeo.attributes.w.needsUpdate = true;
    }
    function setScalp(v) { scalp.material.opacity = 0.05 * v; scalpWire.material.opacity = 0.07 * v; }
    function setPulses(fn) { fn(ppos, pcol, psize, pw, PN); pgeo2.attributes.position.needsUpdate = true; pgeo2.attributes.color.needsUpdate = true; pgeo2.attributes.size.needsUpdate = true; pgeo2.attributes.w.needsUpdate = true; }
    return { scene, pivot, wCur, syncDerived, setScalp, setPulses, PN };
  }

  // ------------------------------------------------- weights per mode ----
  function weightsFor(id, soft) {
    const w = new Float32Array(N);
    if (!id) { w.set(rest); return w; }
    const m = typeof id === 'string' ? byId[id] : id;
    const lo = soft ? 0.22 : 0.1, mid = soft ? 0.45 : 0.4;
    for (let i = 0; i < N; i++) w[i] = lo * (group[i] === 'meg' ? 0.6 : 1);
    m._inv.forEach((i) => { w[i] = mid; });
    m._hot.forEach((i) => { w[i] = 1; });
    return w;
  }

  // ------------------------------------------------------- transitions ----
  // a staged transition: everything settles toward its new level, then the
  // lit set blooms outward from the model's focus, each node with its own
  // delay and a small overshoot.  duration ~1.6 s, half that on the way back.
  function makeAnimator(wCur) {
    const from = new Float32Array(N), to = new Float32Array(N), t0 = new Float32Array(N), dur = new Float32Array(N);
    let active = false, endAt = 0;
    const easeOut = (u) => 1 - Math.pow(1 - u, 3);
    const easeBack = (u) => { const c = 1.7; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };
    function start(target, opts) {
      const now = performance.now();
      const origin = opts.origin || cx;
      const spread = opts.spread == null ? 900 : opts.spread;
      const base = opts.base == null ? 500 : opts.base;
      const instant = opts.instant || reduceMotion;
      let maxD = 1;
      const dist = new Float32Array(N);
      for (let i = 0; i < N; i++) { const p = G.nodes[i].p; dist[i] = Math.hypot(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]); if (dist[i] > maxD) maxD = dist[i]; }
      endAt = now;
      for (let i = 0; i < N; i++) {
        from[i] = wCur[i]; to[i] = target[i];
        const rising = target[i] > wCur[i] + 0.05;
        const delay = rising ? opts.baseDelay + spread * (opts.inward ? 1 - dist[i] / maxD : dist[i] / maxD) : 0;
        t0[i] = now + (instant ? 0 : delay);
        dur[i] = instant ? 1 : (rising ? 560 : base);
        if (t0[i] + dur[i] > endAt) endAt = t0[i] + dur[i];
      }
      active = true;
      if (instant) { wCur.set(target); active = false; }
    }
    function tick(now) {
      if (!active) return false;
      for (let i = 0; i < N; i++) {
        const u = Math.min(1, Math.max(0, (now - t0[i]) / dur[i]));
        const rising = to[i] > from[i] + 0.05;
        const k = rising && to[i] >= 0.99 ? easeBack(u) : easeOut(u);
        wCur[i] = from[i] + (to[i] - from[i]) * k;
      }
      if (now >= endAt) { wCur.set(to); active = false; }
      return true;
    }
    return { start, tick, get active() { return active; } };
  }

  // -------------------------------------------------- annotation engine ----
  // one label, one leader, and a tip that fits what it points at:
  //   point  - nothing or one thing:        a curved arrow
  //   fan    - a few things, or an array:   one trunk, then an arrowhead per element
  //   lasso  - a region or a cluster:       a loop drawn around it
  const SVG = 'http://www.w3.org/2000/svg';
  function tipKind(nodes) {
    if (!nodes || nodes.length <= 1) return 'point';
    const g = group[nodes[0]];
    if (g === 'eeg' || g === 'meg' || g === 'spinal' || g === 'retina' || g === 'cochlea') return 'fan';
    if (nodes.length <= 10) return 'fan';
    return 'lasso';
  }
  function curve(x0, y0, x1, y1, bulge, away) {
    // a single quadratic arc whose control point bows to one side of the chord
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L, ny = dx / L;
    let sgn = away ? Math.sign(nx * ((x0 + x1) / 2 - away[0]) + ny * ((y0 + y1) / 2 - away[1])) || 1 : 1;
    const mx = (x0 + x1) / 2 + nx * bulge * L * sgn, my = (y0 + y1) / 2 + ny * bulge * L * sgn;
    return `M${x0.toFixed(1)},${y0.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }
  function hull2d(pts) {
    if (pts.length < 3) return pts.slice();
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }
  function lassoPath(pts, pad) {
    const h = hull2d(pts);
    if (h.length < 3) return '';
    const c = h.reduce((a, q) => [a[0] + q[0] / h.length, a[1] + q[1] / h.length], [0, 0]);
    const ex = h.map((q) => { const d = Math.hypot(q[0] - c[0], q[1] - c[1]) || 1; return [q[0] + (q[0] - c[0]) / d * pad, q[1] + (q[1] - c[1]) / d * pad]; });
    // catmull-rom through the padded hull, closed
    const n = ex.length; let d = `M${ex[0][0].toFixed(1)},${ex[0][1].toFixed(1)}`;
    for (let i = 0; i < n; i++) {
      const p0 = ex[(i - 1 + n) % n], p1 = ex[i], p2 = ex[(i + 1) % n], p3 = ex[(i + 2) % n];
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6], c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d + 'Z';
  }
  // an annotation's svg: a group holding trunk, branches, loop
  function makeAnnotationSVG(svg, cls) {
    const g = document.createElementNS(SVG, 'g');
    g.setAttribute('class', 'ann-svg ' + cls);
    const trunk = document.createElementNS(SVG, 'path'); trunk.setAttribute('class', 'leader trunk');
    const branches = document.createElementNS(SVG, 'path'); branches.setAttribute('class', 'leader branches');
    const loop = document.createElementNS(SVG, 'path'); loop.setAttribute('class', 'leader loop');
    const dot = document.createElementNS(SVG, 'circle'); dot.setAttribute('class', 'leader-dot'); dot.setAttribute('r', '2.2');
    g.append(loop, branches, trunk, dot);
    svg.appendChild(g);
    return { g, trunk, branches, loop, dot, remove() { g.remove(); } };
  }
  // lay one annotation out.  ctx: { project(p)->[x,y,z], w, h, centre:[x,y] }
  function drawAnnotation(a, labelPt, side, ctx) {
    const nodes = a._nodes || [];
    const kind = a.tip || tipKind(nodes);
    const away = ctx.centre;
    let pts = nodes.map((i) => { const q = ctx.project(G.nodes[i].p); return [q[0], q[1], q[2], i]; }).filter((q) => q[2] < 1);
    const bulge = 0.18;
    a.svg.trunk.setAttribute('marker-end', kind === 'point' ? 'url(#arrow)' : '');
    a.svg.branches.setAttribute('d', ''); a.svg.loop.setAttribute('d', ''); a.svg.dot.setAttribute('r', '0');
    if (kind === 'point' || pts.length === 0) {
      const q = ctx.project(a.anchor);
      const tx = Math.max(8, Math.min(ctx.w - 8, q[0])), ty = Math.max(8, Math.min(ctx.h - 8, q[1]));
      a.svg.trunk.setAttribute('d', curve(labelPt[0], labelPt[1], tx, ty, bulge, away));
      a.svg.trunk.setAttribute('marker-end', 'url(#arrow)');
      return;
    }
    const c = pts.reduce((s, q) => [s[0] + q[0] / pts.length, s[1] + q[1] / pts.length], [0, 0]);
    if (kind === 'fan') {
      // an array: point at the elements facing the viewer, at most eighteen of them
      // an array: the elements on the label's side of it, at most fourteen
      let targets = pts;
      if (targets.length > 14) targets = pts.slice().sort((p, q) => Math.hypot(p[0] - labelPt[0], p[1] - labelPt[1]) - Math.hypot(q[0] - labelPt[0], q[1] - labelPt[1])).slice(0, 14);
      const tc = targets.reduce((s, q) => [s[0] + q[0] / targets.length, s[1] + q[1] / targets.length], [0, 0]);
      const r = Math.max(...targets.map((q) => Math.hypot(q[0] - tc[0], q[1] - tc[1])));
      const dx = labelPt[0] - tc[0], dy = labelPt[1] - tc[1], L = Math.hypot(dx, dy) || 1;
      const hub = [tc[0] + dx / L * (r * 0.6 + 30), tc[1] + dy / L * (r * 0.6 + 30)];
      a.svg.trunk.setAttribute('d', curve(labelPt[0], labelPt[1], hub[0], hub[1], bulge, away));
      a.svg.branches.setAttribute('d', targets.map((q) => curve(hub[0], hub[1], q[0], q[1], 0.12, away)).join(' '));
      a.svg.branches.setAttribute('marker-end', 'url(#arrow-small)');
      a.svg.dot.setAttribute('r', '2.2'); a.svg.dot.setAttribute('cx', hub[0].toFixed(1)); a.svg.dot.setAttribute('cy', hub[1].toFixed(1));
      return;
    }
    // lasso: a loop around the cluster, the trunk arriving at its nearest edge
    const loop = lassoPath(pts.map((q) => [q[0], q[1]]), 9);
    a.svg.loop.setAttribute('d', loop);
    const h = hull2d(pts.map((q) => [q[0], q[1]]));
    let best = h[0], bd = Infinity;
    h.forEach((q) => { const d = Math.hypot(q[0] - labelPt[0], q[1] - labelPt[1]); if (d < bd) { bd = d; best = q; } });
    const dx = best[0] - c[0], dy = best[1] - c[1], L = Math.hypot(dx, dy) || 1;
    const end = [best[0] + dx / L * 11, best[1] + dy / L * 11];
    a.svg.trunk.setAttribute('d', curve(labelPt[0], labelPt[1], end[0], end[1], bulge, away));
    a.svg.dot.setAttribute('r', '2.4'); a.svg.dot.setAttribute('cx', end[0].toFixed(1)); a.svg.dot.setAttribute('cy', end[1].toFixed(1));
  }
  function ensureMarkers(svg) {
    if (svg.querySelector('defs')) return;
    svg.insertAdjacentHTML('afterbegin', `<defs>
      <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L7,4 L0,7.2" fill="none" stroke="currentColor" stroke-width="1.1"/></marker>
      <marker id="arrow-small" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0.8 L7,4 L0,7.2" fill="none" stroke="currentColor" stroke-width="1.3"/></marker>
    </defs>`);
  }

  // ------------------------------------------------------------- views ----
  function cameraFor(aspect) { const c = new THREE.PerspectiveCamera(32, aspect, 10, 3000); return c; }
  function placeCamera(camera, az, el, dist) {
    camera.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az));
    camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
  }
  function projector(camera, w, h) {
    const v = new THREE.Vector3();
    return (p) => { const q = toWorld(p); v.set(q[0], q[1], q[2]).project(camera); return [(v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h, v.z]; };
  }
  function focusAzimuth(p) {
    const w = toWorld(p); const r = Math.hypot(w[0], w[2]);
    return r < 18 ? null : { az: Math.atan2(w[0], w[2]), el: Math.max(0.05, Math.min(0.75, Math.atan2(w[1], r) * 0.6 + 0.2)) };
  }

  // --- pulses ---------------------------------------------------------------
  // a pulse is a short trail moving along one arc from an input node, over
  // the model's focus, to an output node.  ~3 s each, spawned continuously
  // while a materialization is selected, cyan on the way in, amber on the way out.
  function makePulses(S) {
    const IN = [0.39, 0.83, 0.9], OUT = [0.94, 0.7, 0.29];
    const TRAIL = 8, MAXP = Math.floor(S.PN / TRAIL);
    const live = [];
    let model = null, spawnAt = 0;
    const world = (p) => toWorld(p);
    const arc = (a, b) => {
      // an arc over the surface: the midpoint pushed away from the brain's centre
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      const L = Math.hypot(m[0], m[1], m[2]) || 1;
      const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      const k = (Math.max(95, L) + d * 0.25) / L;
      return [m[0] * k, m[1] * k, m[2] * k];
    };
    const bez = (a, c, b, t) => { const u = 1 - t; return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1], u * u * a[2] + 2 * u * t * c[2] + t * t * b[2]]; };
    const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
    function set(m) { model = m; live.length = 0; }
    function spawn(now) {
      const ins = model.inputs.filter((x) => x._nodes.length), outs = model.outputs.filter((x) => x._nodes.length);
      const src = ins.length ? world(G.nodes[pick(pick(ins)._nodes)].p) : world(pick(model.inputs).anchor);
      const dst = outs.length ? world(G.nodes[pick(pick(outs)._nodes)].p) : world(model.focus_anchor);
      const mid = world(G.nodes[pick(model._hot)].p);
      live.push({ t0: now, dur: 2600 + Math.random() * 1200, a: src, c1: arc(src, mid), m: mid, c2: arc(mid, dst), b: dst });
    }
    function tick(now, S_) {
      if (model && now > spawnAt && live.length < MAXP) { spawn(now); spawnAt = now + 140 + Math.random() * 120; }
      for (let i = live.length - 1; i >= 0; i--) if (now - live[i].t0 > live[i].dur) live.splice(i, 1);
      S_.setPulses((pos, col, size, w, PN) => {
        w.fill(0);
        live.forEach((p, k) => {
          const u = (now - p.t0) / p.dur;
          for (let j = 0; j < TRAIL; j++) {
            const uu = u - j * 0.012;
            const idx = k * TRAIL + j;
            if (uu < 0 || uu > 1) { w[idx] = 0; continue; }
            const q = uu < 0.5 ? bez(p.a, p.c1, p.m, uu * 2) : bez(p.m, p.c2, p.b, (uu - 0.5) * 2);
            pos[3 * idx] = q[0]; pos[3 * idx + 1] = q[1]; pos[3 * idx + 2] = q[2];
            const mixc = Math.min(1, Math.max(0, (uu - 0.35) / 0.3));
            col[3 * idx] = IN[0] + (OUT[0] - IN[0]) * mixc; col[3 * idx + 1] = IN[1] + (OUT[1] - IN[1]) * mixc; col[3 * idx + 2] = IN[2] + (OUT[2] - IN[2]) * mixc;
            const fade = Math.min(1, uu * 8) * Math.min(1, (1 - uu) * 8);
            size[idx] = (j === 0 ? 5.2 : 3.6 - j * 0.35);
            w[idx] = (j === 0 ? 1 : 0.7 - j * 0.08) * fade;
          }
        });
      });
    }
    return { set, tick, get active() { return !!model; } };
  }

  // --- the interactive opener -------------------------------------------
  function createHero(els) {
    const { hero, canvas, svg, ring, selTitle, selGroup, selDoc, selMeta, inCol, outCol, closeBtn, hint } = els;
    ensureMarkers(svg);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const S = createScene(renderer);
    const camera = cameraFor(1);
    const anim = makeAnimator(S.wCur);
    const pulses = makePulses(S);
    S.wCur.set(rest); S.syncDerived();
    const state = { selected: null, hover: null };
    const orbit = { az: -0.65, el: 0.32, dist: 430, tAz: -0.65, tEl: 0.32, tDist: 430, vAz: 0, vEl: 0, dragging: false, lastX: 0, lastY: 0, idle: 0, glide: 0.12 };
    let W = 1, H = 1, project = () => [0, 0, 0];

    function resize() {
      W = hero.clientWidth; H = hero.clientHeight;
      renderer.setSize(W, H, false);
      camera.aspect = W / H; camera.updateProjectionMatrix();
      const fit = Math.max(1, 1.15 / Math.min(1, W / H));
      orbit.tDist = (state.selected ? 470 : 430) * fit * (W < 720 ? 1.15 : 1);
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      project = projector(camera, W, H);
      layoutRing();
    }
    // drag to orbit
    const onDown = (x, y) => { orbit.dragging = true; orbit.lastX = x; orbit.lastY = y; orbit.vAz = 0; orbit.vEl = 0; hero.classList.add('dragging'); };
    const onMove = (x, y) => { if (!orbit.dragging) return; const dx = x - orbit.lastX, dy = y - orbit.lastY; orbit.lastX = x; orbit.lastY = y; orbit.tAz -= dx * 0.006; orbit.tEl = Math.max(-1.3, Math.min(1.3, orbit.tEl + dy * 0.006)); orbit.vAz = -dx * 0.006; orbit.vEl = dy * 0.006; orbit.idle = 0; };
    const onUp = () => { orbit.dragging = false; hero.classList.remove('dragging'); };
    canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); onDown(e.clientX, e.clientY); });
    canvas.addEventListener('pointermove', (e) => onMove(e.clientX, e.clientY));
    canvas.addEventListener('pointerup', onUp); canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('wheel', (e) => { if (!e.ctrlKey) return; e.preventDefault(); orbit.tDist = Math.max(220, Math.min(900, orbit.tDist * (1 + e.deltaY * 0.0015))); }, { passive: false });

    // the ring
    const GROUP_ORDER = ['electrophysiology', 'decoding', 'stimulation', 'state', 'slow', 'hemodynamic', 'surrogate'];
    const ordered = GROUP_ORDER.flatMap((g) => MATS.filter((m) => m.group === g)).concat(MATS.filter((m) => !GROUP_ORDER.includes(m.group)));
    const ringItems = [];
    ordered.forEach((m, i) => {
      const el = document.createElement('button');
      el.type = 'button'; el.className = 'ring-item'; el.dataset.id = m.id;
      const first = i === 0 || ordered[i - 1].group !== m.group;
      el.innerHTML = (first ? `<span class="ring-group">${m.group_title}</span>` : '') + `<span class="ring-label">${m.id}</span>`;
      el.addEventListener('click', () => select(m.id));
      el.addEventListener('mouseenter', () => { if (!state.selected) preview(m.id); });
      el.addEventListener('mouseleave', () => { if (!state.selected) preview(null); });
      el.addEventListener('focus', () => { if (!state.selected) preview(m.id); });
      el.addEventListener('blur', () => { if (!state.selected) preview(null); });
      ring.appendChild(el);
      const line = document.createElementNS(SVG, 'path'); line.setAttribute('class', 'leader ring-leader'); svg.appendChild(line);
      ringItems.push({ m, el, line, x: 0, y: 0, side: 'l', t: 0 });
    });
    const hoverAnn = { svg: makeAnnotationSVG(svg, 'hover-ann'), anchor: cx, _nodes: [] };
    function layoutRing() {
      const compact = W < 760;
      hero.classList.toggle('compact', compact);
      if (compact) return;
      const n = ringItems.length, rx = Math.min(W * 0.5 - 200, H * 0.95), ry = H * 0.5 - 70, cy = H * 0.5, cxp = W * 0.5;
      const K = 2.4, S_ = 720, cum = [0];
      for (let k = 1; k <= S_; k++) { const a0 = (k - 1) / S_ * Math.PI * 2, a1 = k / S_ * Math.PI * 2; cum.push(cum[k - 1] + Math.hypot(rx * (Math.cos(a1) - Math.cos(a0)) * K, ry * (Math.sin(a1) - Math.sin(a0)))); }
      const total = cum[S_];
      ringItems.forEach((it, i) => {
        const target = (i / n) * total; let k = 0; while (k < S_ && cum[k + 1] < target) k++;
        const t = -Math.PI / 2 + (k / S_) * Math.PI * 2, c = Math.cos(t);
        it.x = cxp + rx * c; it.y = cy + ry * Math.sin(t); it.t = t;
        it.side = Math.abs(c) < 0.1 ? 'c' : c < 0 ? 'l' : 'r';
        it.el.style.left = it.x + 'px'; it.el.style.top = it.y + 'px'; it.el.dataset.side = it.side;
      });
    }
    function edgePoint(el, side) {
      const r = el.getBoundingClientRect(), h = hero.getBoundingClientRect(), y = r.top - h.top + r.height / 2;
      if (side === 'l') return [r.right - h.left + 4, y];
      if (side === 'r') return [r.left - h.left - 4, y];
      return [r.left - h.left + r.width / 2, r.top - h.top + (r.top - h.top < h.height / 2 ? r.height + 2 : -2)];
    }
    // where the brain's outline is along a ray from its centre: the farthest
    // projected tissue node within a narrow band of the ray, so the arrow lands
    // on the silhouette as it currently stands, not on a bounding ellipse.
    const TISSUE = nodesWhere((i) => group[i] !== 'eeg' && group[i] !== 'meg' && group[i] !== 'retina' && group[i] !== 'cochlea' && group[i] !== 'spinal');
    const px_ = new Float32Array(TISSUE.length), py_ = new Float32Array(TISSUE.length);
    function projectTissue() { for (let k = 0; k < TISSUE.length; k++) { const q = project(G.nodes[TISSUE[k]].p); px_[k] = q[0]; py_[k] = q[1]; } }
    function reach(ox, oy, dx, dy) {
      const band = 11; let best = 0, far = 0;
      for (let k = 0; k < TISSUE.length; k++) {
        const ax = px_[k] - ox, ay = py_[k] - oy, along = ax * dx + ay * dy;
        if (along > far) far = along;
        if (along <= best) continue;
        if (Math.abs(ax * dy - ay * dx) > band) continue;
        best = along;
      }
      return (best || far) + 4;
    }

    // selected mode annotations
    const annItems = [];
    function buildAnnotations(m) {
      annItems.forEach((a) => { a.el.remove(); a.svg.remove(); }); annItems.length = 0;
      inCol.innerHTML = ''; outCol.innerHTML = '';
      const make = (colEl, item, kind) => {
        const el = document.createElement('div');
        el.className = 'ann ann-' + kind + (item.kind === 'intervention' ? ' ann-int' : '');
        const eyebrow = kind === 'in' ? (item.kind === 'intervention' ? 'clamped' : 'observed') : (item.field || 'target');
        el.innerHTML = `<span class="ann-kind">${eyebrow}</span><span class="ann-label">${item.label}</span><span class="ann-id">${item.id}</span>`;
        colEl.appendChild(el);
        annItems.push({ el, svg: makeAnnotationSVG(svg, 'ann-' + kind), anchor: item.anchor, _nodes: item._nodes, kind, side: kind === 'in' ? 'l' : 'r' });
      };
      m.inputs.forEach((x) => make(inCol, x, 'in'));
      m.outputs.forEach((x) => make(outCol, x, 'out'));
    }
    function preview(id) {
      state.hover = id;
      anim.start(weightsFor(id, true), { origin: id ? byId[id].focus_anchor : cx, base: 300, spread: 300, baseDelay: 0 });
      ringItems.forEach((it) => it.el.classList.toggle('is-hover', it.m.id === id));
      hoverAnn._nodes = id ? byId[id]._hot : [];
      hoverAnn.anchor = id ? byId[id].focus_anchor : cx;
      hoverAnn.item = id ? ringItems.find((it) => it.m.id === id) : null;
    }
    function select(id) {
      const m = byId[id]; if (!m) return;
      state.selected = id; state.hover = null;
      hero.classList.add('has-selection'); hero.classList.remove('is-hovering');
      hoverAnn._nodes = []; hoverAnn.item = null;
      anim.start(weightsFor(id, false), { origin: m.focus_anchor, base: 500, spread: 900, baseDelay: 350 });
      ringItems.forEach((it) => it.el.classList.toggle('is-active', it.m.id === id));
      selTitle.textContent = m.id.replace(/_/g, ' ');
      selGroup.textContent = m.group_title;
      selDoc.textContent = m.doc;
      const dt = m.window_dt_s;
      const dtText = dt >= 1 ? `${dt} s` : dt >= 1e-3 ? `${+(dt * 1e3).toPrecision(3)} ms` : dt >= 1e-6 ? `${+(dt * 1e6).toPrecision(3)} µs` : `${+(dt * 1e9).toPrecision(3)} ns`;
      const regions = m.regions.length ? m.regions.join(', ') : (m.systems.length ? m.systems.join(', ') : m.supports.join(', ') || 'whole substrate');
      selMeta.innerHTML = `<span><b>names</b> ${regions}</span><span><b>window</b> ${m.window_n} × ${dtText}</span><span><b>nodes lit</b> ${m._hot.length} of ${N}</span>`;
      buildAnnotations(m);
      setTimeout(() => { if (state.selected === id) pulses.set(m); }, reduceMotion ? 0 : 900);
      const f = focusAzimuth(m.focus_anchor);
      if (f) { let d = f.az - orbit.tAz; d = Math.atan2(Math.sin(d), Math.cos(d)); orbit.tAz += d; orbit.tEl = f.el; }
      orbit.tDist *= 470 / 430; orbit.glide = 0.045;
      closeBtn.focus({ preventScroll: true });
    }
    function deselect() {
      if (!state.selected) return;
      const m = byId[state.selected];
      state.selected = null;
      pulses.set(null);
      hero.classList.remove('has-selection');
      anim.start(weightsFor(null), { origin: m.focus_anchor, base: 450, spread: 650, baseDelay: 120, inward: false });
      ringItems.forEach((it) => it.el.classList.remove('is-active', 'is-hover'));
      annItems.forEach((a) => { a.el.remove(); a.svg.remove(); }); annItems.length = 0;
      inCol.innerHTML = ''; outCol.innerHTML = '';
      orbit.tDist *= 430 / 470; orbit.glide = 0.06;
    }
    closeBtn.addEventListener('click', deselect);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') deselect(); });
    ring.addEventListener('mouseenter', () => hero.classList.add('is-hovering'));
    ring.addEventListener('mouseleave', () => hero.classList.remove('is-hovering'));

    function drawLeaders() {
      const compact = hero.classList.contains('compact');
      const ctx = { project, w: W, h: H, centre: [W / 2, H / 2] };
      if (!state.selected) {
        if (!compact) projectTissue();
        const [ox, oy] = project(cx);
        ringItems.forEach((it) => {
          if (compact) { it.line.setAttribute('d', ''); return; }
          const hovered = state.hover === it.m.id;
          if (hovered) { it.line.setAttribute('d', ''); return; }
          const [x0, y0] = edgePoint(it.el, it.side);
          const L = Math.hypot(x0 - ox, y0 - oy) || 1, dx = (x0 - ox) / L, dy = (y0 - oy) / L;
          const r = reach(ox, oy, dx, dy);
          it.line.setAttribute('d', curve(x0, y0, ox + dx * r, oy + dy * r, 0.12, [W / 2, H / 2]));
          it.line.style.opacity = '';
        });
        if (hoverAnn.item && !compact) {
          const [x0, y0] = edgePoint(hoverAnn.item.el, hoverAnn.item.side);
          hoverAnn.svg.g.style.display = '';
          drawAnnotation(hoverAnn, [x0, y0], hoverAnn.item.side, ctx);
        } else hoverAnn.svg.g.style.display = 'none';
        annItems.forEach((a) => a.svg.g.style.display = 'none');
      } else {
        ringItems.forEach((it) => it.line.setAttribute('d', ''));
        hoverAnn.svg.g.style.display = 'none';
        annItems.forEach((a) => {
          if (compact) { a.svg.g.style.display = 'none'; return; }
          a.svg.g.style.display = '';
          drawAnnotation(a, edgePoint(a.el, a.side), a.side, ctx);
        });
      }
    }

    let last = performance.now(), visible = true;
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) requestAnimationFrame(frame); }, { threshold: 0.02 }).observe(hero);
    function frame(now) {
      if (!visible) return;
      if (W < 2 || H < 2) { requestAnimationFrame(frame); return; }
      const dt = Math.min(0.05, (now - last) / 1000); last = now; orbit.idle += dt;
      if (!orbit.dragging) {
        orbit.tAz += orbit.vAz; orbit.tEl += orbit.vEl; orbit.vAz *= 0.92; orbit.vEl *= 0.92;
        orbit.tEl = Math.max(-1.3, Math.min(1.3, orbit.tEl));
        if (!reduceMotion && orbit.idle > 15 && !state.selected) orbit.tAz += dt * 0.06;
      }
      orbit.glide += (0.12 - orbit.glide) * 0.02;
      const g = orbit.dragging ? 0.2 : orbit.glide;
      orbit.az += (orbit.tAz - orbit.az) * g; orbit.el += (orbit.tEl - orbit.el) * g; orbit.dist += (orbit.tDist - orbit.dist) * g * 0.8;
      placeCamera(camera, orbit.az, orbit.el, orbit.dist);
      window.IBM_ZOOM = 430 / orbit.dist;
      S.pivot.updateMatrixWorld();
      if (anim.tick(now)) S.syncDerived();
      if (pulses.active || now < 100) pulses.tick(now, S);
      renderer.render(S.scene, camera);
      drawLeaders();
      requestAnimationFrame(frame);
    }
    window.addEventListener('resize', resize);
    resize(); orbit.dist = orbit.tDist; placeCamera(camera, orbit.az, orbit.el, orbit.dist);
    requestAnimationFrame(frame);
    hero.classList.add('ready');
    if (hint) setTimeout(() => hint.classList.add('fade'), 9000);
    const hash = new URLSearchParams(location.hash.slice(1)).get('m');
    if (hash && byId[hash]) select(hash);
    return { select, deselect };
  }

  // --- live figures for the article --------------------------------------
  // a figure element carries data-snap; its spec says what to light, where to
  // look from, and what to annotate.  every figure is a view: one shared
  // offscreen renderer draws whichever are on screen, a drag orbits them, and
  // after fifteen idle seconds they turn on their own.
  const IDLE_MS = 15000;
  let snapRenderer = null, snapScenes = {};
  const views = [];
  function isLight() {
    const t = document.documentElement.dataset.theme;
    if (t === 'dark') return false; if (t === 'light') return true;
    return !window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function sceneFor(light) {
    if (!snapRenderer) {
      snapRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      snapRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      snapRenderer.setClearColor(0x000000, 0);
    }
    const key = light ? 'light' : 'dark';
    if (!snapScenes[key]) snapScenes[key] = createScene(snapRenderer, { light });
    return snapScenes[key];
  }
  function layoutView(v) {
    const { fig, spec } = v;
    const W = fig.clientWidth, H = Math.round(W * (spec.aspect || 0.66));
    if (!W) return false;
    v.W = W; v.H = H;
    fig.style.setProperty('--snap-h', H + 'px');
    const pr = snapRenderer.getPixelRatio();
    v.out.width = W * pr; v.out.height = H * pr; v.out.style.width = W + 'px'; v.out.style.height = H + 'px';
    v.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    v.camera = cameraFor(W / H);
    v.orbit.dist = (spec.dist || 430) * Math.max(1, 1.15 / Math.min(1, W / H));
    // labels: placed once per layout from the spec's own view, leaders follow the orbit
    v.svg.querySelectorAll('g').forEach((g) => g.remove()); v.labels.innerHTML = '';
    v.items = spec.annotations ? spec.annotations() : [];
    placeCamera(v.camera, spec.az == null ? -0.65 : spec.az, spec.el == null ? 0.3 : spec.el, v.orbit.dist);
    const project = projector(v.camera, W, H);
    const left = [], right = [];
    v.items.forEach((a) => { const q = project(a.anchor); a._px = q; (a.side ? a.side === 'l' : q[0] < W / 2) ? left.push(a) : right.push(a); });
    const place = (list, side) => {
      list.sort((p, q) => p._px[1] - q._px[1]);
      const gap = H / (list.length + 1);
      list.forEach((a, k) => {
        const el = document.createElement('div');
        el.className = 'ann ann-snap ann-' + (a.cls || 'note'); el.dataset.side = side;
        el.innerHTML = (a.kind_label ? `<span class="ann-kind">${a.kind_label}</span>` : '') + `<span class="ann-label">${a.label}</span>` + (a.sub ? `<span class="ann-id">${a.sub}</span>` : '');
        el.style.top = Math.round(gap * (k + 1)) + 'px';
        v.labels.appendChild(el);
        a.svg = makeAnnotationSVG(v.svg, 'ann-' + (a.cls || 'note'));
        a.el = el; a._side = side;
      });
    };
    place(left, 'l'); place(right, 'r');
    return true;
  }
  function renderView(v) {
    const light = isLight();
    const S = sceneFor(light);
    v.fig.dataset.theme = light ? 'light' : 'dark';
    if (!v.weights) v.weights = v.spec.weights ? v.spec.weights() : weightsFor(v.spec.model || null, false);
    snapRenderer.setSize(v.W, v.H, false);
    S.wCur.set(v.weights); S.syncDerived();
    S.setScalp(v.spec.scalp == null ? 1 : v.spec.scalp);
    placeCamera(v.camera, v.orbit.az, v.orbit.el, v.orbit.dist);
    S.pivot.updateMatrixWorld();
    snapRenderer.render(S.scene, v.camera);
    const g2 = v.out.getContext('2d');
    g2.clearRect(0, 0, v.out.width, v.out.height);
    g2.drawImage(snapRenderer.domElement, 0, 0, v.out.width, v.out.height);
    const project = projector(v.camera, v.W, v.H);
    const ctx = { project, w: v.W, h: v.H, centre: [v.W / 2, v.H / 2] };
    const f = v.fig.getBoundingClientRect();
    v.items.forEach((a) => {
      const r = a.el.getBoundingClientRect();
      const pt = a._side === 'l' ? [r.right - f.left + 4, r.top - f.top + r.height / 2] : [r.left - f.left - 4, r.top - f.top + r.height / 2];
      drawAnnotation(a, pt, a._side, ctx);
    });
    v.fig.classList.add('rendered');
  }
  function mountView(fig, spec) {
    sceneFor(isLight());
    const v = {
      fig, spec, out: fig.querySelector('canvas'), svg: fig.querySelector('svg'), labels: fig.querySelector('.snap-labels'),
      orbit: { az: spec.az == null ? -0.65 : spec.az, el: spec.el == null ? 0.3 : spec.el, dist: 430, tAz: 0, tEl: 0, vAz: 0, vEl: 0, dragging: false, lastX: 0, lastY: 0 },
      items: [], weights: null, visible: false, dirty: true, lastTouch: performance.now(), camera: null, W: 0, H: 0,
    };
    v.orbit.tAz = v.orbit.az; v.orbit.tEl = v.orbit.el;
    ensureMarkers(v.svg);
    const c = v.out;
    c.addEventListener('pointerdown', (e) => { c.setPointerCapture(e.pointerId); v.orbit.dragging = true; v.orbit.lastX = e.clientX; v.orbit.lastY = e.clientY; v.orbit.vAz = 0; v.orbit.vEl = 0; v.lastTouch = performance.now(); fig.classList.add('dragging'); });
    c.addEventListener('pointermove', (e) => {
      if (!v.orbit.dragging) return;
      const dx = e.clientX - v.orbit.lastX, dy = e.clientY - v.orbit.lastY;
      v.orbit.lastX = e.clientX; v.orbit.lastY = e.clientY;
      v.orbit.tAz -= dx * 0.008; v.orbit.tEl = Math.max(-1.3, Math.min(1.3, v.orbit.tEl + dy * 0.008));
      v.orbit.vAz = -dx * 0.008; v.orbit.vEl = dy * 0.008; v.lastTouch = performance.now(); v.dirty = true;
    });
    const up = () => { v.orbit.dragging = false; v.lastTouch = performance.now(); fig.classList.remove('dragging'); };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
    views.push(v);
    return v;
  }
  function viewsLoop(now) {
    let any = false;
    for (const v of views) {
      if (!v.visible) continue;
      if (!v.camera && !layoutView(v)) continue;
      const o = v.orbit;
      let moving = v.dirty;
      if (!o.dragging) {
        if (Math.abs(o.vAz) > 1e-4 || Math.abs(o.vEl) > 1e-4) { o.tAz += o.vAz; o.tEl += o.vEl; o.vAz *= 0.9; o.vEl *= 0.9; o.tEl = Math.max(-1.3, Math.min(1.3, o.tEl)); moving = true; }
        if (!reduceMotion && now - v.lastTouch > IDLE_MS) { o.tAz += 0.0012; moving = true; }
      } else moving = true;
      const dAz = o.tAz - o.az, dEl = o.tEl - o.el;
      if (Math.abs(dAz) > 1e-4 || Math.abs(dEl) > 1e-4) { o.az += dAz * 0.18; o.el += dEl * 0.18; moving = true; }
      if (moving) { renderView(v); v.dirty = false; any = true; }
    }
    requestAnimationFrame(viewsLoop);
  }
  function mountSnapshots(specs) {
    const figs = Array.from(document.querySelectorAll('[data-snap]'));
    figs.forEach((fig) => { const spec = specs[fig.dataset.snap]; if (spec) mountView(fig, spec); });
    const eager = /[?&]eager/.test(location.search);
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => { const v = views.find((x) => x.fig === en.target); if (v) { v.visible = en.isIntersecting; if (v.visible) v.dirty = true; } });
    }, { rootMargin: '200px 0px' });
    views.forEach((v) => { io.observe(v.fig); if (eager) { v.visible = true; } });
    const redo = () => { views.forEach((v) => { v.camera = null; v.dirty = true; }); };
    const rethemed = () => { views.forEach((v) => { v.dirty = true; }); };
    let t = null;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(redo, 200); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rethemed);
    new MutationObserver(rethemed).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    requestAnimationFrame(viewsLoop);
  }

  // annotation helpers for specs
  const A = {
    region: (label, r, h, extra) => Object.assign({ label, _nodes: nodesWhere((i) => region[i] === r && (!h || hemi[i] === h)), anchor: null }, extra || {}),
    group: (label, g, extra) => Object.assign({ label, _nodes: nodesWhere((i) => group[i] === g), anchor: null }, extra || {}),
    model_io: (id) => { const m = byId[id]; return m.inputs.map((x) => ({ label: x.label, sub: x.id, kind_label: x.kind === 'intervention' ? 'clamped' : 'observed', cls: 'in', side: 'l', _nodes: x._nodes, anchor: x.anchor })).concat(m.outputs.map((x) => ({ label: x.label, sub: x.id, kind_label: x.field, cls: 'out', side: 'r', _nodes: x._nodes, anchor: x.anchor }))); },
  };
  const finish = (items) => items.map((a) => { if (!a.anchor) a.anchor = centroid(a._nodes); return a; });

  return { G, byId, MATS, createHero, mountSnapshots, weightsFor, A, finish, nodesWhere, group, region, hemi, rest, N };
})();
