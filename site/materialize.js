/* the materialization diagram: ONE scene, four brains.

   the implicit model sits at the centre in full atlas colour and carries every variable.
   three materialized models ring it, each drained to grey except the sites its own request
   reaches, with its declared inputs in the observed ramp and its outputs in the target one.
   an arrow runs from the centre to each: the models are traced OUT of the substrate, and
   the picture should say that rather than leave it to a caption.

   built as one scene rather than four canvases because four separate renders cannot share
   a camera -- the ring only reads as a ring if all four brains are in the same space and
   turn together. */
(function () {
  const host = document.getElementById("mz");
  const G = window.IBM_GRAPH;
  if (!host || !G || !window.THREE) return;
  const T = window.THREE;

  const MODELS = [
    { id: "eeg_to_image", name: "IBM-1-EEG-to-Image" },
    { id: "speech_envelope", name: "IBM-1-Speech-Envelope" },
    { id: "tms_response", name: "IBM-1-TMS-Response" },
  ];
  const MATS = Object.fromEntries((G.materializations || []).map((m) => [m.id, m]));
  const N = G.nodes.length;
  const CORTEX = G.cortex_range ? G.cortex_range[1] : N;

  const css = getComputedStyle(document.documentElement);
  const IN = new T.Color(css.getPropertyValue("--trace-in").trim() || "#f5424d");
  const OUT = new T.Color(css.getPropertyValue("--trace-out").trim() || "#a159fa");

  /* positions, centred and scaled so a brain is about one unit tall */
  const ctr = new T.Vector3().fromArray(G.center || [0, 0, 0]);
  const pos = new Float32Array(N * 3);
  let span = 0;
  for (let i = 0; i < N; i++) {
    const p = G.nodes[i].p;
    pos[i * 3] = p[0] - ctr.x; pos[i * 3 + 1] = p[1] - ctr.y; pos[i * 3 + 2] = p[2] - ctr.z;
    span = Math.max(span, Math.abs(pos[i * 3]), Math.abs(pos[i * 3 + 1]), Math.abs(pos[i * 3 + 2]));
  }
  const S = 1 / span;
  for (let i = 0; i < pos.length; i++) pos[i] *= S;

  const inRanges = (ranges, i) => {
    if (!ranges) return false;
    for (const r of ranges) if (i >= r[0] && i <= r[1]) return true;
    return false;
  };

  /* one brain: a point cloud, plus the cortex surface as a faint shell */
  function makeBrain(colorOf, opacity) {
    const g = new T.Group();
    const col = new Float32Array(N * 3);
    const c = new T.Color();
    for (let i = 0; i < N; i++) {
      colorOf(i, c);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(pos, 3));
    geo.setAttribute("color", new T.BufferAttribute(col, 3));
    g.add(new T.Points(geo, new T.PointsMaterial({
      size: 0.016, vertexColors: true, transparent: true, opacity,
      sizeAttenuation: true, depthWrite: false,
    })));
    if (G.cortex_faces && G.cortex_faces.length) {
      const sg = new T.BufferGeometry();
      sg.setAttribute("position", new T.BufferAttribute(pos.slice(0, CORTEX * 3), 3));
      sg.setAttribute("color", new T.BufferAttribute(col.slice(0, CORTEX * 3), 3));
      sg.setIndex(G.cortex_faces.flat());
      sg.computeVertexNormals();
      g.add(new T.Mesh(sg, new T.MeshBasicMaterial({
        vertexColors: true, transparent: true, opacity: opacity * 0.3,
        side: T.DoubleSide, depthWrite: false,
      })));
    }
    return g;
  }

  const scene = new T.Scene();
  const world = new T.Group(); scene.add(world);

  /* the implicit model: every site in its own atlas colour */
  const atlas = G.nodes.map((n) => new T.Color(n.c || "#888"));
  const implicit = makeBrain((i, c) => c.copy(atlas[i]), 0.95);
  world.add(implicit);

  /* the ring.  three models evenly spaced, tilted slightly out of plane so the diagram
     reads as a ring rather than as three brains in a row */
  const RING = 2.2, SUB = 0.44, FLAT = 0.82;
  const anchors = [];
  MODELS.forEach((M, k) => {
    const m = MATS[M.id];
    /* start at the TOP: beginning at -90deg put a model brain exactly where the hub
       label goes, and the two collided every frame. */
    const th = Math.PI / 2 + (k / MODELS.length) * Math.PI * 2;
    const grey = new T.Color("#6f7681");
    const b = makeBrain((i, c) => {
      const isIn = (m.inputs || []).some((x) => inRanges(x.nodes, i));
      const isOut = (m.outputs || []).some((x) => inRanges(x.nodes, i));
      if (isIn) return c.copy(IN);
      if (isOut) return c.copy(OUT);
      if (inRanges(m.hot || m.involved, i)) return c.copy(atlas[i]).lerp(grey, 0.35);
      return c.copy(grey);
    }, 0.55);
    b.scale.setScalar(SUB);
    b.position.set(Math.cos(th) * RING, Math.sin(th) * RING * FLAT, Math.sin(th * 2) * 0.26);
    world.add(b);
    anchors.push({ model: M, m, group: b, pos: b.position.clone() });
  });

  /* materialization arrows, centre -> each model */
  const arrowMat = new T.MeshBasicMaterial({ color: 0xf0b34a, transparent: true, opacity: 0.5 });
  anchors.forEach((a) => {
    const dir = a.pos.clone().normalize();
    const L = a.pos.length();
    const end = Math.max(L * 0.55, L - SUB * 1.35);
    const start = Math.min(L * 0.46, end - 0.18);
    const from = dir.clone().multiplyScalar(start);
    const to = dir.clone().multiplyScalar(end);
    const mid = from.clone().lerp(to, 0.5).add(new T.Vector3(0, 0, 0.18));
    const curve = new T.QuadraticBezierCurve3(from, mid, to);
    world.add(new T.Mesh(new T.TubeGeometry(curve, 26, 0.012, 8, false), arrowMat));
    const head = new T.Mesh(new T.ConeGeometry(0.05, 0.14, 14), arrowMat);
    head.position.copy(to);
    head.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), curve.getTangent(1).normalize());
    world.add(head);
  });

  /* ---- annotations, as projected HTML so they stay crisp and themeable ---- */
  const layer = document.createElement("div");
  layer.className = "mz-labels";
  host.appendChild(layer);
  const marks = [];
  /* a mark either sits in world space (the hub, the model names, the variables) or rides
     a brain (its inputs and outputs).  a riding mark stores LOCAL coordinates and is
     transformed by its brain's matrix each frame, so it follows when that brain is turned. */
  const addMark = (p, cls, html, group, ride) => {
    const el = document.createElement("div");
    el.className = "mz-mark " + cls;
    el.innerHTML = html;
    layer.appendChild(el);
    marks.push({ p: p.clone(), el, group, ride: ride || null });
  };

  /* the centre: a sample of the variables it carries, spread ACROSS fields rather than
     taken in order, which would give every component of one field and none of the rest */
  const R = window.IBM_REGISTRY;
  if (R && R.fields) {
    const all = [];
    R.fields.forEach((f) => (f.components || []).forEach((c) => all.push(c.id)));
    const step = Math.max(1, Math.floor(all.length / 6));
    all.filter((_, i) => i % step === 0).slice(0, 6).forEach((id, i, arr) => {
      /* well inside the ring: at the ring radius these landed on the model brains */
      const th = (i / arr.length) * Math.PI * 2 + 0.4;
      addMark(new T.Vector3(Math.cos(th) * 0.86, Math.sin(th) * 0.74, 0.12), "is-var", id, null, null);
    });
  }
  addMark(new T.Vector3(0, -1.12, 0), "is-hub", "<b>IBM-1</b><span>one parameter set · every variable</span>");

  anchors.forEach((a) => {
    /* OUTWARD from the hub, not below: "below" points back at the centre for the model
       at the top of the ring, and its name landed on the implicit brain. */
    const out = a.pos.clone().normalize();
    addMark(a.pos.clone().addScaledVector(out, SUB * 1.9), "is-name",
      `<b>${a.model.name}</b><span>${a.model.id}</span>`);
    const io = (list, cls, tag) => (list || []).slice(0, 2).forEach((x) => {
      const local = new T.Vector3().fromArray(x.anchor).sub(ctr).multiplyScalar(S);
      addMark(local, cls, `<b>${x.label}</b><span>${tag}</span>`, a.model.id, a.group);
    });
    io(a.m.inputs, "is-in", "in");
    io(a.m.outputs, "is-out", "out");
  });

  /* ---- camera, render ---- */
  const cam = new T.PerspectiveCamera(30, 1, 0.1, 60);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const canvasHost = document.createElement("div");
  canvasHost.className = "mz-canvas";
  host.insertBefore(canvasHost, layer);
  canvasHost.appendChild(renderer.domElement);

  /* the camera does not move.  each brain turns on its own axis, and a drag turns
     whichever brain it started nearest -- the diagram is four objects, not one scene the
     reader spins. */
  const dist = 7.4;
  const spinners = [{ g: implicit, r: 1.0 }].concat(anchors.map((a) => ({ g: a.group, r: SUB })));
  spinners.forEach((s2) => { s2.ry = 0; s2.rx = 0; s2.auto = 1; });
  let drag = null;
  function place() {
    cam.position.set(0, 0.9, dist);
    cam.lookAt(0, 0, 0);
  }
  /* pick by projected distance: a raycast would need colliders on a point cloud */
  function pick(cx, cy) {
    const b = canvasHost.getBoundingClientRect();
    const x = cx - b.left, y = cy - b.top;
    let best = null, bd = Infinity;
    spinners.forEach((s2) => {
      const v = s2.g.position.clone().project(cam);
      const px = (v.x + 1) / 2 * b.width, py = (-v.y + 1) / 2 * b.height;
      const d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = s2; }
    });
    return bd < 190 ? best : null;
  }
  function resize() {
    const w = host.clientWidth, h = Math.max(380, Math.round(w * 0.58));
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }
  canvasHost.addEventListener("pointerdown", (e) => {
    const s2 = pick(e.clientX, e.clientY);
    if (!s2) return;
    drag = { x: e.clientX, y: e.clientY, ry: s2.ry, rx: s2.rx, s: s2 };
    s2.auto = 0;
    canvasHost.setPointerCapture(e.pointerId);
  });
  canvasHost.addEventListener("pointermove", (e) => {
    if (!drag) return;
    drag.s.ry = drag.ry + (e.clientX - drag.x) * 0.009;
    drag.s.rx = Math.max(-0.9, Math.min(0.9, drag.rx + (e.clientY - drag.y) * 0.006));
  });
  const up = () => { drag = null; };
  canvasHost.addEventListener("pointerup", up);
  canvasHost.addEventListener("pointercancel", up);

  let vis = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; }, { rootMargin: "250px" })
      .observe(host);

  const t0 = performance.now();
  function frame() {
    requestAnimationFrame(frame);
    if (!vis) return;
    const tt = (performance.now() - t0) / 1000;
    spinners.forEach((s2, i) => {
      if (s2.auto) s2.ry = Math.sin(tt / 7 + i * 1.7) * 0.5;
      s2.g.rotation.set(s2.rx, s2.ry, 0);
    });
    place();
    renderer.render(scene, cam);
    const w = host.clientWidth, h = renderer.domElement.clientHeight;
    const put = (m, x, y, vz) => {
      m.el.style.opacity = vz < 1 ? "1" : "0";
      m.el.style.transform = `translate(${x.toFixed(0)}px, ${y.toFixed(0)}px)`;
    };
    const byGroup = new Map();
    marks.forEach((m) => {
      const wp = m.ride ? m.ride.localToWorld(m.p.clone()) : m.p.clone();
      const v = wp.project(cam);
      const e = { m, x: (v.x + 1) / 2 * w, y: (-v.y + 1) / 2 * h, z: v.z };
      if (m.group == null) return put(m, e.x, e.y, e.z);
      if (!byGroup.has(m.group)) byGroup.set(m.group, []);
      byGroup.get(m.group).push(e);
    });
    byGroup.forEach((list) => {
      list.sort((a, b) => a.y - b.y);
      for (let i = 1; i < list.length; i++)
        /* 22, not 15: these labels are two lines (name over IN/OUT), so a 15px gap
           still let the second line of one sit under the first line of the next. */
        if (list[i].y - list[i - 1].y < 22) list[i].y = list[i - 1].y + 22;
      list.forEach((e) => put(e.m, e.x, e.y, e.z));
    });
  }
  resize();
  window.addEventListener("resize", resize);
  frame();
  host.classList.add("is-ready");
})();
